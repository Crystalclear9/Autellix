"""Opt-in native operations for Pascal GPUs with SGLang 0.4.9.post6.

Both comparison arms use these native PyTorch operations on Pascal.
Only dense text models using the standard RMSNorm/rotary/activation ops are targeted.
"""


def install_native_ops(prefix_attention=True, attention_score_elements=2**26):
    import ast
    import inspect
    import logging
    import textwrap
    import torch._dynamo
    from sglang.srt.custom_op import CustomOp
    from sglang.srt.layers import activation, layernorm, rotary_embedding

    # Native functions can also carry torch.compile decorators. Disable Dynamo
    # in this scheduler process so they do not re-enter unsupported Triton code.
    torch._dynamo.config.disable = True
    for module in (activation, layernorm, rotary_embedding):
        for cls in vars(module).values():
            if (isinstance(cls, type) and issubclass(cls, CustomOp)
                    and cls is not CustomOp
                    and cls.forward_native is not CustomOp.forward_native):
                cls.forward_cuda = cls.forward_native

    # The pinned runner rejects Pascal before loading weights and forces FP16.
    # Remove only that known guard for this explicit native path, without
    # spoofing device capability or modifying files in the installed package.
    from sglang.srt.model_executor.model_runner import ModelRunner
    original = ModelRunner.load_model
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    guards = [node for node in function.body if isinstance(node, ast.If)
              and "SGLang only supports sm75 and above." in ast.unparse(node)]
    if len(guards) != 1:
        raise RuntimeError("unexpected SGLang load_model; legacy GPU patch refused")
    function.body.remove(guards[0])
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename, "exec"),
         original.__globals__, namespace)
    ModelRunner.load_model = namespace[original.__name__]
    if prefix_attention:
        from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
        from . import attention
        attention.MAX_SCORE_ELEMENTS = attention_score_elements
        TorchNativeAttnBackend._run_sdpa_forward_extend = attention.prefix_attention
    install_pipeline_embeddings()
    logging.getLogger(__name__).warning(
        "Autellix legacy GPU mode: native PyTorch operations; "
        "optimized SGLang GPU kernels are bypassed for dense text models."
    )


def install_pipeline_embeddings():
    """Load tied Llama output embeddings on the pipeline's last stage."""
    from sglang.srt.models.llama import LlamaForCausalLM, ParallelLMHead
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    original_init = LlamaForCausalLM.__init__
    original_load = LlamaForCausalLM.load_weights

    def init(self, config, quant_config=None, prefix=""):
        original_init(self, config, quant_config, prefix)
        self._autellix_tied_pp_head = (self.pp_group.world_size > 1
                                      and self.pp_group.is_last_rank and config.tie_word_embeddings)
        if self._autellix_tied_pp_head:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                          quant_config=quant_config, prefix="lm_head")

    def load(self, weights):
        def with_tied_head():
            for name, weight in weights:
                if self._autellix_tied_pp_head and name == "model.embed_tokens.weight":
                    parameter = self.lm_head.weight
                    getattr(parameter, "weight_loader", default_weight_loader)(parameter, weight)
                yield name, weight
        return original_load(self, with_tied_head())

    LlamaForCausalLM.__init__, LlamaForCausalLM.load_weights = init, load


def run_native_scheduler_process(*args, prefix_attention=True, attention_score_elements=2**26, **kwargs):
    """P40 kernels only; leave SGLang's scheduling and KV management untouched."""
    install_native_ops(prefix_attention=prefix_attention, attention_score_elements=attention_score_elements)
    from importlib import import_module
    module = import_module("sglang.srt.managers.scheduler")
    from .profiling import install
    install(module.Scheduler)
    return module.run_scheduler_process(*args, **kwargs)
