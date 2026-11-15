"""Opt-in native operations for Pascal GPUs with SGLang 0.4.9.post6.

This trades optimized kernels for compatibility; it is not a performance baseline.
Only dense text models using the standard RMSNorm/rotary/activation ops are targeted.
"""


def install_native_ops():
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
    # Remove only that known guard for this explicit FP32/native path, without
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
    logging.getLogger(__name__).warning(
        "Autellix legacy GPU mode: FP32/native PyTorch operations; "
        "optimized SGLang GPU kernels are bypassed for dense text models."
    )
