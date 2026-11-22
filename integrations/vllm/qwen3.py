"""Dense Qwen3 with vLLM 0.6.1's Qwen2 execution and weight loader."""
from torch import nn
from vllm.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2MLP, Qwen2Model


class Qwen3Attention(nn.Module):
    def __init__(self, config, cache_config, quant_config):
        super().__init__()
        heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = heads * self.head_dim
        self.kv_size = kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim, heads, kv_heads,
                                          bias=config.attention_bias, quant_config=quant_config)
        self.o_proj = RowParallelLinear(self.q_size, config.hidden_size, bias=False, quant_config=quant_config)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(self.head_dim, self.head_dim, config.max_position_embeddings,
                                   config.rope_theta, rope_scaling=config.rope_scaling)
        self.attn = Attention(heads, self.head_dim, self.head_dim ** -0.5, num_kv_heads=kv_heads,
                              cache_config=cache_config, quant_config=quant_config)

    def forward(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim).contiguous()).view_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim).contiguous()).view_as(k)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v, kv_cache, attn_metadata))
        return output


class Qwen3DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, cache_config, quant_config):
        nn.Module.__init__(self)
        self.self_attn = Qwen3Attention(config, cache_config, quant_config)
        self.mlp = Qwen2MLP(config.hidden_size, config.intermediate_size, config.hidden_act, quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3Model(Qwen2Model):
    def __init__(self, config, cache_config, quant_config):
        nn.Module.__init__(self)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.start_layer, self.end_layer = 0, config.num_hidden_layers
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config, cache_config, quant_config)
                                    for _ in range(self.end_layer))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config, cache_config=None, quant_config=None, lora_config=None):
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.model = Qwen3Model(config, cache_config, quant_config)
        self.lm_head = self.model.embed_tokens if config.tie_word_embeddings else ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
