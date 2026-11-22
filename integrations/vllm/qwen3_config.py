"""Register local Qwen3 checkpoints with the pinned vLLM engine."""
import json
from pathlib import Path


def register_local_qwen3(model, tensor_parallel_size):
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_bytes())
    if config.get("model_type") != "qwen3":
        return
    if tensor_parallel_size != 1:
        raise ValueError("Qwen3 on vLLM 0.6.1 requires one GPU")

    from transformers import AutoConfig, Qwen2Config
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.model_executor.models import ModelRegistry
    from .qwen3 import Qwen3ForCausalLM

    if "qwen3" not in CONFIG_MAPPING:
        class Qwen3Config(Qwen2Config):
            model_type = "qwen3"

        AutoConfig.register("qwen3", Qwen3Config)
    ModelRegistry.register_model("Qwen3ForCausalLM", Qwen3ForCausalLM)
