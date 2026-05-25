from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AutellixRequestMetadata:
    program_id: str
    call_id: str
    thread_id: str | None = None
    parents: tuple[str, ...] = ()
    framework_metadata: dict[str, Any] = field(default_factory=dict)


class VLLMUnavailableError(RuntimeError):
    pass


def require_vllm():
    try:
        import vllm  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise VLLMUnavailableError(
            "vLLM is not installed. Install vLLM v0.6.1 in a Linux/WSL CUDA "
            "environment before using this integration."
        ) from exc
    return vllm


class AutellixVLLMAdapter:
    """Adapter shape for future vLLM scheduler integration."""

    def __init__(self, *, require_backend: bool = False) -> None:
        self.vllm = require_vllm() if require_backend else None

    def annotate_request(self, request: Any, metadata: AutellixRequestMetadata) -> Any:
        setattr(request, "autellix_program_id", metadata.program_id)
        setattr(request, "autellix_call_id", metadata.call_id)
        setattr(request, "autellix_thread_id", metadata.thread_id)
        setattr(request, "autellix_parents", metadata.parents)
        setattr(request, "autellix_metadata", dict(metadata.framework_metadata))
        return request
