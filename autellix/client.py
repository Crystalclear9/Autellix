from __future__ import annotations

from typing import Any

from .response import SimulatedChatResponse
from .service import AutellixService


class _ChatCompletions:
    def __init__(self, service: AutellixService) -> None:
        self._service = service

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        program_id: str | None = None,
        call_id: str | None = None,
        model: str | None = None,
        parents: tuple[str, ...] = (),
        max_tokens: int | None = None,
        model_time: int | None = None,
        prefill_tokens: int | None = None,
        decode_tokens: int | None = None,
        release_delay: int = 0,
        **kwargs: Any,
    ) -> SimulatedChatResponse:
        if session_id is None:
            session = self._service.start_session(program_id)
            session_id = session.session_id
        metadata = dict(kwargs)
        if model is not None:
            metadata["model"] = model
        return self._service.chat_completion(
            session_id,
            messages,
            call_id=call_id,
            model_time=model_time,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            parents=parents,
            release_delay=release_delay,
            max_tokens=max_tokens,
            **metadata,
        )


class _Chat:
    def __init__(self, service: AutellixService) -> None:
        self.completions = _ChatCompletions(service)


class AutellixClient:
    """Minimal OpenAI/vLLM-shaped client for simulated chat completions."""

    def __init__(self, service: AutellixService | None = None, **service_kwargs: Any) -> None:
        self.service = service or AutellixService(**service_kwargs)
        self.chat = _Chat(self.service)
