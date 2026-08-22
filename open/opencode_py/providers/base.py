"""Provider protocol + streaming events.

Every provider implements `stream_chat(messages, tools, on_event)`. Events
flow to a callback so the same engine drives both the TUI and headless mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, accumulated across deltas
    index: int = 0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderEvent:
    kind: str  # text_delta | reasoning_delta | tool_call | usage | error | done
    text: str = ""
    tool_call: ToolCall | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    error: str = ""
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status = status


class RateLimitError(ProviderError):
    def __init__(self, message: str = "rate limit", *, retry_after: float | None = None, status: int | None = 429):
        super().__init__(message, retryable=True, status=status)
        self.retry_after = retry_after


class ContextOverflowError(ProviderError):
    """The model's context window was exceeded.

    Distinct from a hard provider error: the caller (agent loop) can trim the
    history and retry instead of surfacing a scary failure to the user.
    """

    def __init__(self, message: str = "context length exceeded", *, status: int | None = None):
        super().__init__(message, retryable=True, status=status)


class StreamInterrupted(Exception):
    """The user aborted an in-flight stream (e.g. Esc).

    Raised inside a provider's SSE iteration as soon as the interrupt flag
    flips. It is NOT a provider failure: the agent loop must not retry or
    rotate, it just ends the turn as interrupted.
    """


class Provider(Protocol):
    id: str
    name: str
    is_free: bool

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_event: Callable[[ProviderEvent], None],
        **kwargs: Any,
    ) -> None: ...


def tool_to_openai_schema(tool) -> dict[str, Any]:
    """Convert a registry Tool to an OpenAI function-calling JSON schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
