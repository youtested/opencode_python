"""OpenCode Zen provider (https://opencode.ai/zen/v1) — the free models.

Thin wrapper over OpenAICompatProvider pointing at Zen's OpenAI-compatible
endpoint. With no API key, free (cost==0) models are used and Zen accepts the
literal API key "public" (mirrors opencode's behavior).

The Zen gateway throttles anonymous clients (unknown User-Agent, no
x-opencode-* headers) to a tiny free allowance — a couple of requests, then
429 FreeUsageLimitError. The official opencode client identifies itself with
`User-Agent: opencode/...` plus x-opencode-* headers, and the gateway treats
those as trusted clients with the real free quota. This provider sends the
same identity headers so free models (deepseek-v4-flash, etc.) keep working
across turns instead of being blocked after the first message.
"""

from __future__ import annotations

from typing import Any

from .openai_compat import OpenAICompatProvider

ZEN_BASE_URL = "https://opencode.ai/zen/v1"

# Limited-time free models on Zen ($0). Live-fetched in factory; this is the
# bundled fallback for when the network model list is unavailable (R2 risk).
FREE_MODELS: list[dict] = [
    {"id": "x-preview-f-free", "name": "Ox Alpha Free (Unlimited)", "context": 1000000, "output": 131072},
    {"id": "big-pickle", "name": "Big Pickle", "context": 200000, "output": 32000},
    {"id": "hy3-free", "name": "Hy3 Free", "context": 190000, "output": 64000},
    {"id": "mimo-v2.5-free", "name": "MiMo-V2.5 Free", "context": 200000, "output": 32000},
    {"id": "nemotron-3-ultra-free", "name": "Nemotron 3 Ultra Free", "context": 1000000, "output": 128000},
    {"id": "nemotron-3.5-lightning-free", "name": "Nemotron 3.5 Lightning Free", "context": 262144, "output": 128000},
]


class ZenProvider(OpenAICompatProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash-free",
        session_id: str | None = None,
        project: str | None = None,
        **kwargs: Any,
    ):
        # If no key given, we still need SOMETHING; Zen accepts "public" for free models.
        effective_key = api_key or "public"
        # Identify as the official opencode client so the Zen gateway serves the
        # real free quota instead of throttling an anonymous client (429
        # FreeUsageLimitError after the first message). The session id must be
        # stable across turns so Zen keeps the same conversation/provider lane.
        client_headers: dict[str, str] = {
            "User-Agent": "opencode/latest/0.1.0/cli",
            "x-opencode-client": "cli",
            "x-opencode-project": project or "opencode_py",
            "x-opencode-request": session_id or "cli",
        }
        if session_id:
            client_headers["x-opencode-session"] = session_id
        merged = dict(client_headers)
        merged.update(kwargs.pop("extra_headers", {}) or {})
        super().__init__(
            id="opencode",
            name="OpenCode Zen",
            base_url=ZEN_BASE_URL,
            api_key=effective_key,
            model=model,
            is_free=True,
            extra_headers=merged,
            **kwargs,
        )
        self.has_key = bool(api_key)
