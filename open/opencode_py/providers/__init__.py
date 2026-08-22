"""Provider registry and factory.

Lightweight by design (low-RAM armv7 phones): importing this package does NOT
pull in httpx or the provider backends. Each name is resolved lazily through
PEP 562 ``__getattr__`` the first time it is referenced, so a bare
``opencode_py --print-config`` / ``--version`` / TUI boot never loads the
~20 MB HTTP stack until a network call actually needs it.
"""

from __future__ import annotations

import importlib

__all__ = [
    "Provider",
    "ProviderError",
    "ProviderEvent",
    "RateLimitError",
    "ContextOverflowError",
    "StreamInterrupted",
    "ToolCall",
    "Usage",
    "tool_to_openai_schema",
    "OpenAICompatProvider",
    "AnthropicProvider",
    "ZenProvider",
    "FREE_MODELS",
    "ZEN_BASE_URL",
    "OllamaProvider",
    "FREE_PROVIDERS",
    "FREE_DEFAULT_MODELS",
    "PAID_PROVIDERS",
    "Rotation",
    "build_provider",
    "build_rotation",
    "fetch_zen_models",
    "fetch_openrouter_models",
    "fetch_live_models",
    "check_provider",
    "model_context_size",
    "model_output_limit",
]

# name -> dotted submodule relative to this package. Keep it as the single
# source of truth so __getattr__ and star-imports stay in sync.
_LAZY_SOURCES: dict[str, str] = {
    # .base (light, no third-party deps): kept *eager* below is fine, but route
    # through the same table so the mapping is obvious.
    "Provider": ".base",
    "ProviderError": ".base",
    "ProviderEvent": ".base",
    "RateLimitError": ".base",
    "ContextOverflowError": ".base",
    "StreamInterrupted": ".base",
    "ToolCall": ".base",
    "Usage": ".base",
    "tool_to_openai_schema": ".base",
    "OpenAICompatProvider": ".openai_compat",
    "AnthropicProvider": ".anthropic",
    "ZenProvider": ".zen",
    "FREE_MODELS": ".zen",
    "ZEN_BASE_URL": ".zen",
    "OllamaProvider": ".ollama",
    "FREE_PROVIDERS": ".rotation",
    "FREE_DEFAULT_MODELS": ".rotation",
    "PAID_PROVIDERS": ".rotation",
    "Rotation": ".rotation",
    "build_provider": ".rotation",
    "build_rotation": ".rotation",
    "fetch_zen_models": ".rotation",
    "fetch_openrouter_models": ".rotation",
    "fetch_live_models": ".rotation",
    "check_provider": ".rotation",
    "model_context_size": ".rotation",
    "model_output_limit": ".rotation",
}

_loaded: dict[str, bool] = {}


def __getattr__(name: str):
    mod = _LAZY_SOURCES.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if not _loaded.get(mod):
        importlib.import_module(mod, __name__)
        _loaded[mod] = True
    return getattr(importlib.import_module(mod, __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SOURCES))