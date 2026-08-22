"""Tool registry: name -> Tool dataclass with JSON schema + run().

Mirrors opencode's Tool.define pattern. Each tool declares a name, description,
parameter JSON schema, an optional permission key, and a run(input) -> dict.
The run result dict carries `output` (text), plus optional metadata the TUI
renders (e.g. edit diff, bash exit code).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    permission: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.mcp_servers: list[Any] = []
        # In-flight HTTP responses opened by the webfetch tools on this engine.
        # The interrupt path (engine.abort) calls abort_fetches() to force-close
        # them, waking a blocked socket read immediately so ESC/Ctrl+C aborts a
        # running fetch without waiting for its timeout.
        self._fetch_lock = threading.Lock()
        self._active_fetches: list[Any] = []

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for all registered tools."""
        out = []
        for tool in self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _provider_schema(tool.parameters),
                    },
                }
            )
        return out

    def register_fetch(self, resp: Any) -> None:
        """Track an in-flight webfetch response so an interrupt can close it."""
        with self._fetch_lock:
            self._active_fetches.append(resp)

    def unregister_fetch(self, resp: Any) -> None:
        """Stop tracking a fetch response (called when its read finishes)."""
        with self._fetch_lock:
            try:
                self._active_fetches.remove(resp)
            except ValueError:
                pass

    def abort_fetches(self) -> None:
        """Force-close every in-flight fetch this engine is running.

        Mirrors the provider-stream abort: ``socket.shutdown(SHUT_RDWR)`` wakes
        a reader blocked in ``iter_bytes`` so the tool sees the interrupt (the
        interrupt flag is already set by the caller) and returns immediately.
        """
        from ..util.net import force_close_response

        with self._fetch_lock:
            responses = list(self._active_fetches)
            self._active_fetches.clear()
        for resp in responses:
            try:
                force_close_response(resp)
            except Exception:
                pass

    def close(self) -> None:
        """Release the MCP server processes this registry holds.

        Servers are shared process-wide and reference-counted: this drops each
        one's reference, terminating it when the last holding registry (or a
        finished sub-agent) releases it. Safe to call more than once.
        """
        servers = list(self.mcp_servers)
        self.mcp_servers.clear()
        for server in servers:
            release = getattr(server, "release", None)
            if release is not None:
                try:
                    release()
                except Exception:  # pragma: no cover - best effort at teardown
                    pass


def _param(
    type_: str,
    description: str,
    required: bool = True,
    enum: list[str] | None = None,
    default: Any = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": type_, "description": description}
    if enum is not None:
        schema["enum"] = enum
    if default is not None:
        schema["default"] = default
    if not required:
        schema["optional"] = True
    return schema


def schema_with(params: dict[str, dict], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": params, "required": required}


def _provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a tool schema for a provider, dropping non-standard keywords.

    ``schema_with``/``_param`` add an ``"optional": True`` hint for the TUI
    (and tool builders); it is NOT part of JSON Schema, and strict provider
    tool-schema validators (Anthropic, some OpenAI-compatible gateways) reject
    unknown keywords. Return a copy so the stored tool definition keeps its
    hint while the wire format stays valid.
    """
    if not isinstance(schema, dict):
        return schema
    clean: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "optional":
            continue
        if key == "properties" and isinstance(value, dict):
            clean[key] = {
                name: _provider_schema(prop) if isinstance(prop, dict) else prop
                for name, prop in value.items()
            }
        else:
            clean[key] = value
    return clean
