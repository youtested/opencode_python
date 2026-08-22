"""Tools package: registry builder mirroring opencode's builtin tool order."""

from __future__ import annotations

import atexit
import threading
from typing import Any

from ..config import Config
from . import bash as bash_mod
from . import edit as edit_mod
from . import glob as glob_mod
from . import grep as grep_mod
from . import question as question_mod
from . import read as read_mod
from . import remember as remember_mod
from . import summarize_file as summarize_file_mod
from . import task as task_mod
from . import todo as todo_mod
from . import webfetch as webfetch_mod
from . import write as write_mod
from .registry import Registry, Tool, schema_with

TOOL_NAMES = ["bash", "read", "glob", "grep", "edit", "write", "webfetch", "webfetch_many", "todowrite", "task", "question", "remember", "summarize_file"]

# Process-wide cache of running MCP servers keyed by (name, command, args).
# build_registry() runs once per engine AND once per spawned sub-agent; without
# sharing, N sub-agents would spawn N+1 server processes per server that were
# never reaped. A cache makes every registry reuse ONE process per server, and
# MCPServer's reference counting terminates it when the last registry closes it.
_MCP_SERVER_CACHE: dict[tuple, Any] = {}
_MCP_CACHE_LOCK = threading.Lock()


def _mcp_server_get(name: str, command: str, args: list) -> Any:
    """Return a cached (running or respawnable) MCP server, creating it on
    first use."""
    from .mcp import MCPServer

    key = (str(name), command, tuple(args))
    with _MCP_CACHE_LOCK:
        server = _MCP_SERVER_CACHE.get(key)
        if server is None:
            server = MCPServer(name=str(name), command=command, args=args)
            _MCP_SERVER_CACHE[key] = server
        return server


def _mcp_server_close_all() -> None:
    """Terminate every cached MCP server process (atexit + engine teardown)."""
    with _MCP_CACHE_LOCK:
        servers = list(_MCP_SERVER_CACHE.values())
        _MCP_SERVER_CACHE.clear()
    for server in servers:
        try:
            server.close()
        except Exception:  # pragma: no cover - best effort at process exit
            pass


atexit.register(_mcp_server_close_all)


def build_registry(cfg: Config | None = None) -> Registry:
    cfg = cfg or Config()
    registry = Registry()
    state: dict = {}

    registry.register(bash_mod.tool(
        max_lines=cfg.tool_output_max_lines,
        max_bytes=cfg.tool_output_max_bytes,
        default_timeout=cfg.bash_default_timeout,
        registry=registry,
    ))
    registry.register(read_mod.tool())
    registry.register(glob_mod.tool())
    registry.register(grep_mod.tool())
    registry.register(edit_mod.tool())
    registry.register(write_mod.tool())
    registry.register(webfetch_mod.tool(registry=registry))
    registry.register(webfetch_mod.batch_tool(registry=registry))
    registry.register(todo_mod.tool(state))
    registry.register(task_mod.tool(registry))
    registry.register(question_mod.tool(registry))
    registry.register(remember_mod.tool())
    registry.register(summarize_file_mod.tool())

    # config-driven tool toggles: tools.<name> = false removes it (opencode behavior)
    enabled: dict | None = None
    if cfg and cfg.raw:
        enabled = cfg.raw.get("tools")
    if enabled:
        for name in list(registry.names()):
            if enabled.get(name) is False:
                registry._tools.pop(name, None)

    _load_plugins(registry, cfg)
    _load_mcp_servers(registry, cfg)

    return registry


def _load_plugins(registry: Registry, cfg: Config | None) -> None:
    """Plugin-lite: config key "plugins": ["my.tools.module"] where the module
    exposes TOOLS = [{name, description, parameters, run}, ...]."""
    raw = (cfg and cfg.raw) or {}
    import importlib
    import sys
    from pathlib import Path

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    for mod_path in raw.get("plugins", []) or []:
        try:
            mod = importlib.import_module(str(mod_path))
            for tool_def in getattr(mod, "TOOLS", []) or []:
                registry.register(
                    Tool(
                        name=tool_def["name"],
                        description=tool_def.get("description", ""),
                        parameters=tool_def.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                        run=tool_def["run"],
                    )
                )
        except Exception as e:
            registry.register(
                Tool(
                    name=f"plugin:{mod_path}",
                    description="plugin failed to load",
                    parameters={"type": "object", "properties": {}},
                    run=lambda args, err=e, mod=mod_path: {
                        "output": f"plugin {mod} failed to load: {err}",
                        "error": True,
                    },
                )
            )


def _load_mcp_servers(registry: Registry, cfg: Config | None) -> None:
    """MCP-lite: config key "mcpServers": {name: {command, args}} -> tools named
    mcp__<name>__<tool>.

    Server processes are SHARED process-wide (see the module cache above): a
    parent registry and every sub-agent registry reuse the SAME process instead
    of spawning one per registry build, and each registry holds a reference via
    ``server.acquire()`` that ``Registry.close()`` returns so the process dies
    when its last user is done.
    """
    raw = (cfg and cfg.raw) or {}
    servers = raw.get("mcpServers", {}) or {}
    if not servers:
        return
    from .mcp import MCPError

    for sname, spec in servers.items():
        command = spec.get("command")
        args = spec.get("args") or []
        if not command:
            continue
        server = _mcp_server_get(sname, command, args)
        server.acquire()
        registry.mcp_servers.append(server)
        try:
            remote_tools = server.list_tools()
        except MCPError as e:
            server.release()
            registry.mcp_servers.remove(server)
            registry.register(
                Tool(
                    name=f"mcp:{sname}",
                    description=f"mcp server {sname} failed to start",
                    parameters={"type": "object", "properties": {}},
                    run=lambda args, err=str(e): {"output": f"mcp {sname}: {err}", "error": True},
                )
            )
            continue
        for t in remote_tools:
            if not isinstance(t, dict):
                continue
            # `tools/list` is server-controlled: a malformed item (missing
            # "name") used to raise KeyError here and crash build_registry —
            # i.e. every engine and every sub-agent — for a config typo on one
            # remote server. Skip nameless tools instead of taking down startup.
            remote_name = t.get("name")
            if not remote_name:
                continue
            registry.register(
                Tool(
                    name=f"mcp__{sname}__{remote_name}",
                    description=t.get("description") or f"{sname}: {remote_name}",
                    parameters=t.get("inputSchema") or {"type": "object", "properties": {}},
                    run=_mcp_run(server, remote_name),
                )
            )


def _mcp_run(server, remote_name: str):
    def run(arguments: dict) -> dict[str, Any]:
        try:
            return server.run_tool(remote_name, arguments)
        except Exception as e:
            return {"output": f"mcp tool {remote_name} failed: {e}", "error": True}

    return run


__all__ = [
    "Registry",
    "Tool",
    "schema_with",
    "build_registry",
    "TOOL_NAMES",
]
