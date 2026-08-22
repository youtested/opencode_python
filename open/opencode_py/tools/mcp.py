"""Minimal MCP (Model Context Protocol) stdio client.

Speaks newline-delimited JSON-RPC 2.0 over the child's stdin/stdout (the
transport used by the reference MCP servers, e.g. `npx -y @modelcontextprotocol/...`).
Remote tools are exposed to the model as `mcp__<server>__<tool>`.

Pure-Python + subprocess; safe for armv7. No SDK dependency.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import threading
import time
from typing import Any


class MCPError(Exception):
    pass


class MCPServer:
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        timeout: float = 20.0,
        tool_timeout: float | None = None,
    ):
        self.name = name
        self.command = command
        self.args = list(args)
        self.timeout = timeout
        # Handshake calls (initialize/tools/list) are fast; a remote `tools/call`
        # can legitimately run a long operation (DB query, script) and the same
        # 20s budget produced false timeouts on healthy servers. Give tool calls
        # their own, larger budget (still a hard guard against a hung server).
        self.tool_timeout = tool_timeout if tool_timeout is not None else max(timeout, 60.0)
        self.proc: subprocess.Popen | None = None
        self._counter = 0
        # Persistent read buffer: one os.read() chunk can carry several newline-
        # delimited JSON messages (e.g. a notification coalesced with the
        # real response). _read_line consumes one line and keeps the rest here,
        # instead of discarding anything after the first newline (which silently
        # lost responses -> false timeouts).
        self._rbuf = b""
        # Serialize call() so a shared server is safe across parallel workers /
        # parent + sub-agent threads (each call sends one request and reads
        # until its matching id).
        self._lock = threading.Lock()
        # Reference count held by registries sharing this server; the process
        # is terminated when the last user releases (see acquire/release).
        # Guarded by _ref_lock so a `release()` from one thread can't race an
        # `acquire()` from another into a premature close / negative count.
        self._users = 0
        self._ref_lock = threading.Lock()

    # -- low level --------------------------------------------------------
    def _ensure_started(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        try:
            self.proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise MCPError(f"mcp/{self.name}: could not start '{self.command}': {e}") from e
        try:
            self.call(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "opencode_py", "version": "0.1.0"},
                },
            )
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except MCPError:
            self.close()
            raise

    def _send(self, payload: dict) -> None:
        if not self.proc or not self.proc.stdin:
            raise MCPError(f"mcp/{self.name}: not running")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _read_line(self, timeout: float | None = None) -> dict:
        if not self.proc or not self.proc.stdout:
            raise MCPError(f"mcp/{self.name}: not running")
        # Read one line with a hard wall-clock deadline. A buffered TextIOWrapper
        # defeats `select` on the wrapper (it buffers internally while the OS
        # pipe looks empty), so read from the raw fd instead using a small chunk
        # and track bytes ourselves. `select` guarantees some bytes are ready.
        # Data spanning multiple messages stays in `self._rbuf`; a chatty server
        # coalescing a notification + response into one read must NOT lose the
        # second message (it used to be discarded after the first newline).
        # The deadline is the budget for THIS line, sized by the caller (fast
        # handshake vs. potentially-slow tools/call).
        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        fd = self.proc.stdout.fileno()
        while True:
            newline = self._rbuf.find(b"\n")
            if newline == -1:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError(f"mcp/{self.name}: timeout waiting for response")
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    raise MCPError(f"mcp/{self.name}: timeout waiting for response")
                try:
                    data = os.read(fd, 65536)
                except (OSError, ValueError) as e:
                    raise MCPError(f"mcp/{self.name}: read failed: {e}") from e
                if not data:
                    if self._rbuf:
                        raise MCPError(f"mcp/{self.name}: server closed stdout mid-line")
                    raise MCPError(f"mcp/{self.name}: server closed stdout")
                self._rbuf += data
                if len(self._rbuf) > 8 * 1024 * 1024:
                    raise MCPError(f"mcp/{self.name}: response line too large")
                continue
            line = self._rbuf[:newline]
            self._rbuf = self._rbuf[newline + 1:]
            if not line.strip():
                # blank/whitespace separator lines are legal between JSON
                # messages; skip them instead of aborting the connection
                continue
            try:
                return json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as e:
                raise MCPError(f"mcp/{self.name}: bad response: {e}") from e

    def call(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        self._counter += 1
        rid = self._counter
        # One request at a time on this server: a shared server object can be
        # used by the parent engine and its sub-agents concurrently, and two
        # interleaved send/read pairs would hand each thread the other's reply.
        with self._lock:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            while True:
                msg = self._read_line(timeout=timeout)
                if msg.get("id") != rid:
                    continue  # notification / another request id
                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(f"mcp/{self.name}: {err.get('message', err)}")
                return msg.get("result") or {}

    # -- public API -------------------------------------------------------
    def list_tools(self) -> list[dict]:
        self._ensure_started()
        res = self.call("tools/list")
        return res.get("tools", []) or []

    def run_tool(self, remote_name: str, arguments: dict) -> dict[str, Any]:
        self._ensure_started()
        res = self.call("tools/call", {"name": remote_name, "arguments": arguments or {}}, timeout=self.tool_timeout)
        if "isError" in res and res.get("isError"):
            text = _content_text(res) or "mcp tool errored"
            return {"output": text, "error": True}
        text = _content_text(res)
        if text is None:
            text = json.dumps(res.get("content", res))
        return {"output": text}

    def acquire(self) -> None:
        """Record one registry/user of this server. The process is kept alive
        while at least one user holds a reference."""
        with self._ref_lock:
            self._users += 1

    def release(self) -> None:
        """Drop one user's reference; terminate the process when the last one
        goes away. Idempotent-ish and safe to call multiple times."""
        with self._ref_lock:
            if self._users > 0:
                self._users -= 1
            if self._users <= 0:
                self.close()

    def close(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        self._rbuf = b""


def _content_text(result: dict) -> str | None:
    """Concatenate MCP content blocks: text / resource / image(placeholder)."""
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "resource":
            resource = block.get("resource", {})
            if isinstance(resource, dict) and "text" in resource:
                parts.append(resource["text"])
    return "\n".join(p for p in parts if p) or None
