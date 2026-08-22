"""bash tool: run a shell command in a persistent shell session with timeout."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from io import UnsupportedOperation
from typing import Callable

from .registry import Registry, Tool, schema_with

READ_CHUNK = 65536


def _kill_proc_tree(proc: subprocess.Popen, pgid: int | None = None) -> None:
    """Kill the shell AND every process it spawned.

    The shell is started as a new session/process-group leader, so background
    grandchildren (`sleep 1000 &`) live in the same group as the shell; a bare
    `proc.kill()` only signals the shell PID and leaks the children, which keep
    holding the stdout pipe and the cwd hostage. Killing the group reaps them.

    ``pgid`` should be the shell's process-group id captured while the shell
    was still alive. Passing it by value matters: by the time we clean up, the
    shell PID is often already gone (the command finished), but its group id
    still exists as long as a stray background child keeps a membership — and
    `os.getpgid(proc.pid)` would then fail, losing the handle on that group.
    Fall back to resolving it live only when no pgid was given.

    Children that daemonize into their own session (`setsid`, ...) escape the
    shell's group. On Linux we catch those that are still direct children of
    the live shell (the timeout/interrupt paths) by scanning /proc by
    parentage; a fully double-forked daemon already reparented to init is out
    of scope. That sweep is skipped when the shell has already exited — its
    children have been reparented and nothing is left to catch, and killing the
    process group already handled any non-daemonized strays. Scanning /proc
    opens a file per process and is measurable overhead on every command, so
    the common (clean-exit) path must not pay it.
    """
    if sys.platform == "win32" or not hasattr(os, "killpg"):
        try:
            proc.kill()
        except Exception:
            pass
        return
    shell_alive = proc.poll() is None
    group_killed = False
    if pgid:
        try:
            os.killpg(pgid, signal.SIGKILL)
            group_killed = True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if not group_killed:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            group_killed = True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    # Catch daemonized (`setsid`) children that escaped the group while their
    # parent shell is still alive (timeout/interrupt paths). Once the shell has
    # exited those children have reparented, so the /proc parentage scan finds
    # nothing — skip it then.
    if shell_alive and not group_killed and os.path.isdir("/proc"):
        ppid = proc.pid
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open("/proc/%s/stat" % entry, "rb") as fh:
                        stat = fh.read()
                    # format: pid (comm) state ppid pgrp ...
                    rparen = stat.rfind(b")")
                    if rparen < 0 or rparen + 2 >= len(stat):
                        continue
                    rest = stat[rparen + 2 :].split()
                    if len(rest) >= 2 and rest[1].isdigit() and int(rest[1]) == ppid:
                        os.kill(int(entry), signal.SIGKILL)
                except Exception:
                    continue
        except Exception:
            pass
    if not group_killed:
        try:
            proc.kill()
        except Exception:
            pass


def _stream_output(
    proc: subprocess.Popen,
    deadline: float,
    max_bytes: int,
    is_interrupted: Callable[[], bool] | None = None,
) -> tuple[bytes, bool, str]:
    """Read the process output to a natural end, capping the captured bytes.

    Returns ``(raw, capped, end)`` where ``end`` is one of:

    - ``"shell_exited"`` — the main shell terminated (poll() non-None).
      This is the honest end of the command. The stdout pipe may still be open
      because a stray background child inherited it; we do NOT wait for that
      pipe's EOF, and we drain only what is already buffered. It is the
      caller's job to reap the group so the stray child dies too.
    - ``"eof"`` — the pipe reached EOF (every writer closed it).
    - ``"timeout"`` — the deadline was hit before the command finished.
    - ``"interrupted"`` — ``is_interrupted()`` became true mid-command.

    Uses the raw fd (not the buffered pipe object, whose internal buffer can
    hold data `select` doesn't see) so nothing gets stuck. The fd is set
    non-blocking so EOF (``os.read`` -> ``b""``) is detectable even while a
    background child keeps the pipe open; bytes beyond the cap are still
    drained (discarded) so the child never blocks on a full pipe and memory
    stays bounded regardless of output size.
    """
    chunks: list[bytes] = []
    captured = 0
    capped = False
    eof = False
    stdout = proc.stdout
    if stdout is None:
        return b"", False, "shell_exited"
    try:
        fd = stdout.fileno()
    except (OSError, ValueError, UnsupportedOperation):
        return b"", False, "shell_exited"
    try:
        os.set_blocking(fd, False)
    except OSError:
        pass

    def append(data: bytes) -> None:
        nonlocal captured, capped
        if not capped:
            room = max_bytes - captured
            if room > 0:
                chunks.append(data[:room])
                captured += len(data[:room])
            if len(data) > room:
                capped = True

    while True:
        # Honor a user interrupt the moment it lands (2nd ESC / ctrl+C flips
        # the shared flag the app wires into the engine's interrupt callback).
        if is_interrupted is not None:
            try:
                if is_interrupted():
                    return b"".join(chunks), capped, "interrupted"
            except Exception:
                pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return b"".join(chunks), capped, "timeout"
        if proc.poll() is not None:
            # The shell is done. The pipe may still be open because a
            # background grandchild inherited it: do NOT wait for that pipe's
            # EOF, or a finished command would block to the deadline and fake
            # a timeout. Drain what is already buffered and let the caller
            # reap the group (killing the stray child outright).
            if not eof:
                while True:
                    try:
                        data = os.read(fd, READ_CHUNK)
                    except BlockingIOError:
                        break
                    except OSError:
                        break
                    if not data:
                        eof = True
                        break
                    append(data)
            return b"".join(chunks), capped, "shell_exited"
        if eof:
            # Every writer closed the pipe, but the shell itself is still
            # alive (e.g. it is waiting on a child that redirected stdout).
            # Wait for the shell in small slices instead of re-selecting on a
            # permanently-readable fd (busy spin).
            try:
                proc.wait(timeout=min(remaining, 0.25))
            except subprocess.TimeoutExpired:
                continue
            return b"".join(chunks), capped, "shell_exited"
        # Wait for data (in short slices so the deadline stays live). This runs
        # whether or not the main shell has exited: a background child that
        # keeps the pipe open can still write after the shell is done. Status
        # changes are detected at the top of the loop, not here.
        ready, _, _ = select.select([fd], [], [], min(remaining, 0.25))
        if not ready:
            continue
        try:
            data = os.read(fd, READ_CHUNK)
        except BlockingIOError:
            # select reported ready but the read raced an empty pipe; retry
            continue
        except OSError:
            return b"".join(chunks), capped, "eof"
        if not data:
            eof = True
            continue
        append(data)


def _bash(
    command: str,
    timeout: int = 120,
    workdir: str | None = None,
    max_lines: int = 2000,
    max_bytes: int = 51200,
    is_interrupted: Callable[[], bool] | None = None,
) -> dict:
    cwd = Path(workdir).resolve() if workdir else Path.cwd()
    if not cwd.exists():
        return {"output": "(no output)", "error": True, "exit_code": 1}

    shell = os.environ.get("SHELL", "/bin/sh")
    proc = None
    try:
        # start_new_session makes the shell the leader of its own process group
        # so a timeout can SIGKILL the whole group (background children too),
        # not just the shell PID.
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=shell,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=sys.platform != "win32",
        )
    except Exception as e:
        return {"output": f"failed to run command: {e}", "error": True, "exit_code": 127}

    deadline = time.monotonic() + max(float(timeout), 0.1)
    # Capture the shell's process-group id now, while the shell PID is still
    # alive. When the command finishes the shell dies, but stray background
    # children keep this pgid; a later getpgid(proc.pid) would fail and lose
    # the handle on them, so we kill by the captured id instead.
    pgid: int | None = None
    if sys.platform != "win32" and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    raw: bytes = b""
    capped = False
    end = "shell_exited"
    try:
        raw, capped, end = _stream_output(proc, deadline, max_bytes, is_interrupted)
    finally:
        # No matter which path we took — the command finished, failed, timed
        # out, or was interrupted — the tool call is over. Reap the shell's
        # whole process group and any daemonized children so a finished
        # command's background grandchildren can never outlive this call and
        # hold the pipe / cwd hostage. (This is the real fix for leaked
        # processes: before, the group was only killed on timeout.)
        _kill_proc_tree(proc, pgid)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # A child wedged in uninterruptible sleep can delay the reap; don't
            # block the caller forever on top of it.
            pass

    text = raw.decode("utf-8", errors="replace")
    if end in ("timeout", "interrupted"):
        result: dict = {
            "output": text,
            "exit_code": -1,
            "error": True,
            "metadata": {"exit_code": -1, "truncated": capped},
        }
        if end == "timeout":
            result["metadata"]["timeout"] = True
            result["metadata"]["timed_out"] = True
        else:
            result["metadata"]["stopped"] = True
            result["metadata"]["interrupted"] = True
            result["stopped"] = True
            result["interrupted"] = True
        return _apply_caps(result, max_lines, max_bytes)

    code = proc.returncode if proc.returncode is not None else 0
    result = {
        "output": text,
        "exit_code": code,
        "metadata": {"exit_code": code, "truncated": capped},
    }
    return _apply_caps(result, max_lines, max_bytes)


def _apply_caps(result: dict, max_lines: int, max_bytes: int) -> dict:
    output = result.get("output", "")
    truncated = bool(result.get("metadata", {}).get("truncated"))
    raw_len = len(output.encode("utf-8", errors="replace"))
    if raw_len > max_bytes:
        trimmed = output.encode("utf-8", errors="replace")[:max_bytes].decode(
            "utf-8", errors="ignore"
        )
        output = trimmed + f"\n... plus {raw_len - max_bytes} more bytes (truncated)"
        truncated = True
    lines = output.splitlines()
    if len(lines) > max_lines:
        output = "\n".join(lines[:max_lines])
        output += f"\n... plus {len(lines) - max_lines} more lines (truncated)"
        truncated = True
    result["output"] = output
    result["metadata"]["truncated"] = truncated
    return result


def tool(
    max_lines: int = 2000,
    max_bytes: int = 51200,
    default_timeout: int = 120,
    registry: Registry | None = None,
) -> Tool:
    description = """Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

Be aware: OS: {os}, Shell: {shell}

All commands run in the current working directory by default. Use the `workdir` parameter if you need to run a command in a different directory. AVOID using `cd <directory> && <command>` patterns - use `workdir` instead.

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for that instead."""

    import sys

    description = description.format(os=sys.platform, shell=os.environ.get("SHELL", "bash"))

    def run(input: dict) -> dict:
        command = input["command"]
        timeout_ms = int(input.get("timeout") or default_timeout * 1000)
        timeout = max(0.1, timeout_ms / 1000.0)
        workdir = input.get("workdir")
        # Read the engine's interrupt callback at call time (the registry hook
        # is installed by AgentLoop.__init__) so ESC/ctrl+C aborts a running
        # command instead of letting it run to its timeout. Each engine (main
        # + sub-agents) owns its own registry, so workers always see their own
        # turn's interrupt state.
        is_interrupted: Callable[[], bool] | None = None
        if registry is not None:
            checker = getattr(registry, "interrupt_check", None)
            if callable(checker):
                is_interrupted = checker
        return _bash(
            command,
            timeout=timeout,
            workdir=workdir,
            max_lines=max_lines,
            max_bytes=max_bytes,
            is_interrupted=is_interrupted,
        )

    return Tool(
        name="bash",
        description=description,
        parameters=schema_with(
            {
                "command": {
                    "type": "string",
                    "description": "The command to execute",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in milliseconds (default 120000)",
                    "optional": True,
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory to run the command in",
                    "optional": True,
                },
            },
            ["command"],
        ),
        run=run,
        permission="bash",
    )
