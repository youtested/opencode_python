"""grep tool: ripgrep subprocess if present, else pure-python re fallback."""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

from .read import _is_binary_sample
from .registry import Tool, schema_with

MAX_RESULTS = 100


def _expand_braces(pattern: str) -> list[str]:
    """Expand `{a,b,c}` in a fnmatch-style pattern into a list of patterns."""
    if "{" not in pattern:
        return [pattern]
    out: list[str] = []
    open_idx = pattern.find("{")
    close_idx = pattern.find("}", open_idx + 1)
    if close_idx == -1:
        return [pattern]
    for alt in pattern[open_idx + 1 : close_idx].split(","):
        expanded = pattern[:open_idx] + alt + pattern[close_idx + 1 :]
        out.append(expanded)
    return out


def _fnmatch_any(name: str, pattern: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in _expand_braces(pattern))


def _glob_matches(p: Path, base: Path, pattern: str) -> bool:
    """Match an include glob the way rg does.

    rg (globset) matches a glob CONTAINING a path separator against the path
    relative to the search root, and a separator-less glob against the file
    basename. The old code only tested basenames, so ``include="src/*.ts"``
    matched nothing. ``**`` is its own path segment in globset; fnmatch treats
    it as two stars (still permissive), so fall back to the separator rule.
    """
    for pat in _expand_braces(pattern):
        if "/" in pat:
            root = base if base.is_dir() else base.parent
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = None
            if rel is not None and fnmatch.fnmatch(str(rel), pat):
                return True
        elif fnmatch.fnmatch(p.name, pat):
            return True
    return False


def _parse_rg_line(line: str) -> tuple[str, int, str] | None:
    """Parse an `rg --no-heading --line-number` line: path:lineno:content.

    File names may themselves contain colons, so we scan left-to-right for the
    first `:<digits>:` marker (the line number) and treat everything before it
    as the path. rg always emits `path:line:content`, so the first ALL-DIGITS
    colon segment after the last path separator is the line number.
    """
    path = ""
    for i, char in enumerate(line):
        if char == ":":
            rest = line[i + 1 :]
            idx = rest.find(":")
            if idx <= 0:
                continue
            try:
                lineno = int(rest[:idx])
            except ValueError:
                continue
            return line[:i], lineno, rest[idx + 1 :]
    return None


def _grep_rg(pattern: str, base: Path, include: str | None = None) -> list[tuple[str, int, str]] | None:
    # "--" guards the pattern/glob: a search for "-foo" or "--bar" must not be
    # consumed as an rg flag (rg would otherwise read stdin / error out).
    #
    # rg's cwd-set invocation: ripgrep anchors slash-containing -g globs (and
    # its matched paths) to the *current working directory*, not the search
    # root. When the caller passes an absolute path outside the agent's cwd, a
    # glob like "src/*.ts" silently matched nothing. Running rg with cwd=base
    # makes the globs behave exactly like the pure-python fallback (anchored
    # relative to the search root), and results are normalized back to
    # absolute paths so both paths agree.
    cwd = base if base.is_dir() else base.parent
    search_arg = "." if base.is_dir() else base.name
    # --with-filename: rg omits the path when exactly one file is searched,
    # but _parse_rg_line (and the caller's grouping) require a path prefix.
    cmd = ["rg", "--no-heading", "--with-filename", "--line-number", "--color", "never"]
    if include:
        # rg understands brace globs natively, so pass the pattern as-is
        cmd += ["-g", include]
    cmd += ["--", pattern, search_arg]
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode not in (0, 1):
        return None
    base_resolved = base.resolve()
    results = []
    for line in proc.stdout.splitlines():
        parsed = _parse_rg_line(line)
        if parsed is None:
            continue
        filepath, lineno, text = parsed
        pp = Path(filepath)
        if not pp.is_absolute():
            pp = (cwd / pp).resolve()
        results.append((str(pp), lineno, text))
        if len(results) >= MAX_RESULTS:
            break
    return results


def _grep_py(pattern: str, base: Path, include: str | None = None) -> list[tuple[str, int, str]]:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return [("", 0, f"invalid regex: {e}")]
    results = []
    base_parts_len = len(base.resolve().parts)
    if base.is_dir():
        iterator = base.rglob("*")
    else:
        iterator = iter([base])

    def _is_hidden(p: Path) -> bool:
        return any(seg.startswith(".") for seg in p.parts[base_parts_len:])

    try:
        for p in iterator:
            if p.is_dir():
                continue
            if include and not _glob_matches(p, base, include):
                continue
            # skip hidden paths (mirrors rg's default) but not the search base itself
            if _is_hidden(p):
                continue
            # mirror rg: don't match inside binary files (errors=replace would
            # otherwise surface garbage "matches" of random high bytes)
            try:
                with p.open("rb") as bf:
                    head = bf.read(1024)
            except OSError:
                continue
            if _is_binary_sample(head):
                continue
            try:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if regex.search(line.rstrip("\n")):
                            results.append((str(p), i + 1, line.rstrip("\n")))
                            if len(results) >= MAX_RESULTS:
                                return results
            except OSError:
                continue
        return results
    except (OSError, RecursionError, ValueError):
        return results


def _grep(pattern: str, path: str | None = None, include: str | None = None) -> dict:
    base = Path(path).resolve() if path else Path.cwd()
    if not base.exists():
        return {"output": f"Path does not exist: {base}", "error": True}

    if shutil.which("rg"):
        results = _grep_rg(pattern, base, include)
    else:
        results = None
    if results is None:
        results = _grep_py(pattern, base, include)

    if results and results[0][0] == "" and results[0][1] == 0:
        return {"output": results[0][2], "error": True}

    truncated = len(results) >= MAX_RESULTS
    if not results:
        return {"output": "No files found"}

    # group per file
    grouped: dict[str, list[tuple[int, str]]] = {}
    for filepath, lineno, text in results:
        grouped.setdefault(filepath, []).append((lineno, text))

    total = len(results)
    header = f"Found {total} match" + ("es" if total != 1 else "")
    if truncated:
        header += " (more matches available)"
    lines = [header]
    for filepath, hits in grouped.items():
        lines.append(f"{filepath}:")
        for lineno, text in hits:
            lines.append(f"  Line {lineno}: {text}")
    if truncated:
        lines.append("(Results truncated. Consider using a more specific path or pattern.)")
    return {"output": "\n".join(lines), "metadata": {"matches": total, "truncated": truncated}}


def tool() -> Tool:
    description = """- Fast content search tool that works with any codebase size
- Searches file contents using regular expressions
- Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
- Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
- Returns file paths and line numbers with matching lines
- Use this tool when you need to find files containing specific patterns
- If you need to identify/count the number of matches within files, use the Bash tool with `rg` (ripgrep) directly. Do NOT use `grep`.
- When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the Task tool instead"""

    def run(input: dict) -> dict:
        return _grep(input["pattern"], input.get("path"), input.get("include"))

    return Tool(
        name="grep",
        description=description,
        parameters=schema_with(
            {
                "pattern": {"type": "string", "description": "The regular expression to search for"},
                "path": {"type": "string", "description": "The directory or file to search in", "optional": True},
                "include": {"type": "string", "description": "File pattern to filter (e.g. '*.js')", "optional": True},
            },
            ["pattern"],
        ),
        run=run,
        permission="grep",
    )
