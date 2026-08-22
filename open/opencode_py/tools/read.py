"""read tool: read a file or directory with line numbers + offset/limit."""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

from .registry import Tool, schema_with

MAX_LINES = 2000
MAX_CHARS = 2000
MAX_OUTPUT = 50 * 1024  # 50 KB

# Files up to this size are read whole in one syscall and sliced in memory;
# anything larger streams the window so memory stays bounded for huge files.
# Kept small-ish: on a large file a top-of-file read stops after the window,
# which is cheaper than slurping the whole payload just to slice a few lines.
FAST_READ_BYTES = 256 * 1024

IMAGE_EXTENSIONS = {".png", ".jpeg", ".jpg", ".gif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def _is_binary_sample(sample: bytes) -> bool:
    if not sample:
        return False
    sample = sample[:1024]
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    nonprintable = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return nonprintable / len(sample) > 0.30


def _fuzzy_suggestion(path: Path) -> str | None:
    try:
        candidates = [p for p in path.parent.iterdir() if p.is_file()]
    except OSError:
        return None
    name = path.name
    matches = []
    for p in candidates:
        if p.stem == name or name in p.stem or p.stem in name:
            matches.append(p.name)
    return ", ".join(matches[:3]) or None


def _read_error(path: Path, error: BaseException) -> dict:
    suggestion = _fuzzy_suggestion(path)
    msg = f"Could not read file {path}: {error}"
    if suggestion:
        msg += f"\n\nDid you mean one of these?\n{suggestion}"
    return {"output": msg, "error": True}


def _read_window_lines(
    lines: list[str],
    offset: int,
    limit: int,
) -> tuple[list[str], int | None, bool, bool]:
    """Slice a line list into the requested window, capping the output size.

    Shared by the fast (whole-file) and streaming (large-file) read paths so
    both produce byte-identical results. Returns ``(numbered, total,
    reached_eof, truncated_out)`` where ``total`` is the last line index read
    (None if the window started past EOF) and ``numbered`` holds
    ``f"{lineno}: {content}"`` rows.
    """
    numbered: list[str] = []
    total: int | None = None
    truncated_out = False
    out_chars = 0
    start = max(0, offset - 1)
    limit = max(1, int(limit))
    end = start + limit
    last_existing = min(len(lines), end)
    if last_existing < start + 1:
        # window starts past EOF: mirror the streaming path, which reads every
        # line (total = line count) and reports end-of-file
        total = len(lines)
        return [], total, True, False
    total = last_existing
    for lineno in range(start + 1, last_existing + 1):
        content = lines[lineno - 1].rstrip("\n").rstrip("\r")
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + f"... (line truncated to {MAX_CHARS} chars)"
        numbered.append(f"{lineno}: {content}")
        capped_chars = len(content) + 16
        if out_chars + capped_chars > MAX_OUTPUT:
            truncated_out = True
            break
        out_chars += capped_chars
    # reached EOF when the window consumed the file's last line without the
    # output cap cutting it short
    reached_eof = total >= len(lines) and not truncated_out
    return numbered, total, reached_eof, truncated_out


def _read_fast(path: Path, offset: int, limit: int) -> dict:
    """Read a small file in one shot: fewer syscalls and no per-line IO loop.

    Used for files up to ~2 MB; the whole payload is decoded at once and the
    requested window is sliced from the line list. Universal-newline semantics
    (CRLF / lone-CR folding) are mirrored so line numbers match the streaming
    path exactly. Bigger files still use the streaming path so memory stays
    bounded.
    """
    try:
        with path.open("rb") as f:
            data = f.read()
    except OSError as e:
        return _read_error(path, e)
    if _is_binary_sample(data):
        return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
    # newline=None (universal) in the streaming path folds \r\n and lone \r to
    # \n before splitting into lines — mirror that here. splitlines(keepends)
    # yields exactly the same items the TextIOWrapper iterator would (each line
    # with its terminator, or a final unterminated line; nothing for an empty
    # file), so line counts match the streaming path for CRLF too.
    text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(keepends=True)
    numbered, total, reached_eof, truncated_out = _read_window_lines(lines, max(1, int(offset)), limit)

    body = "\n".join(numbered)
    if truncated_out:
        body = body[:MAX_OUTPUT]

    if reached_eof and total is not None:
        footer = f"(End of file - total {len(lines)} lines)"
    else:
        shown = max(1, int(offset)) + len(numbered) - 1
        footer = f"(Showing line {max(1, int(offset))}-{shown}. Use offset={shown + 1} to continue.)"
    if truncated_out:
        footer += " (Output capped at 50 KB.)"

    return {
        "output": f"<{path}>…</{path}>\n<type>file</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read_file(path: Path, offset: int = 1, limit: int = MAX_LINES) -> dict:
    # image / pdf -> base64 file attachment
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS or suffix in PDF_EXTENSIONS:
        try:
            data = path.read_bytes()
        except OSError as e:
            return _read_error(path, e)
        mime, _ = mimetypes.guess_type(str(path))
        b64 = base64.b64encode(data).decode()
        return {
            "output": f"<{path.name}><type>{mime}</type><content>data:{mime};base64,{b64}</content>",
            "metadata": {"loaded": [str(path)], "mime": mime},
        }

    # Whole-file fast path for ordinary-sized files; larger files stream so we
    # never hold a huge payload in memory just to read a window of it.
    try:
        if path.stat().st_size <= FAST_READ_BYTES:
            return _read_fast(path, offset, limit)
    except OSError as e:
        return _read_error(path, e)

    # binary detection + text window share ONE open: the old code opened the
    # file once for the binary sample and again for the text window, so a file
    # modified between the two opens could be torn (or the second open could
    # hit a different inode / fail on a deleted path). Seek back after the
    # sample and read the window from the same handle.
    start = max(0, offset - 1)
    limit = max(1, int(limit))
    end = start + limit

    # stream the window line-by-line, holding only the selected lines in memory
    numbered: list[str] = []
    total: int | None = None
    reached_eof = True
    out_chars = 0
    truncated_out = False
    try:
        with path.open("rb") as f:
            if _is_binary_sample(f.read(1024)):
                return {"output": f"File {path} is a binary file and cannot be read as text.", "error": True}
            f.seek(0)
            # newline=None so universal-newline splitting matches the old
            # text-mode open exactly (a CRLF / lone-CR file is not treated as
            # one giant line).
            stream = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline=None)
            try:
                for lineno, line in enumerate(stream, 1):
                    if lineno > end:
                        reached_eof = False
                        break
                    total = lineno
                    if lineno <= start:
                        continue
                    content = line.rstrip("\n").rstrip("\r")
                    if len(content) > MAX_CHARS:
                        content = content[:MAX_CHARS] + f"... (line truncated to {MAX_CHARS} chars)"
                    numbered.append(f"{lineno}: {content}")
                    capped_chars = (len(content) + 16)
                    if out_chars + capped_chars > MAX_OUTPUT:
                        truncated_out = True
                        reached_eof = False
                        break
                    out_chars += capped_chars
            finally:
                # keep the wrapper from closing/stealing the raw buffer when it
                # goes out of scope (the `with` on `f` manages fd lifetime).
                stream.detach()
    except OSError as e:
        return _read_error(path, e)

    body = "\n".join(numbered)
    if truncated_out:
        body = body[:MAX_OUTPUT]

    if reached_eof and total is not None:
        footer = f"(End of file - total {total} lines)"
    else:
        shown = start + len(numbered)
        footer = f"(Showing line {start + 1}-{shown}. Use offset={shown + 1} to continue.)"
    if truncated_out:
        footer += " (Output capped at 50 KB.)"

    return {
        "output": f"<{path}>…</{path}>\n<type>file</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read_directory(path: Path, offset: int = 1, limit: int = MAX_LINES) -> dict:
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return {"output": f"Could not read directory {path}: {e}", "error": True}
    total = len(entries)
    start = max(0, offset - 1)
    selected = entries[start : start + limit]
    lines = [(f"{p.name}/" if p.is_dir() else p.name) for p in selected]
    body = "\n".join(lines)
    if start == 0 and total <= limit:
        footer = f"(Showing {total} entries)"
    else:
        end = min(start + limit, total)
        footer = f"(Showing {start + 1}-{end} of {total} entries. Use offset={end + 1} to continue.)"
    return {
        "output": f"<{path}>…</{path}>\n<type>directory</type>\n<content>\n{body}\n</content>\n{footer}",
        "metadata": {"loaded": [str(path)]},
    }


def _read(filePath: str, offset: int = 1, limit: int = MAX_LINES) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        suggestion = _fuzzy_suggestion(path)
        msg = f"Path {path} does not exist."
        if suggestion:
            msg += f"\n\nDid you mean one of these?\n{suggestion}"
        return {"output": msg, "error": True}
    if path.is_dir():
        return _read_directory(path, offset=offset, limit=limit)
    return _read_file(path, offset=offset, limit=limit)


def tool() -> Tool:
    description = """Read a file or directory from the local filesystem. If the path does not exist, an error is returned.

Usage:
- The filePath parameter should be an absolute path.
- By default, this tool returns up to 2000 lines from the start of the file.
- The offset parameter is the line number to start from (1-indexed).
- To read later sections, call this tool again with a larger offset.
- Use the grep tool to find specific content in large files or files with long lines.
- If you are unsure of the correct file path, use the glob tool to look up filenames by glob pattern.
- Contents are returned with each line prefixed by its line number as `<line>: <content>`. For example, if a file has contents "foo\n", you will receive "1: foo\n". For directories, entries are returned one per line (without line numbers) with a trailing `/` for subdirectories.
- Any line longer than 2000 characters is truncated.
- Call this tool in parallel when you know there are multiple files you want to read.
- Avoid tiny repeated slices (30 line chunks). If you need more context, read a larger window.
- This tool can read image files and PDFs and return them as file attachments."""

    def run(input: dict) -> dict:
        return _read(
            input["filePath"],
            offset=int(input.get("offset") or 1),
            limit=int(input.get("limit") or MAX_LINES),
        )

    return Tool(
        name="read",
        description=description,
        parameters=schema_with(
            {
                "filePath": {"type": "string", "description": "The absolute path to the file or directory to read"},
                "offset": {"type": "integer", "description": "The line number to start from (1-indexed)", "optional": True},
                "limit": {"type": "integer", "description": "The number of lines to read", "optional": True},
            },
            ["filePath"],
        ),
        run=run,
        permission="read",
    )
