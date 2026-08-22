"""write tool: create/overwrite a file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .registry import Tool, schema_with


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` atomically: temp file in the same directory, then
    ``os.replace``. Opening the destination with ``"w"`` directly truncates it
    before the new content is fully written, so a mid-write failure (disk full,
    crash) would leave a torn/empty file behind. Same-directory temp keeps the
    rename on one filesystem (atomic on POSIX)."""
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        # newline="" preserves the caller's exact line endings instead of
        # letting universal-newline mode normalize them on write.
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write(filePath: str, content: str) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, content)
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}
    return {
        "output": "Wrote file successfully.",
        "metadata": {"filePath": str(path), "content": content},
    }


def tool() -> Tool:
    description = """Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.
- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."""

    def run(input: dict) -> dict:
        return _write(input["filePath"], input["content"])

    return Tool(
        name="write",
        description=description,
        parameters=schema_with(
            {
                "content": {"type": "string", "description": "The content to write to the file"},
                "filePath": {"type": "string", "description": "The absolute path to the file to write"},
            },
            ["content", "filePath"],
        ),
        run=run,
        permission="edit",
    )
