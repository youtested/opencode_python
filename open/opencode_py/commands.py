"""Slash command registry + handlers.

Built-ins mirror opencode's TUI command set. Each handler receives a context
with the engine, session, config, auth, and a reply callback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config, save_config

# where to get a key per provider (shown by /models)
KEY_HINTS: dict[str, str] = {
    "groq": "https://console.groq.com/keys",
    "cerebras": "https://cloud.cerebras.ai/",
    "google": "https://aistudio.google.com/apikey",
    "openrouter": "https://openrouter.ai/keys",
    "nvidia": "https://build.nvidia.com/",
    "mistral": "https://console.mistral.ai/",
    "github": "https://github.com/settings/tokens",
    "sambanova": "https://cloud.sambanova.ai/",
    "togetherai": "https://api.together.ai/",
    "anthropic": "https://console.anthropic.com/",
    "openai": "https://platform.openai.com/api-keys",
    "ollama": "local",
}


@dataclass
class CommandContext:
    config: Config
    auth: Any
    session: Any = None
    engine: Any = None
    worktree: str = ""
    reply: Callable[[str], None] = field(default=lambda s: print(s))
    get_session: Callable[[], Any] | None = None
    set_agent: Callable[[str], None] | None = None
    set_model: Callable[[str], None] | None = None
    exit_app: Callable[[], None] | None = None
    resume: Callable[[str], None] | None = None
    connect: Callable[[str], None] | None = None
    registry: Any = None


@dataclass
class Command:
    name: str
    aliases: list[str]
    description: str
    handler: Callable[[CommandContext, str], None]
    hidden: bool = False
    preview: bool = True  # safe to run for the centered popup preview


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def list(self) -> list[Command]:
        seen: set[str] = set()
        out = []
        for c in self._commands.values():
            if c.name not in seen:
                seen.add(c.name)
                out.append(c)
        return out

    def names(self) -> list[str]:
        return [c.name for c in self.list()]


def _print_help(ctx: CommandContext, args: str) -> None:
    ctx.reply(
        "Commands:\n"
        + "\n".join(f"  /{c.name:<12} {c.description}" for c in ctx.registry.list() if not c.hidden)
    )


def _new(ctx: CommandContext, args: str) -> None:
    if ctx.get_session:
        s = ctx.get_session()
        if s:
            ctx.reply("Starting a new session (clear current view).")
    else:
        ctx.reply("New session.")


def _clear(ctx: CommandContext, args: str) -> None:
    if ctx.engine:
        ctx.engine.set_history([])
    ctx.reply("Conversation cleared.")


def _models(ctx: CommandContext, args: str) -> None:
    from .providers import FREE_DEFAULT_MODELS, FREE_PROVIDERS, fetch_zen_models

    models = fetch_zen_models()
    free = [m for m in models if m.get("free")]
    paid = [m for m in models if not m.get("free")]
    lines = ["Free models (OpenCode Zen) - no key needed:"]
    for m in free:
        ctx_size = f"{m['context']:,}" if m.get("context") else "?"
        lines.append(f"  opencode/{m['id']:<26} {m.get('name') or m['id']}  ctx={ctx_size}")
    lines.append("")
    lines.append("Paid models (need OPENCODE_API_KEY - https://opencode.ai/auth):")
    for m in paid[:20]:
        lines.append(f"  opencode/{m['id']:<26} {m.get('name') or m['id']}")
    if len(paid) > 20:
        lines.append(f"  ... and {len(paid) - 20} more")
    lines.append("")
    lines.append("Free-tier providers (bring your own key):")
    for pid in FREE_PROVIDERS:
        model = FREE_DEFAULT_MODELS.get(pid, "?")
        url = KEY_HINTS.get(pid, "")
        lines.append(f"  {pid:<12} {model:<26} {url}")
    lines.append("")
    lines.append("Local: ollama  (localhost:11434, no key)")
    ctx.reply("\n".join(lines))


def _connect(ctx: CommandContext, args: str) -> None:
    arg = args.strip()
    if ctx.connect:
        # TUI: hand off to the connect screen (optionally preselecting a provider)
        ctx.connect(arg.split()[0] if arg else "")
        return
    if not arg:
        ctx.reply(
            "Usage: /connect <provider>\n\nProviders: opencode (zen), groq, cerebras, "
            "google, openrouter, nvidia, mistral, github, sambanova, togetherai, anthropic, openai, ollama\n\n"
            "In the TUI, /connect opens the key-entry screen."
        )
        return
    provider = arg.split()[0]
    ctx.reply(f"Connecting {provider}... Use the TUI to paste your API key, or set the env var.")


def _permissions(ctx: CommandContext, args: str) -> None:
    perm = ctx.config.permission or {}
    ctx.reply(f"Current permission config:\n{json.dumps(perm, indent=2)}")


def _config(ctx: CommandContext, args: str) -> None:
    action = args.strip()
    if action == "print":
        ctx.reply(json.dumps(ctx.config.as_dict(), indent=2))
        return
    if action == "validate":
        problems = _validate_config(ctx.config)
        if problems:
            ctx.reply("Config issues:\n" + "\n".join(f"  - {p}" for p in problems))
        else:
            ctx.reply("Config is valid.")
        return
    ctx.reply("Usage: /config print|validate")


def _validate_config(cfg: Config) -> list[str]:
    """Best-effort config validation: known providers/themes, rotation shape, permission values."""
    problems: list[str] = []
    from .providers import FREE_PROVIDERS

    known = set(FREE_PROVIDERS) | {"opencode", "zen", "anthropic", "openai", "ollama"} | set(cfg.providers or {})
    if cfg.provider and cfg.provider not in known:
        problems.append(f"unknown provider '{cfg.provider}' (known: {', '.join(sorted(known))})")
    if not cfg.model:
        problems.append("no model configured (set 'model' or use /model)")
    for i, lane in enumerate(cfg.rotation or []):
        if not isinstance(lane, dict):
            problems.append(f"rotation[{i}] must be an object {{provider, model}}")
            continue
        if lane.get("provider") not in known:
            problems.append(f"rotation[{i}]: unknown provider '{lane.get('provider')}'")
        if not lane.get("model"):
            problems.append(f"rotation[{i}]: missing model")
    valid_actions = ("allow", "deny", "ask")
    for tool, action in (cfg.permission or {}).items():
        if action not in valid_actions:
            problems.append(f"permission['{tool}'] = '{action}' (want one of {', '.join(valid_actions)})")
    return problems


def _theme(ctx: CommandContext, args: str) -> None:
    theme = args.strip()
    if not theme:
        ctx.reply(f"Current theme: {ctx.config.theme}. Usage: /theme <name> (e.g. opencode, solarized)")
        return
    ctx.config.theme = theme
    save_config(ctx.config)
    ctx.reply(f"Theme set to {theme}.")


def _help(ctx: CommandContext, args: str) -> None:
    _print_help(ctx, args)


def _exit(ctx: CommandContext, args: str) -> None:
    if ctx.exit_app:
        ctx.exit_app()
    else:
        ctx.reply("Exiting.")


def _resume(ctx: CommandContext, args: str) -> None:
    session_id = args.strip()
    if not session_id:
        ctx.reply("Usage: /resume <session-id>  (see /sessions)")
        return
    if ctx.resume:
        ctx.resume(session_id)
    else:
        ctx.reply(f"Resuming session {session_id}.")


def _sessions(ctx: CommandContext, args: str) -> None:
    from .session import list_sessions

    sessions = list_sessions()
    if not sessions:
        ctx.reply("No saved sessions.")
        return
    lines = ["Sessions:"]
    for s in sessions[:20]:
        title = s.title or "(untitled)"
        lines.append(f"  {s.id[:12]}  {title}  ({s.model or '?'})  [{s.agent}]")
    ctx.reply("\n".join(lines))


def _export(ctx: CommandContext, args: str) -> None:
    """Write a session transcript (tool calls included) to a Markdown file."""
    from pathlib import Path

    from .session import load_session, session_to_markdown

    session_id = args.strip() or (ctx.engine.session_id if ctx.engine else "")
    if not session_id:
        ctx.reply("Usage: /export <session-id>  (omit to export the current session)")
        return
    sess = load_session(session_id)
    if sess is None:
        ctx.reply(f"Session {session_id} not found.")
        return
    path = Path.cwd() / f"opencode-session-{sess.id[:12]}.md"
    try:
        path.write_text(session_to_markdown(sess), encoding="utf-8")
    except OSError as e:
        ctx.reply(f"Export failed: {e}")
        return
    ctx.reply(f"Exported {len(sess.messages)} messages → {path.name}")


def _agent(ctx: CommandContext, args: str) -> None:
    name = args.strip()
    if not name:
        ctx.reply("Usage: /agent build|plan")
        return
    if ctx.set_agent:
        ctx.set_agent(name)
        ctx.reply(f"Switched to {name} agent.")
    else:
        ctx.reply(f"Agent: {name}")


def _model(ctx: CommandContext, args: str) -> None:
    model = args.strip()
    if not model:
        ctx.reply("Usage: /model <model-id>  (e.g. big-pickle, deepseek-v4-flash-free)")
        return
    if ctx.set_model:
        ctx.set_model(model)
        ctx.reply(f"Model set to opencode/{model}.")
    else:
        ctx.reply(f"Model: {model}")


def _undo(ctx: CommandContext, args: str) -> None:
    if not ctx.engine:
        ctx.reply("No active session to undo.")
        return
    ctx.reply(ctx.engine.undo_last())


def _compact(ctx: CommandContext, args: str) -> None:
    if ctx.engine:
        history = ctx.engine.get_history()
        if len(history) <= 2:
            ctx.reply("History too short to compact.")
            return
        # Upstream opencode's /compact runs the same summary-based compaction
        # as auto-compaction (anchored summary + recent tail), it does not just
        # drop the older turns. Delegate to the engine so the model summarizes
        # and the resulting ` Compaction ` panel shows in the session.
        try:
            summary = ctx.engine.force_compact()
        except Exception as e:
            ctx.reply(f"Compaction failed: {e}")
            return
        if summary:
            ctx.reply("History compacted.")
        else:
            ctx.reply("History compacted (kept last turn).")
    else:
        ctx.reply("No active session.")


def _exit_fn(ctx: CommandContext, args: str) -> None:
    _exit(ctx, args)


def build_registry() -> CommandRegistry:
    reg = CommandRegistry()
    for cmd in [
        Command("help", [], "Show help", _help),
        Command("new", ["clear"], "Start a new session", _new, preview=False),
        Command("models", [], "List models (free first)", _models),
        Command("connect", [], "Add a provider/API key", _connect, preview=False),
        Command("permissions", [], "Show permission config", _permissions),
        Command("config", [], "Print/validate config", _config),
        Command("theme", [], "Switch theme", _theme),
        Command("exit", ["quit", "q"], "Exit", _exit_fn, preview=False),
        Command("resume", ["continue", "sessions"], "Resume a session", _resume),
        Command("sessions", ["ls"], "List sessions", _sessions),
        Command("export", ["save"], "Export session transcript to Markdown", _export),
        Command("agent", [], "Switch agent (build|plan)", _agent),
        Command("model", [], "Switch model", _model),
        Command("undo", [], "Revert last tool action", _undo, preview=False),
        Command("compact", ["summarize"], "Compact conversation history", _compact, preview=False),
        Command("init", [], "Guided AGENTS.md setup", _init, preview=False),
        Command("review", [], "Review changes", _review),
    ]:
        reg.register(cmd)
    return reg


def _init(ctx: CommandContext, args: str) -> None:
    from pathlib import Path

    path = Path(ctx.worktree) / "AGENTS.md"
    if path.exists():
        ctx.reply("AGENTS.md already exists.")
        return
    path.write_text("# Project Instructions\n\nAdd guidance for the agent here.\n", encoding="utf-8")
    ctx.reply(f"Created {path}")


def _review(ctx: CommandContext, args: str) -> None:
    ctx.reply("Run `git diff` manually to review changes.")


def handle_command(registry: CommandRegistry, ctx: CommandContext, line: str) -> bool:
    """Handle a /command line. Returns True if it was a command."""
    if not line.startswith("/"):
        return False
    parts = line.split(maxsplit=1)
    name = parts[0][1:]
    args = parts[1] if len(parts) > 1 else ""
    cmd = registry.get(name)
    if cmd is None:
        ctx.reply(f"Unknown command: /{name}. Try /help")
        return True
    cmd.handler(ctx, args)
    return True


# attach registry to context convenience
def attach_registry(reg: CommandRegistry, ctx: CommandContext) -> CommandContext:
    ctx.registry = reg  # type: ignore[attr-defined]
    return ctx
