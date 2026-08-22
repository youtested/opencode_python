"""Settings screen: interactive, arrow-navigable list of every app/tool setting.

Reached via ctrl+p / ctrl+s as a centered popup. All of the config + tool output
settings are listed; use Up/Down to move the selection, Enter to edit a value:

  - boolean settings toggle on Enter
  - enum settings (theme, agent, permission mode) cycle on Left/Right/Enter
  - numeric / free-text settings open an inline input to type a value
  - model settings open the model picker

Esc closes the popup (or cancels an in-progress edit). The Model and Close
buttons remain for mouse/touch users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from ..config import save_config
from .model_picker import ModelPicker
from .theme import THEMES, get_theme


@dataclass
class Row:
    label: str
    get: Callable[[], str]
    kind: str = "info"  # info | bool | enum | int | string | model
    choices: list[str] | None = None
    apply: Callable[[str], None] | None = None
    propagate: bool = True  # a model pick only retargets the app engine if True


class SettingsScreen(ModalScreen[None]):
    """Centered popup listing every app/tool setting, keyboard navigable."""

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = kwargs.pop("cfg")
        self.engine = kwargs.pop("engine")
        self.auth = kwargs.pop("auth")
        self.session = kwargs.pop("session", None)
        self.on_model_change = kwargs.pop("on_model_change", None)
        super().__init__(**kwargs)
        self.rows: list[Row] = []
        self.index = 0
        self.editing = False
        self._edit_row: Row | None = None
        self._row_ys: list[int] = []
        self._row_heights: list[int] = []

    # -- row model ---------------------------------------------------------
    def _build_rows(self) -> list[Row]:
        cfg = self.cfg
        engine = self.engine
        theme_names = [k for k in THEMES]

        rows: list[Row] = []
        rows.append(Row("— General —", lambda: "", kind="info"))
        rows.append(Row("provider", lambda: cfg.provider))
        rows.append(Row(
            "model",
            lambda: f"opencode/{cfg.model}",
            kind="model",
            apply=lambda v: setattr(cfg, "model", v),
        ))
        rows.append(Row(
            "small model",
            lambda: f"opencode/{cfg.small_model}",
            kind="model",
            apply=lambda v: setattr(cfg, "small_model", v),
            propagate=False,
        ))
        rows.append(Row(
            "default agent",
            lambda: cfg.default_agent,
            kind="enum",
            choices=["build", "plan"],
            apply=lambda v: setattr(cfg, "default_agent", v),
        ))
        rows.append(Row("active agent", lambda: engine.agent))
        rows.append(Row(
            "model read timeout (s)",
            lambda: str(cfg.model_read_timeout),
            kind="int",
            apply=lambda v: setattr(cfg, "model_read_timeout", float(v)),
        ))
        rows.append(Row(
            "subagent depth",
            lambda: str(cfg.subagent_depth),
            kind="int",
            apply=lambda v: setattr(cfg, "subagent_depth", int(v)),
        ))

        rows.append(Row("— Appearance —", lambda: "", kind="info"))
        rows.append(Row(
            "theme",
            lambda: cfg.theme,
            kind="enum",
            choices=theme_names,
            apply=lambda v: setattr(cfg, "theme", v),
        ))

        rows.append(Row("— Tools & Output —", lambda: "", kind="info"))
        rows.append(Row(
            "bash timeout (s)",
            lambda: str(cfg.bash_default_timeout),
            kind="int",
            apply=lambda v: setattr(cfg, "bash_default_timeout", int(v)),
        ))
        rows.append(Row(
            "tool output max lines",
            lambda: str(cfg.tool_output_max_lines),
            kind="int",
            apply=lambda v: setattr(cfg, "tool_output_max_lines", int(v)),
        ))
        rows.append(Row(
            "tool output max bytes",
            lambda: str(cfg.tool_output_max_bytes),
            kind="int",
            apply=lambda v: setattr(cfg, "tool_output_max_bytes", int(v)),
        ))
        rows.append(Row(
            "context budget (tokens)",
            lambda: str(cfg.context_budget),
            kind="int",
            apply=lambda v: setattr(cfg, "context_budget", int(v)),
        ))

        rows.append(Row("— Diff —", lambda: "", kind="info"))
        rows.append(Row(
            "diff style",
            lambda: cfg.diff_style,
            kind="enum",
            choices=["split", "stacked"],
            apply=lambda v: setattr(cfg, "diff_style", v),
        ))
        rows.append(Row(
            "diff wrap",
            lambda: cfg.diff_wrap_mode,
            kind="enum",
            choices=["word", "none"],
            apply=lambda v: setattr(cfg, "diff_wrap_mode", v),
        ))
        rows.append(Row(
            "diff backgrounds",
            lambda: "off" if cfg.suppress_backgrounds else "on",
            kind="enum",
            choices=["on", "off"],
            apply=lambda v: setattr(cfg, "suppress_backgrounds", v == "off"),
        ))

        rows.append(Row("— Conversation —", lambda: "", kind="info"))
        rows.append(Row(
            "auto compact",
            lambda: "yes" if cfg.compaction_enabled else "no",
            kind="bool",
            apply=lambda v: setattr(cfg, "compaction_enabled", v == "yes"),
        ))
        rows.append(Row(
            "compact tail turns",
            lambda: str(cfg.compaction_tail_turns),
            kind="int",
            apply=lambda v: setattr(cfg, "compaction_tail_turns", int(v)),
        ))

        rows.append(Row("— Permissions —", lambda: "", kind="info"))
        rows.append(Row(
            "permission mode",
            lambda: engine.permission.mode,
            kind="enum",
            choices=["auto", "ask", "deny"],
            apply=lambda v: setattr(engine.permission, "mode", v),
        ))
        for tool, action in (cfg.permission or {}).items():
            rows.append(Row(f"permission:{tool}", lambda a=action: str(a)))

        rows.append(Row("— Rotation —", lambda: "", kind="info"))
        for lane in (cfg.rotation or []):
            rows.append(Row(
                "lane",
                lambda l=lane: f"{l.get('provider', '?')}/{l.get('model', '?')}",
            ))

        rows.append(Row("— Plugins & MCP —", lambda: "", kind="info"))
        for p in (cfg.raw.get("plugins") or []):
            rows.append(Row("plugin", lambda p=p: str(p)))
        for name in (cfg.raw.get("mcpServers") or {}):
            rows.append(Row("mcp", lambda name=name: name))
        rows.append(Row(
            "auth keys",
            lambda: ", ".join(sorted(self.auth.list())) if self.auth and self.auth.list() else "(none)",
        ))
        if self.session:
            rows.append(Row(
                "session",
                lambda: f"{self.session.id[:12]} [{self.session.agent}] {self.session.model or '?'}",
            ))
        return rows

    @property
    def _row(self) -> Row:
        return self.rows[self.index]

    # -- UI ----------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Vertical(classes="cmd-popup settings"):
            yield Static("  Settings  ", classes="cmd-popup-title")
            with VerticalScroll(id="settings-scroll", classes="settings-scroll"):
                self.body = Static("", id="settings-body", classes="settings-body")
                yield self.body
            yield Input(
                id="settings-edit-input",
                classes="settings-edit",
                placeholder="enter a value",
            )
            with Horizontal(classes="cmd-popup-actions"):
                yield Button("Model", id="settings-model", variant="default")
                yield Button("Save", id="settings-save", variant="primary")
                yield Button("Close", id="settings-close", variant="default")
        yield Static(
            "↑/↓ select   Enter edit   ←/→ cycle   Esc close",
            id="settings-hint",
            classes="settings-hint",
        )

    def _scroll_container(self) -> Any:
        return self.query_one("#settings-scroll")

    def on_mount(self) -> None:
        self.rows = self._build_rows()
        self.index = next((i for i, r in enumerate(self.rows) if r.kind != "info"), 0)
        self._set_edit_input_visible(False)
        self._toggle_row_visibility()
        # defer render until layout settles to avoid first-frame sizing issues
        self.set_timer(0.05, self._render_settings)
        self._scroll_container().can_focus = True
        self._scroll_container().focus()

    def _keep_selection_visible(self) -> None:
        """Scroll the settings body so the selected row stays on screen."""
        if not self.is_attached:
            return
        if self.editing or not self._row_ys:
            return
        width = max(30, self.body.size.width)
        top = self._row_ys[max(0, self.index)]
        row_h = self._row_heights[self.index] if self.index < len(self._row_heights) else 1
        box = self._scroll_container()
        box_h = max(1, box.size.height)
        bottom = top + row_h
        scrollbar = box.scroll_offset.y
        if top < scrollbar:
            box.scroll_to(y=top, animate=False)
        elif bottom > scrollbar + box_h:
            box.scroll_to(y=max(0, bottom - box_h), animate=False)

    def _set_edit_input_visible(self, visible: bool) -> None:
        self.query_one("#settings-edit-input", Input).display = visible
        if not visible:
            self.query_one("#settings-edit-input", Input).value = ""

    def _toggle_row_visibility(self) -> None:
        self.query_one("#settings-body", Static).display = not self.editing

    # -- rendering ---------------------------------------------------------
    def _render_settings(self) -> None:
        if not self.is_attached:
            return
        theme = get_theme(self.cfg.theme or "opencode")
        accent = theme.c("accent")
        muted = theme.c("text_muted")
        highlight = theme.c("primary")
        lines: list[str] = []
        plain_lines: list[str] = []
        for i, row in enumerate(self.rows):
            selected = i == self.index and not self.editing
            marker = "▶" if selected else " "
            value = row.get()
            if row.kind == "info":
                if row.label.startswith("—"):
                    rendered = f"[bold {accent}]{value or row.label}[/]"
                    plain = value or row.label
                else:
                    rendered = f"[{muted}]  {row.label:<22}[/] {value}"
                    plain = f"  {row.label}  {value}"
                lines.append(rendered)
                plain_lines.append(plain)
                continue
            if selected:
                lines.append(f"[{highlight}]{marker} {row.label:<24}[/] [bold]{value}[/]")
                plain = f"{marker} {row.label} {value}"
            else:
                lines.append(f"[{muted}]{marker}[/] {row.label:<24} {value}")
                plain = f"{marker} {row.label} {value}"
            plain_lines.append(plain)

        self.query_one("#settings-body", Static).update("\n".join(lines))

        # record each row's top-y (in display lines) so we can keep it visible
        width = max(20, self.body.size.width)
        ys: list[int] = []
        heights: list[int] = []
        acc = 0
        for p in plain_lines:
            ys.append(acc)
            h = max(1, (len(p) + width - 1) // width)
            heights.append(h)
            acc += h
        self._row_ys = ys
        self._row_heights = heights
        self.call_after_refresh(self._keep_selection_visible)

    def _after_change(self) -> None:
        # persist the merged config so changes survive a restart
        try:
            save_config(self.cfg)
        except Exception:
            pass
        self._scroll_container().focus()

    # -- navigation / editing ---------------------------------------------
    def _move(self, delta: int) -> None:
        if self.editing:
            return
        n = len(self.rows)
        if n == 0:
            return
        new = self.index
        steps = 0
        while steps < n:
            new = (new + delta) % n
            if self.rows[new].kind != "info":
                self.index = new
                break
            steps += 1
        self._render_settings()

    def _activate(self) -> None:
        if self.editing:
            return self._commit_edit()
        row = self._row
        if self.editing:
            return
        if row.kind in ("model",):
            self._open_model_picker(row.apply, propagate=row.propagate)
        elif row.kind == "bool":
            cur = row.get() == "yes"
            if row.apply:
                row.apply("no" if cur else "yes")
            self._after_change()
            self._render_settings()
        elif row.kind == "enum":
            self._cycle(1)
        elif row.kind in ("int", "string"):
            self._begin_edit()
        # info rows do nothing

    def _cycle(self, delta: int) -> None:
        row = self._row
        if row.kind != "enum" or not row.choices:
            return
        choices = row.choices
        try:
            cur = choices.index(row.get())
        except ValueError:
            cur = -1
        nxt = (cur + delta) % len(choices)
        if row.apply:
            row.apply(choices[nxt])
        self._after_change()
        self._render_settings()
        self.app.notify(f"{row.label}: {row.get()}")

    def _begin_edit(self) -> None:
        row = self._row
        if row.kind not in ("int", "string"):
            return
        self.editing = True
        self._edit_row = row
        inp = self.query_one("#settings-edit-input", Input)
        inp.value = row.get()
        inp.display = True
        self._toggle_row_visibility()
        inp.focus()

    def _commit_edit(self) -> None:
        row = self._edit_row
        if row and row.apply is not None:
            value = self.query_one("#settings-edit-input", Input).value.strip()
            try:
                row.apply(value)
            except (ValueError, TypeError):
                self.app.notify(f"Invalid value: {value!r}", severity="warning")
        self._cancel_edit()
        self._after_change()
        self._render_settings()

    def _cancel_edit(self) -> None:
        self.editing = False
        self._edit_row = None
        self._set_edit_input_visible(False)
        self._toggle_row_visibility()
        self._scroll_container().focus()

    def _open_model_picker(self, apply: Callable[[str], None] | None, propagate: bool = True) -> None:
        def on_picked(choice: str | None) -> None:
            if not choice or not apply:
                self._scroll_container().focus()
                return
            provider, _, model = choice.partition("/")
            if provider and model:
                self.cfg.provider = provider
            apply(model)
            # only a real model pick on a row that owns the app model should
            # propagate to the app engine (the "small model" row must not)
            if propagate and self.on_model_change:
                self.on_model_change(model)
            self._after_change()
            self._render_settings()
            self._scroll_container().focus()

        self.app.push_screen(
            ModelPicker(current=self.cfg.model, cfg=self.cfg, auth=self.auth),
            on_picked,
        )

    # -- events ------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "settings-model":
            self._open_model_picker(lambda v: setattr(self.cfg, "model", v))
        elif bid == "settings-save":
            self._after_change()
            self.app.notify("Settings saved.")
        elif bid == "settings-close":
            self.dismiss(None)

    def on_key(self, event: Key) -> None:
        key = event.key
        if key == "escape":
            if self.editing:
                self._cancel_edit()
            else:
                self.dismiss(None)
            event.stop()
            return
        if self.editing:
            # only Enter commits / Esc cancels; everything else goes to the input
            if key == "enter":
                self._commit_edit()
                event.stop()
            return
        if key in ("down", "j"):
            self._move(1)
            event.stop()
        elif key in ("up", "k"):
            self._move(-1)
            event.stop()
        elif key == "enter" or key == "space":
            self._activate()
            event.stop()
        elif key in ("right", "left"):
            self._cycle(1 if key == "right" else -1)
            event.stop()