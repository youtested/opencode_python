"""Model picker screen (/models + Settings): live, grouped, auto-refreshing.

Mirrors opencode's model picker: providers are group headers with their models
listed underneath, the whole thing sorted with the free providers / free models
first. A search box filters the list as you type. Selecting a model dismisses
with a "provider/model" string so settings (and /models) can switch both at
once.

Model lists are fetched live from each provider's `/models` endpoint (only when
an API key is present), fall back to a bundled default when unavailable, and
auto-refresh every REFRESH_SECONDS.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListView, ListItem, Static

from ..providers import (
    FREE_PROVIDERS,
    FREE_DEFAULT_MODELS,
    PAID_PROVIDERS,
    fetch_zen_models,
    fetch_openrouter_models,
    fetch_live_models,
)
from ..question import QuestionInfo
from .question_dialog import QuestionDialog

REFRESH_SECONDS = 60

# (provider id, display name) — free providers first, paid after.
FREE_SECTION: list[tuple[str, str]] = [
    ("opencode", "OpenCode Zen"),
    ("openrouter", "OpenRouter"),
    ("groq", "Groq"),
    ("cerebras", "Cerebras"),
    ("google", "Google AI Studio"),
    ("nvidia", "NVIDIA NIM"),
    ("mistral", "Mistral"),
    ("github", "GitHub Models"),
    ("sambanova", "SambaNova"),
    ("togetherai", "Together"),
    ("ollama", "Ollama (local)"),
]

PAID_SECTION: list[tuple[str, str]] = [
    ("anthropic", "Anthropic Claude"),
    ("openai", "OpenAI"),
    ("deepseek", "DeepSeek"),
    ("xai", "xAI"),
    ("deepinfra", "DeepInfra"),
]

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("free", FREE_SECTION),
    ("paid", PAID_SECTION),
]

# curated fallback when a paid provider has no key / the live list is down.
DEFAULT_PAID_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
    "openai": ["gpt-4o", "gpt-4o-mini", "o3-mini"],
    "deepseek": ["deepseek-chat"],
    "xai": ["grok-2-latest"],
    "deepinfra": ["meta-llama/Meta-Llama-3.3-70B-Instruct"],
}

_MODEL_PICKER_CSS = """
ModelPicker {
    background: #0a0a0a;
}
#model-picker {
    width: 100%;
    height: 100%;
    layout: vertical;
    padding: 1 2;
}
#models-header {
    height: auto;
    align-horizontal: right;
    margin-bottom: 1;
}
.screen-title {
    height: auto;
    width: 1fr;
    color: #eeeeee;
    text-style: bold;
}
.esc-hint {
    height: auto;
    color: #808080;
}
#models-search {
    height: 1;
    border: none;
    padding: 0 1;
    background: transparent;
    color: #808080;
    margin-bottom: 1;
}
#models-search:focus {
    border: none;
    background: #141414;
    background-tint: transparent;
}
#models-search > .input--cursor {
    background: #fab283;
    color: #0a0a0a;
    text-style: bold;
}
#models-search > .input--placeholder {
    color: #808080;
}
#models-status {
    height: auto;
    margin-bottom: 1;
    color: #808080;
}
#models-list {
    height: 1fr;
    border: none;
    background: #0a0a0a;
}
.group-header {
    height: auto;
    padding: 1 0 0 1;
    color: #9d7cd8;
    text-style: bold;
}
.zen-sub-group {
    height: auto;
    padding: 0 0 0 2;
    color: #6f78d0;
    text-style: bold;
}
.model-item {
    height: auto;
    padding: 0 0 0 3;
    color: #eeeeee;
}
.model-item .free-tag {
    color: #7fd88f;
}
.model-item .current-mark {
    color: #5c9cf5;
}
#models-actions {
    height: auto;
    padding-top: 1;
    align-horizontal: right;
}
#models-actions Button {
    margin-left: 1;
}
"""


class ModelsNav(Message):
    """The search input wants the list to move/select (mirrors opentui, where
    the filter input drives the selection while it keeps focus)."""

    def __init__(self, action: str) -> None:
        super().__init__()
        self.action = action


class _ModelsInput(Input):
    """Search box whose Up/Down/Enter/Escape drive the list instead of being
    consumed by the Input itself (Enter would otherwise just "submit")."""

    BINDINGS = [
        Binding("up", "nav_up", show=False),
        Binding("down", "nav_down", show=False),
        Binding("enter", "nav_select", "Select", show=False),
        Binding("escape", "nav_close", "Close", show=False),
    ]

    def action_nav_up(self) -> None:
        self.post_message(ModelsNav("up"))

    def action_nav_down(self) -> None:
        self.post_message(ModelsNav("down"))

    def action_nav_select(self) -> None:
        self.post_message(ModelsNav("select"))

    def action_nav_close(self) -> None:
        self.post_message(ModelsNav("close"))


class ModelPicker(ModalScreen[str | None]):
    """Full-screen model list; Enter selects, Esc dismisses, R refreshes."""

    CSS = _MODEL_PICKER_CSS

    BINDINGS = [
        Binding("r", "refresh_models", "Refresh"),
        Binding("escape", "dismiss_pop", "Close"),
    ]

    def __init__(
        self,
        current: str = "",
        on_select: Callable[[str], None] | None = None,
        cfg: Any = None,
        auth: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.current = current
        self.on_select = on_select
        self.cfg = cfg
        self.auth = auth
        self.models: dict[str, list[dict]] = {}
        self._item_lookup: list[dict] = []
        self._fetching = False
        self._timer: Any = None
        # Custom providers registered under config `providers.<id>`: they get
        # their own section at the end of the list, fetched live from each
        # provider's /models endpoint when an API key is present.
        self.custom_providers: dict[str, dict] = dict((cfg.providers or {}).items()) if cfg else {}
        # Guard against re-entrant selection: programmatic `lv.index = …` during
        # a refresh/search fires a *spurious* ListView.Selected that must not be
        # treated as the user picking a model, and once dismissed, queued events
        # must not call dismiss() again (pop_screen would raise on an empty
        # stack).
        self._rebuilding = False
        self._dismissed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker"):
            with Horizontal(id="models-header"):
                yield Label("Models", classes="screen-title")
                yield Label("esc", classes="esc-hint")
            yield _ModelsInput(placeholder="Search models…", id="models-search")
            yield Static("Loading models...", id="models-status")
            yield ListView(id="models-list")
            with Horizontal(id="models-actions"):
                yield Button("Refresh", id="models-refresh", variant="default")
                yield Button("Close", id="models-close", variant="primary")

    def on_mount(self) -> None:
        self.set_loading()
        self._start_worker()
        self._timer = self.set_interval(REFRESH_SECONDS, self._periodic_refresh)

    def on_unmount(self) -> None:
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None

    # -- fetching ----------------------------------------------------------
    def set_loading(self) -> None:
        if not self.is_attached:
            return
        try:
            self.query_one("#models-status", Static).update(
                f"Fetching model lists from providers... (auto-refresh every {REFRESH_SECONDS}s)"
            )
        except Exception:
            pass

    def _start_worker(self) -> None:
        if self._fetching:
            return
        self._fetching = True
        self.set_loading()
        self.run_worker(self._fetch_models, thread=True)

    def _periodic_refresh(self) -> None:
        self._start_worker()

    def _fetch_models(self) -> None:
        pids = [pid for _, providers in SECTIONS for pid, _ in providers]
        pids += list(self.custom_providers)
        per_provider: dict[str, list[dict]] = {}
        try:
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {ex.submit(self._fetch_provider_models, pid): pid for pid in pids}
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        per_provider[pid] = future.result() or []
                    except Exception:
                        per_provider[pid] = []
        finally:
            self._fetching = False
        self.app.call_from_thread(self.populate, per_provider)

    def _fetch_provider_models(self, pid: str) -> list[dict]:
        if pid == "opencode":
            return fetch_zen_models()
        if pid == "openrouter":
            return fetch_openrouter_models()
        if pid == "ollama":
            return [
                {"id": "llama3.2", "name": "Llama 3.2", "context": 128000, "free": True},
                {"id": "llama3.1", "name": "Llama 3.1", "context": 128000, "free": True},
            ]
        custom = self.custom_providers.get(pid)
        if custom:
            key = self.auth.get(pid) if self.auth else None
            models = fetch_live_models(pid, key, custom.get("base_url"))
            if models:
                return models
            # live fetch failed (no key / endpoint down): show the configured
            # model so the provider's section is never empty
            cur = (self.cfg.model or "").split("/")[-1] if self.cfg else ""
            return [{"id": cur, "name": cur, "context": 0, "free": False}] if cur else []
        meta = FREE_PROVIDERS.get(pid) or PAID_PROVIDERS.get(pid) or {}
        key = self.auth.get(pid) if self.auth else None
        models = (
            fetch_live_models(pid, key, meta.get("base_url"), meta.get("api_kind", "openai"))
            if meta
            else []
        )
        if models:
            # the whole provider is in the free section, so badge its models FREE
            is_free_section = any(p == pid for p, _ in FREE_SECTION)
            for m in models:
                m["free"] = is_free_section
            return models
        return _fallback_models(pid, has_key=bool(key))

    # -- display -----------------------------------------------------------
    def _query(self) -> str:
        if not self.is_attached:
            return ""
        try:
            return (self.query_one("#models-search", Input).value or "").strip().lower()
        except Exception:
            return ""

    def _set_status(self, text: str) -> None:
        if not self.is_attached:
            return
        try:
            self.query_one("#models-status", Static).update(text)
        except Exception:
            pass

    def populate(self, per_provider: dict[str, list[dict]]) -> None:
        # The fetch worker may complete after the screen was dismissed (Esc /
        # Close / model picked). Guard the widget lookups so a pruned screen
        # doesn't raise NoMatches and crash the whole app.
        if not self.is_attached:
            return
        self.models = per_provider
        self._populate_list()

    def _populate_list(self) -> None:
        if not self.is_attached or self._dismissed:
            return
        lv = self.query_one("#models-list", ListView)
        # Rebuilding fires spurious ListView.Selected events when lv.index is
        # set below; ignore them so refresh/search don't auto-dismiss.
        self._rebuilding = True
        try:
            lv.clear()
            self._item_lookup = []
            q = self._query()
            total_free, total_paid, shown, add_row = self._populate_rows(lv, q)
            if shown == 0 and not add_row:
                self._set_status("No models match your search.")
                return
        finally:
            self._rebuilding = False

        # highlight the current model when present, else the first real row.
        # The `__custom__` sentinel row carries model == "" so it must be
        # excluded (an empty `current` must match nothing, not the "add custom
        # provider" entry) or the picker highlights "Add custom" on open.
        current_hit = [
            e["row"]
            for e in self._item_lookup
            if e["provider"] != "__custom__"
            and self.current
            and (
                f"{e['provider']}/{e['model']}" == self.current
                or e["model"] == self.current.split("/")[-1]
            )
        ]
        self._rebuilding = True
        try:
            lv.index = current_hit[0] if current_hit else self._item_lookup[0]["row"]
        finally:
            self._rebuilding = False

        def _fmt_count():
            if q:
                return f"{shown} model{'s' if shown != 1 else ''} — filtered by '{q}'"
            return f"{total_free} free, {total_paid} paid"

        self._set_status(
            f"{_fmt_count()} — updated {time.strftime('%H:%M:%S')} — Enter select · R refresh"
        )

    def _populate_rows(self, lv: ListView, q: str) -> tuple[int, int, int, bool]:
        """Append provider rows/models; returns (total_free, total_paid, shown, add_row)."""
        total_free = 0
        total_paid = 0
        shown = 0
        row = 0  # sequential index of the row being appended
        for _, providers in SECTIONS:
            for pid, display in providers:
                items = self.models.get(pid) or []
                if not items:
                    continue
                # free models first within the provider (official sorts free
                # before paid), then by id
                ordered = sorted(items, key=lambda m: (not bool(m.get("free")), m["id"]))
                if q:
                    ordered = [m for m in ordered if q in m["id"].lower() or q in (m.get("name") or "").lower()]
                if not ordered:
                    continue
                lv.append(ListItem(Label(f"  {display}", classes="group-header")))
                row += 1
                # OpenCode Zen mixes many upstream vendors under one provider,
                # so cluster it: free models first, then non-free by family.
                if pid == "opencode":
                    free_items = [m for m in ordered if m.get("free")]
                    paid_items = [m for m in ordered if not m.get("free")]
                    groups: list[tuple[str | None, list[dict]]] = []
                    if free_items:
                        groups.append(("Free", free_items))
                    by_family: dict[str, list[dict]] = {}
                    for m in paid_items:
                        by_family.setdefault(_zen_family(m["id"]), []).append(m)
                    for family in sorted(by_family):
                        groups.append((family, by_family[family]))
                else:
                    groups = [(None, ordered)]
                for label, group in groups:
                    if label is not None:
                        lv.append(ListItem(Label(f"   {label}", classes="zen-sub-group")))
                        row += 1
                    for m in group:
                        idx = f"{pid}/{m['id']}"
                        # row must be a plain running count: len(lv.children) is
                        # stale while appends await the DOM refresh (previous rows
                        # are still registered during the rebuild)
                        self._item_lookup.append({"row": row, "provider": pid, "model": m["id"]})
                        if m.get("free"):
                            total_free += 1
                        else:
                            total_paid += 1
                        shown += 1
                        lv.append(ListItem(_model_row_label(idx, m, self.current)))
                        row += 1
        # custom providers from config `providers.<id>` — their own section at
        # the end, so a provider added from the "add custom" row shows up here.
        for pid in sorted(self.custom_providers):
            meta = self.custom_providers[pid]
            items = self.models.get(pid) or []
            if not items:
                continue
            display = meta.get("name") or pid
            ordered = sorted(items, key=lambda m: (not bool(m.get("free")), m["id"]))
            if q:
                ordered = [
                    m
                    for m in ordered
                    if q in m["id"].lower() or q in (m.get("name") or "").lower() or q in pid
                ]
            if not ordered:
                continue
            lv.append(ListItem(Label(f"  {display}  (custom)", classes="group-header")))
            row += 1
            for m in ordered:
                idx = f"{pid}/{m['id']}"
                self._item_lookup.append({"row": row, "provider": pid, "model": m["id"]})
                total_paid += 1
                shown += 1
                lv.append(ListItem(_model_row_label(idx, m, self.current)))
                row += 1
        # the "add custom provider" entry lives at the very end of the list.
        add_row = False
        if not q or "custom" in q:
            add_row = True
            lv.append(ListItem(Label("  Custom", classes="group-header")))
            row += 1
            self._item_lookup.append({"row": row, "provider": "__custom__", "model": ""})
            lv.append(
                ListItem(
                    Label("   [bold #fab283]＋[/] Add custom provider (URL + API key + model)", classes="model-item")
                )
            )
            row += 1
        return total_free, total_paid, shown, add_row

    def _close(self, result: str | None = None) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        self.dismiss(result)

    # -- events ------------------------------------------------------------
    def on_models_nav(self, event: Any) -> None:
        action = getattr(event, "action", "")
        if action == "up":
            self._move_selection(-1)
        elif action == "down":
            self._move_selection(1)
        elif action == "select":
            self._choose_current()
        elif action == "close":
            self._close(None)

    def _move_selection(self, direction: int) -> None:
        if not self.is_attached or self._dismissed:
            return
        # Only real model rows are navigable: headers and the "add custom
        # provider" sentinel (`__custom__`) are skipped, so Up/Down clamp at
        # the first/last model.
        rows = sorted({e["row"] for e in self._item_lookup if e["provider"] != "__custom__"})
        if not rows:
            return
        lv = self.query_one("#models-list", ListView)
        current = lv.index
        if current is None or current not in rows:
            # no selection yet: land on the first/last real model row
            target = rows[0] if direction > 0 else rows[-1]
        else:
            target = current
            while True:
                target += direction
                if target in rows:
                    break
                if (direction > 0 and target > rows[-1]) or (direction < 0 and target < rows[0]):
                    return
        lv.index = target

    def _choose_current(self) -> None:
        if not self.is_attached or self._dismissed:
            return
        lv = self.query_one("#models-list", ListView)
        self._choose_row(lv.index)

    def _choose_row(self, index: int | None) -> None:
        # The refresh/search rebuild sets `lv.index` programmatically, which
        # fires a ListView.Selected that is NOT a user pick — ignore it. Also
        # once the picker is dismissed, later queued events must be no-ops.
        if self._rebuilding or self._dismissed or not self.is_attached:
            return
        for entry in self._item_lookup:
            if entry["row"] == index:
                if entry["provider"] == "__custom__":
                    self._add_custom_provider()
                    return
                choice = f"{entry['provider']}/{entry['model']}"
                if self.on_select:
                    self.on_select(choice)
                self._close(choice)
                return

    # -- add custom provider ---------------------------------------------
    def _add_custom_provider(self) -> None:
        """Ask for provider id / base URL / API key / model, save them, and
        switch to the new provider. Prompted via the existing QuestionDialog so
        the flow stays consistent with the rest of the TUI."""
        if self._dismissed or not self.is_attached:
            return
        questions = [
            QuestionInfo(
                question="Give this provider a short id (letters, numbers, _ and -), e.g. teamo.",
                header="Provider id",
            ),
            QuestionInfo(
                question="OpenAI-compatible base URL, e.g. https://api.teamorouter.com/v1",
                header="Base URL",
            ),
            QuestionInfo(
                question="API key (sk-...). Saved to auth.json (0600), never to the config file.",
                header="API key",
            ),
            QuestionInfo(
                question="Model id to use by default, e.g. deepseek-v4-flash-free",
                header="Model",
            ),
        ]
        self.app.push_screen(QuestionDialog(questions), self._on_custom_done)

    def _on_custom_done(self, result: Any) -> None:
        """Apply a completed add-custom-provider flow (or cancel)."""
        if not result or self._dismissed or not self.is_attached:
            return

        def _answer(i: int) -> str:
            return (result[i][0] if i < len(result) and result[i] else "").strip()

        import re

        pid = re.sub(r"[^a-z0-9_-]+", "", _answer(0).lower())
        base = _answer(1).rstrip("/")
        key = _answer(2)
        model = _answer(3)

        if not pid or not base or not model:
            self.app.notify("Custom provider needs an id, base URL and model.", severity="warning")
            return
        if not key:
            self.app.notify("Custom provider needs an API key.", severity="warning")
            return
        if not base.startswith("http://") and not base.startswith("https://"):
            base = "https://" + base
        from ..providers import FREE_PROVIDERS, PAID_PROVIDERS

        if pid in FREE_PROVIDERS or pid in PAID_PROVIDERS or pid in ("opencode", "ollama"):
            self.app.notify(f"'{pid}' is a built-in provider — pick a different id.", severity="warning")
            return

        if self.auth is not None:
            try:
                self.auth.set(pid, key)
            except Exception as e:
                self.app.notify(f"Failed to save API key: {e}", severity="error")
                return
        if self.cfg is not None:
            display_name = base.split("//")[-1] or pid
            self.cfg.providers[pid] = {"name": display_name, "base_url": base}
            self.custom_providers[pid] = self.cfg.providers[pid]
            self.cfg.provider = pid
            self.cfg.model = model
            try:
                from ..config import save_config

                save_config(self.cfg)
            except Exception as e:
                self.app.notify(f"Failed to save config: {e}", severity="error")
                return
        self.app.notify(f"Added custom provider {pid}/{model}")
        choice = f"{pid}/{model}"
        if self.on_select:
            self.on_select(choice)
        self._close(choice)

    def on_list_view_selected(self, event: Any) -> None:
        if self._rebuilding or self._dismissed:
            return
        index = event.index if event.index is not None else (getattr(event.item, "index", None) or 0)
        self._choose_row(index)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "models-search":
            if self.is_attached and not self._dismissed and self.models:
                self._populate_list()

    def action_refresh_models(self) -> None:
        if self._dismissed:
            return
        if self._query():
            # re-running the worker would clear the search input's siblings;
            # just re-render against the current data instead
            self._populate_list()
        else:
            self._start_worker()

    def action_dismiss_pop(self) -> None:
        self._close(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "models-close":
            self._close(None)
        elif bid == "models-refresh":
            self.action_refresh_models()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self._close(None)
            event.stop()


def _zen_family(model_id: str) -> str:
    """Upstream vendor behind an OpenCode Zen model, from its id prefix.

    Zen's ids carry no provider prefix, but the leading token (gpt, claude,
    gemini, …) reveals the originator; unknown prefixes land in "Other".
    """
    import re

    match = re.match(r"^[a-z]+", model_id.lower())
    prefix = match.group(0) if match else model_id
    families = {
        "claude": "Anthropic",
        "gemini": "Google",
        "gpt": "OpenAI",
        "kimi": "Moonshot",
        "grok": "xAI",
        "deepseek": "DeepSeek",
        "glm": "Zhipu AI",
        "minimax": "MiniMax",
        "qwen": "Alibaba",
        "nemotron": "NVIDIA",
        "mimo": "Xiaomi",
        "laguna": "Poolside",
        "north": "Cohere",
        "big": "Other",
        "hy": "Other",
        "ling": "Other",
    }
    return families.get(prefix, "Other")


def _model_row_label(idx: str, m: dict, current: str) -> Label:
    """Build a model row matching opencode: name + FREE tag, current marked."""
    name = m.get("name") or m["id"]
    free = bool(m.get("free"))
    marked = idx == current or m["id"] == current
    mark = "[#fab283]●[/] " if marked else "   "
    free_tag = " [#7fd88f]FREE[/]" if free else ""
    return Label(f"{mark}{name}{free_tag}", classes="model-item")


def _fallback_models(pid: str, has_key: bool) -> list[dict]:
    """Bundled model list when the live fetch fails or no key is present."""
    if pid in FREE_PROVIDERS:
        mid = FREE_DEFAULT_MODELS.get(pid)
        return [{"id": mid, "name": mid, "context": 0, "free": True}] if mid else []
    out = []
    for mid in DEFAULT_PAID_MODELS.get(pid, []):
        out.append({"id": mid, "name": mid, "context": 0, "free": False})
    return out


def _format_context(value: Any) -> str:
    """Format a context size for display, tolerating "128k"/"1m" strings and junk."""
    if value is None:
        return "?"
    if isinstance(value, str):
        s = value.strip().lower()
        mult = 1
        if s.endswith("k"):
            mult, s = 1000, s[:-1]
        elif s.endswith("m"):
            mult, s = 1000000, s[:-1]
        try:
            return f"{int(float(s) * mult):,}"
        except (ValueError, TypeError):
            return value
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return "?"
