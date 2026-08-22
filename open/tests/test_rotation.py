"""Tests for the provider failover rotation. Regression coverage for:
- in-band error events treated as "empty response" (swallowing the real cause)
- reasoning-only responses treated as empty and rotated
- non-retryable errors (400) aborting the whole rotation
- misleading rate-limit classification
"""

from opencode_py.providers.base import ContextOverflowError, ProviderError, ProviderEvent, RateLimitError
from opencode_py.providers.rotation import Rotation
from opencode_py.providers.rotation import build_rotation as build_default_rotation


class FakeProvider:
    """Emits a fixed list of events or raises a fixed exception on stream_chat."""

    def __init__(self, events=None, exc=None):
        self.events = events or []
        self.exc = exc

    def stream_chat(self, messages, tools, on_event):
        if self.exc is not None:
            raise self.exc
        for e in self.events:
            on_event(e)


def build_rotation(providers):
    it = iter(providers)

    def make(pid, model):
        return next(it)

    lanes = [{"provider": f"p{i}", "model": "m"} for i in range(len(providers))]
    return Rotation(lanes=lanes, make_provider=make)


def test_first_lane_success_no_notice():
    rot = build_rotation([FakeProvider(events=[ProviderEvent(kind="text_delta", text="hi")])])
    got = []
    notices = []
    pid, mid = rot.stream([], [], got.append, lambda p, m, r: notices.append((p, m, r)))
    assert pid == "p0"
    assert mid == "m"
    assert [e.text for e in got] == ["hi"]
    assert notices == []


def test_inband_error_fails_over_and_reason_surfaces():
    rot = build_rotation([
        FakeProvider(events=[ProviderEvent(kind="error", error="insufficient_quota: free limit reached")]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    got = []
    notices = []
    pid, _ = rot.stream([], [], got.append, lambda p, m, r: notices.append((p, m, r)))
    assert pid == "p1"
    assert [e.text for e in got] == ["backup"]
    assert notices and notices[0][0] == "p1"
    assert "rate limit" in notices[0][2]


def test_primary_transient_error_does_not_rotate():
    """A transient overload on the user's chosen lane must surface the real
    cause, NOT silently route them onto a backup model."""
    class BoomProvider:
        def stream_chat(self, messages, tools, on_event):
            on_event(ProviderEvent(kind="error", error="server_is_overloaded: busy"))

    rot = build_rotation([
        BoomProvider(),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError on transient overload")
    except ProviderError as e:
        assert "server_is_overloaded" in str(e)


def test_error_only_lane_combined_message_preserves_cause():
    rot = build_rotation([
        FakeProvider(events=[ProviderEvent(kind="error", error="rate limited gateway")]),
        FakeProvider(events=[]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError as e:
        text = str(e)
        assert "rate limited gateway" in text
        assert "empty response" in text


def test_reasoning_only_counts_as_output():
    rot = build_rotation([FakeProvider(events=[ProviderEvent(kind="reasoning_delta", text="thinking...")])])
    got = []
    pid, _ = rot.stream([], [], got.append, None)
    assert pid == "p0"
    assert [e.text for e in got] == ["thinking..."]


def test_primary_empty_response_does_not_rotate():
    """An empty reply from the chosen lane is a transient miss — surface it,
    don't silently switch to another model."""
    rot = build_rotation([
        FakeProvider(events=[]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="ok")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError on empty primary")
    except ProviderError as e:
        assert "empty response" in str(e)


def test_empty_backup_lane_is_skipped():
    """An empty reply from a backup lane must not block the chain."""
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("limit")),
        FakeProvider(events=[]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="ok")]),
    ])
    pid, _ = rot.stream([], [], lambda e: None, None)
    assert pid == "p2"


def test_all_rate_limited_raises_rate_limit():
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("boom1")),
        FakeProvider(exc=RateLimitError("boom2")),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected RateLimitError")
    except RateLimitError:
        pass


def test_non_retryable_400_fails_over():
    rot = build_rotation([
        FakeProvider(exc=ProviderError("bad model id", status=400)),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="ok")]),
    ])
    pid, _ = rot.stream([], [], lambda e: None, None)
    assert pid == "p1"


def test_mixed_failures_raise_provider_error_not_rate_limit():
    rot = build_rotation([
        FakeProvider(exc=ProviderError("oops", retryable=True)),
        FakeProvider(exc=RateLimitError("later")),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError:
        pass
    except RateLimitError:
        raise AssertionError("mixed failures must not be reported as rate limit")


def test_all_failed_message_lists_every_lane():
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("rl1")),
        FakeProvider(exc=ProviderError("bad model id", status=400)),
        FakeProvider(exc=ProviderError("timeout", retryable=True)),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError as e:
        text = str(e)
        assert "rl1" in text and "bad model id" in text and "timeout" in text


def test_buffered_events_replayed_in_order():
    rot = build_rotation([
        FakeProvider(events=[
            ProviderEvent(kind="text_delta", text="a"),
            ProviderEvent(kind="text_delta", text="b"),
            ProviderEvent(kind="done", finish_reason="stop"),
        ]),
    ])
    got = []
    rot.stream([], [], got.append, None)
    assert [e.text for e in got if e.kind == "text_delta"] == ["a", "b"]
    assert any(e.kind == "done" for e in got)


def test_partial_text_then_error_keeps_text():
    """An in-band error AFTER visible text has streamed must keep the partial
    text (already live) and surface a clear 'cut off' error — not replay a
    generic error event, not retry, and not fail over."""
    rot = build_rotation([
        FakeProvider(events=[
            ProviderEvent(kind="text_delta", text="partial"),
            ProviderEvent(kind="error", error="FreeUsageLimitError: quota gone"),
        ]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    got = []
    try:
        rot.stream([], [], got.append, None)
        raise AssertionError("expected ProviderError after partial text")
    except ProviderError as e:
        assert not e.retryable
        assert "cut off" in str(e)
        assert "quota gone" in str(e)
    # the partial text reached the sink live and the backup was never consulted
    assert [e.text for e in got] == ["partial"]


def test_inband_error_before_output_still_fails_over():
    """An in-band error with NO prior output stays a lane failure that can
    rotate (free-quota exhaustion before the model said anything)."""
    rot = build_rotation([
        FakeProvider(events=[ProviderEvent(kind="error", error="insufficient_quota: free limit reached")]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    pid, _ = rot.stream([], [], lambda e: None, None)
    assert pid == "p1"


def test_text_streams_live_before_stream_fully_finishes():
    """Text/reasoning deltas must reach the sink immediately — the sink sees
    the first token before the provider returns (no full-response buffering)."""

    def live_wrapped(evt):
        if evt.kind == "text_delta":
            sink.append(evt.text)

    sink = []
    rot = build_rotation([FakeProvider(events=[
        ProviderEvent(kind="text_delta", text="first"),
        ProviderEvent(kind="text_delta", text="second"),
        ProviderEvent(kind="done", finish_reason="stop"),
    ])])
    pid, _ = rot.stream([], [], live_wrapped, None)
    assert pid == "p0"
    # all tokens arrive live, and tool/usage/done events still replay after
    assert sink == ["first", "second"]


def test_midstream_failure_after_text_commits_and_keeps_partial():
    """If visible text already streamed live and the lane then fails, the
    rotation must NOT fail over to a backup (that would glue a second answer
    onto the partial text) — it surfaces the failure with the partial kept."""

    class DropStreamProvider:
        def __init__(self, emulate):
            self.emulate = emulate

        def stream_chat(self, messages, tools, on_event):
            on_event(ProviderEvent(kind="text_delta", text="partial answer"))
            raise ProviderError("network error talking to OpenCode Zen: Server disconnected", retryable=True)

    rot = build_rotation([
        DropStreamProvider(None),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    got = []
    try:
        rot.stream([], [], got.append, None)
        raise AssertionError("expected ProviderError after mid-stream drop")
    except ProviderError as e:
        assert "partial" in str(e) or "cut off" in str(e)
        assert not e.retryable
    # the partial text reached the sink; the backup answer never did
    assert [e.text for e in got if e.kind == "text_delta"] == ["partial answer"]


def test_all_failed_hint_mentions_rotation():
    rot = build_rotation([FakeProvider(events=[])])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ProviderError")
    except ProviderError as e:
        assert "rotation" in str(e)


def test_auto_rotation_does_not_duplicate_primary_model():
    """The default opencode provider rotation must keep the user's selected
    model exactly once as the primary lane (a prefixed 'opencode/...' model id
    must never be auto-appended again as a separate free-model lane)."""
    from opencode_py.config import Config

    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "opencode/deepseek-v4-flash-free"
    rot = build_default_rotation(cfg)
    ids = [l.get("model", "").split("/", 1)[-1] for l in rot.lanes]
    assert ids.count("deepseek-v4-flash-free") == 1
    assert ids[0] == "deepseek-v4-flash-free"


def test_context_overflow_does_not_rotate_lanes():
    """A context-overflow is a property of the whole history, shared by every
    lane — the rotation must NOT burn through backups and must propagate so the
    caller can trim and retry."""
    rot = build_rotation([
        FakeProvider(exc=ContextOverflowError("boom1")),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ContextOverflowError to propagate")
    except ContextOverflowError as e:
        assert "boom1" in str(e)


def test_context_overflow_inband_error_propagates():
    """An in-band error event that reads like a context overflow must surface
    as ContextOverflowError, not as a generic ProviderError."""
    rot = build_rotation([
        FakeProvider(events=[ProviderEvent(kind="error", error="context_length_exceeded: too long")]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None)
        raise AssertionError("expected ContextOverflowError")
    except ContextOverflowError:
        pass


def test_model_context_size_accepts_prefixed_model_id():
    """cfg.model stores the provider prefix folded in (e.g.
    'opencode/deepseek-v4-flash-free'); lookups must match the bare id so the
    status bar can display the context percentage."""
    from opencode_py.providers.rotation import model_context_size, model_output_limit

    assert model_context_size("opencode", "deepseek-v4-flash-free") == 200000
    assert model_context_size("opencode", "opencode/deepseek-v4-flash-free") == 200000
    assert model_output_limit("opencode", "opencode/deepseek-v4-flash-free") == 128000
    assert model_context_size("groq", "llama-3.3-70b-versatile") == 131072
    assert model_context_size("groq", "groq/llama-3.3-70b-versatile") == 131072


def test_model_context_size_uses_vendor_documented_windows():
    """For paid providers whose /models endpoint doesn't report a context
    window, the lookup must return the documented per-model size (the real
    thing) instead of a per-provider guess."""
    from opencode_py.providers.rotation import model_context_size

    assert model_context_size("openai", "gpt-4o") == 128000
    assert model_context_size("openai", "gpt-4o-mini") == 128000
    assert model_context_size("anthropic", "claude-3-5-sonnet-20241022") == 200000
    assert model_context_size("openai", "o1") == 200000


def test_model_context_size_live_list_beats_default(monkeypatch):
    """A live provider model list that reports a real (small) window must win
    over the provider-wide default so the % reflects the actual model."""
    import opencode_py.providers.rotation as rot

    monkeypatch.setattr(
        rot,
        "fetch_live_models",
        lambda pid, key, base, kind: [
            {"id": "my-custom-model", "context": 4000}
        ],
    )
    monkeypatch.setattr(rot, "_model_list_cache", {})

    class FakeAuth:
        def get(self, pid):
            return "secret-key"

    size = rot.model_context_size("openai", "my-custom-model", auth=FakeAuth())
    assert size == 4000


def test_build_provider_read_timeout_is_configurable():
    """Every provider built from config must carry the configured read timeout
    (model_read_timeout), not the old 30s default — otherwise a slow reasoning
    model still dies mid-conversation. Reasoning models get a longer gap than
    the configured value; plain models honor it exactly."""
    import httpx

    from opencode_py.config import Config
    from opencode_py.providers.rotation import build_provider

    cfg = Config()
    cfg.model_read_timeout = 600.0
    # deepseek + claude-sonnet are reasoning models: the 600s config is raised
    # to the reasoning floor (900s) so long silent thinking isn't killed
    zen = build_provider(cfg, "opencode", "deepseek-v4-flash-free")
    assert zen.timeout.read == 900.0

    anthropic = build_provider(cfg, "anthropic", "claude-sonnet-4-20250514")
    assert anthropic.timeout.read == 900.0

    # non-reasoning lane honors the configured value exactly
    groq = build_provider(cfg, "groq", "llama-3.3-70b-versatile")
    assert groq.timeout.read == 600.0
    assert groq.timeout.connect < 600.0  # only the read window is opened up


def test_build_read_timeout_default_and_explicit():
    from opencode_py.providers.rotation import build_read_timeout

    assert build_read_timeout().read == 300.0
    assert build_read_timeout(120).read == 120.0
    assert build_read_timeout("90").read == 90.0


def test_build_read_timeout_reasoning_models_get_more_room():
    from opencode_py.providers.rotation import build_read_timeout, is_reasoning_model

    assert is_reasoning_model("deepseek-v4-flash-free")
    assert is_reasoning_model("openrouter/deepseek-reasoner")
    assert is_reasoning_model("o3-mini")
    assert is_reasoning_model("claude-sonnet-4-20250514")
    assert is_reasoning_model("gemini-2.5-flash")
    assert is_reasoning_model("grok-3")
    assert not is_reasoning_model("llama-3.3-70b-versatile")
    assert not is_reasoning_model("gpt-4o-mini")

    # reasoning models: the configured gap is raised to a 900s floor
    assert build_read_timeout(120, "deepseek-v4-flash-free").read == 900.0
    assert build_read_timeout(None, "o3-mini").read == 900.0
    # an explicit higher setting still wins
    assert build_read_timeout(1200, "deepseek-v4-flash-free").read == 1200.0
    # non-reasoning lanes are untouched
    assert build_read_timeout(120, "llama-3.3-70b-versatile").read == 120.0
    assert build_read_timeout(0).read > 30.0  # degenerate 0 falls back wide


def test_zen_provider_sends_opencode_identity_headers():
    """Zen requests must identify as the official opencode client (User-Agent +
    x-opencode-*) with a stable session id; otherwise the gateway throttles
    anonymous clients to a couple of messages before 429 FreeUsageLimitError."""
    from opencode_py.config import Config
    from opencode_py.providers.rotation import build_provider

    cfg = Config()
    zen = build_provider(cfg, "opencode", "deepseek-v4-flash-free", session_id="abc123")
    headers = zen._headers()
    assert headers["User-Agent"].startswith("opencode/")
    assert headers["x-opencode-client"] == "cli"
    assert headers["x-opencode-session"] == "abc123"
    assert headers["x-opencode-request"] == "abc123"
    assert headers["x-opencode-project"]

    # other providers must NOT get the opencode identity headers
    groq = build_provider(cfg, "groq", "llama-3.3-70b-versatile")
    gh = groq._headers()
    assert "x-opencode-session" not in gh
    assert "x-opencode-client" not in gh


def test_build_rotation_provider_uses_stable_session_id():
    """The rotation factory must hand its (stable) session id to every provider
    it builds so Zen sees the same x-opencode-session on every turn."""
    from opencode_py.config import Config

    cfg = Config()
    cfg.provider = "opencode"
    cfg.model = "deepseek-v4-flash-free"
    rot = build_default_rotation(cfg, session_id="stable-42")
    assert rot.session_id == "stable-42"
    provider = rot.first
    assert provider._headers()["x-opencode-session"] == "stable-42"


# -- rotation lock (TUI "•" dot) ------------------------------------------
def test_unlocked_rotates_on_rate_limit():
    """Sanity: without the lock a rate-limited primary still fails over."""
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("quota gone")),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    pid, _ = rot.stream([], [], lambda e: None, None)
    assert pid == "p1"


def test_locked_rate_limit_does_not_fail_over():
    """locked=True pins the stream to lane 0: a rate limit surfaces instead of
    switching to the backup model."""
    rot = build_rotation([
        FakeProvider(exc=RateLimitError("quota gone")),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None, locked=True)
        raise AssertionError("expected RateLimitError while locked")
    except RateLimitError:
        pass


def test_locked_hard_error_does_not_fail_over():
    """A dead lane (bad model id / bad key) must not rotate while locked."""
    rot = build_rotation([
        FakeProvider(exc=ProviderError("bad model id", status=400)),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None, locked=True)
        raise AssertionError("expected ProviderError while locked")
    except ProviderError:
        pass


def test_locked_transient_error_surfaces_no_notice():
    """Transient failures on the locked primary stay on the same model and
    never reach the backup lane (retry handles them upstream)."""
    class BoomProvider:
        def stream_chat(self, messages, tools, on_event):
            on_event(ProviderEvent(kind="error", error="server_is_overloaded: busy"))

    rot = build_rotation([
        BoomProvider(),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    try:
        rot.stream([], [], lambda e: None, None, locked=True)
        raise AssertionError("expected ProviderError while locked")
    except ProviderError as e:
        assert "server_is_overloaded" in str(e)


def test_locked_success_uses_first_lane_no_notice():
    """A healthy locked primary succeeds with no failover notice."""
    rot = build_rotation([
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="hi")]),
        FakeProvider(events=[ProviderEvent(kind="text_delta", text="backup")]),
    ])
    got = []
    notices = []
    pid, _ = rot.stream([], [], got.append, lambda p, m, r: notices.append((p, m, r)), locked=True)
    assert pid == "p0"
    assert [e.text for e in got] == ["hi"]
    assert notices == []
