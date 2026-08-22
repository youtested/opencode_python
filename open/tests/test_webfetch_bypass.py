"""webfetch tool: Cloudflare-bypass fallback integration tests."""

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from opencode_py.tools import webfetch as wf


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, status, body, ctype="text/html"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/blocked"):
            self._send(
                403,
                "<html><title>Just a moment...</title>"
                "<body>Checking your browser before accessing. "
                "This process is automatic.</body></html>",
            )
        elif self.path.startswith("/plain"):
            self._send(200, "plain hello")
        else:
            self._send(200, "<h1>Real</h1><p>content here</p>")


def _start_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


def test_normal_fetch_converts_html():
    srv, t = _start_server()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/ok"
        r = wf._webfetch(url, "markdown", 10)
        assert r.get("error") is None
        assert r["output"].strip() == "Real\ncontent here"
    finally:
        srv.shutdown()


def test_plain_text_passthrough():
    srv, t = _start_server()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/plain"
        r = wf._webfetch(url, "text", 10)
        assert r.get("error") is None
        assert r["output"].strip() == "plain hello"
    finally:
        srv.shutdown()


def test_blocked_fallback_bypasses():
    srv, t = _start_server()
    fake = {"success": True, "method": "requests", "content": "<h1>Bypassed</h1><p>ok</p>"}
    try:
        with mock.patch.object(wf, "UltimateBypass") as UB:
            UB.return_value.fetch.return_value = fake
            url = f"http://127.0.0.1:{srv.server_port}/blocked"
            r = wf._webfetch(url, "markdown", 10)
            assert r.get("error") is None
            assert r["output"].strip() == "Bypassed\nok"
            assert r["metadata"]["bypassed"] is True
            assert r["metadata"]["bypass_method"] == "requests"
            UB.return_value.fetch.assert_called_once_with(url)
    finally:
        srv.shutdown()


def test_blocked_fallback_failure_reports_error():
    srv, t = _start_server()
    fake = {"success": False, "method": "requests", "error": "Status: 403"}
    try:
        with mock.patch.object(wf, "UltimateBypass") as UB:
            UB.return_value.fetch.return_value = fake
            url = f"http://127.0.0.1:{srv.server_port}/blocked"
            r = wf._webfetch(url, "markdown", 10)
            assert r.get("error") is True
            assert "bypass failed" in r["output"]
            assert "HTTP 403" in r["output"]
    finally:
        srv.shutdown()


def test_looks_like_block_heuristic():
    assert wf._looks_like_block("Just a moment... checking your browser", 200) is True
    assert wf._looks_like_block("<html>cf-chl challenge</html>", 200) is True
    assert wf._looks_like_block("normal content", 200) is False
    assert wf._looks_like_block("", 200) is False
    # a legit 200 page that merely mentions cloudflare is NOT a bot wall
    assert wf._looks_like_block("Cloudflare captcha docs ..." * 500, 200) is False
    # ...but the same words on a non-200 status / tiny page count as blocked
    assert wf._looks_like_block("request blocked by cloudflare", 403) is True


# --------------------------------------------------------------------------
# Regression: the bypass cascade used GNU-only shell command strings
# (`wget -q ... --user-agent="..."`, `shell=True` with the URL interpolated).
# On Termux/Android the system wget (BusyBox/toybox) rejects the GNU-only
# flags, and because wget was the LAST method its misleading error masked the
# real reason all earlier methods failed. Verify the cascade now shells out
# flag-literal argument lists and wget degrades gracefully when -q/-U are
# unsupported.
# --------------------------------------------------------------------------

def _fake_run(script: list):
    """Return a _run_cmd replacement that simulates BusyBox wget rejecting -q."""

    def fake(cmd, timeout=25, env=None):
        if cmd[0] == "wget" and "-q" in cmd[0:5]:
            return (False, "", "wget: error: no such option: -q")
        if cmd[0] == "wget":
            return (True, "<html><body>" + "real via wget " * 20 + "</body></html>", "")
        if cmd[0] == "curl":
            return (True, "<html><body>" + "real via curl " * 20 + "</body></html>", "")
        return (False, "", "no such binary")

    return fake


def test_scripts_run_as_argument_lists_no_shell():
    from opencode_py.tools.cloudflare_bypass import UltimateBypass

    ub = UltimateBypass()
    with mock.patch.object(subprocess, "run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "boom"
        run.return_value.stderr = ""
        ub._run_cmd(["curl", "-A", '"sed-ish"'])  # would fail in a shell
        # never invoked through a shell
        for c in run.call_args.args[:1]:
            assert c == ["curl", "-A", '"sed-ish"']
        assert run.call_args.kwargs.get("shell") is False


def test_wget_skips_unsupported_flag_then_succeeds():
    from opencode_py.tools.cloudflare_bypass import UltimateBypass

    ub = UltimateBypass()
    with mock.patch.object(ub, "_run_cmd", side_effect=_fake_run(None)):
        result = ub.try_wget("https://example.com/page")
        assert result["success"] is True
        assert result["method"] == "wget"
        assert "real via wget" in result["content"]


def test_headers_are_browser_like():
    from opencode_py.tools.cloudflare_bypass import UltimateBypass

    ub = UltimateBypass(user_agent="test-agent")
    ub.use_rotation = False  # keep UA deterministic
    h = ub._get_headers()
    assert h["User-Agent"] == "test-agent"
    for sec in ("sec-ch-ua", "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site"):
        assert sec.upper() in h or any(sec in k.lower() for k in h)
    assert h["Accept-Language"].startswith("en-US")


def test_webfetch_primary_uses_browser_ua():
    assert wf.BROWSER_UA.startswith("Mozilla/5.0")


def test_bypass_failure_now_reports_real_reason_not_wget_flags():
    srv, t = _start_server()
    try:
        with mock.patch.object(wf, "UltimateBypass") as UB:
            UB.return_value.fetch.return_value = {
                "success": False,
                "error": "Status: 403",
            }
            url = f"http://127.0.0.1:{srv.server_port}/blocked"
            r = wf._webfetch(url, "markdown", 10)
            assert r.get("error") is True
            assert "Status: 403" in r["output"]
            assert "no such option" not in r["output"]
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------
# IP rotation: the cascade can retry a blocked fetch through a fresh proxy so
# per-IP rate limiting can't starve it. Pool is env-driven (OPENCODE_PROXY_POOL
# / OPENCODE_HARVEST_PROXIES) and needs no Tor.
# --------------------------------------------------------------------------

def test_proxy_pool_rotates_from_env():
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    with mock.patch.dict(
        os.environ,
        {"OPENCODE_PROXY_POOL": "http://1.2.3.4:8080, http://5.6.7.8:3128"},
        clear=False,
    ):
        pool = ProxyPool(harvest=False)
        assert pool.available is True
        first = pool.next()
        assert first == "http://1.2.3.4:8080"
        assert pool.next() == "http://5.6.7.8:3128"
        assert pool.next() == "http://1.2.3.4:8080"  # wraps around


def test_proxy_pool_honors_custom_list():
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(proxies=["http://a:1", "http://b:2"], harvest=False)
    assert [pool.next(), pool.next()] == ["http://a:1", "http://b:2"]


def test_harvest_keeps_only_validated_proxies_ranked_by_latency():
    """The harvester must admit only proxies that pass a real HTTPS probe, and
    rank the survivors fastest-first so the pool leads with the best IPs."""
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(harvest=False)

    def checker(proxy):
        if proxy == "http://dead:1":
            return None  # fails the HTTPS probe
        lat = {"http://a:1": 0.9, "http://b:2": 0.3, "http://c:3": 2.1}[proxy]
        return (proxy, lat)

    pool._gather_candidates = lambda: ["http://a:1", "http://dead:1", "http://b:2", "http://c:3"]
    got = pool._validate_candidates(pool._gather_candidates(), max_keep=2, checker=checker)
    # dead proxy dropped; fastest survivors kept, in order
    assert got == ["http://b:2", "http://a:1"]


def test_harvest_validation_rejects_blocked_body():
    """A proxy that tunnels but returns a bot-wall page is not 'the best IP' —
    it must be filtered out by the same block heuristic the cascade uses."""
    from opencode_py.tools.cloudflare_bypass import ProxyPool, _check_block

    pool = ProxyPool(harvest=False)
    blocked_html = "<html><title>Just a moment...</title><body>Checking your browser...</body></html>"
    assert _check_block(blocked_html, 200) is True

    def checker(proxy):
        return None  # simulates a blocked response failing the probe

    assert pool._validate_candidates(["http://x:1"], checker=checker) == []


# --------------------------------------------------------------------------
# Disk cache: the validated proxy list is saved to ~/.cache/opencode/proxies.json
# and re-used on later starts so the expensive ~30s harvest doesn't rerun every
# process boot.
# --------------------------------------------------------------------------

def test_proxy_cache_roundtrip(tmp_path, monkeypatch):
    import opencode_py.tools.cloudflare_bypass as cfb

    monkeypatch.setattr(cfb, "PROXY_CACHE_PATH", tmp_path / "proxies.json")
    monkeypatch.setattr(cfb, "PROXY_CACHE_TTL", 600)
    pool = cfb.ProxyPool(harvest=False)
    pool._save_cache(["http://a:1", "http://b:2"])
    assert pool._load_cache() == ["http://a:1", "http://b:2"]


def test_proxy_cache_stale_is_rejected(tmp_path, monkeypatch):
    import time

    import opencode_py.tools.cloudflare_bypass as cfb

    monkeypatch.setattr(cfb, "PROXY_CACHE_PATH", tmp_path / "proxies.json")
    monkeypatch.setattr(cfb, "PROXY_CACHE_TTL", 10)
    pool = cfb.ProxyPool(harvest=False)
    pool._save_cache(["http://a:1"])
    # backdate the cache past the TTL
    cfb.PROXY_CACHE_PATH.write_text(
        json.dumps({"ts": time.time() - 60, "proxies": ["http://a:1"]})
    )
    assert pool._load_cache() is None
    # but a stale cache is still usable as a last-resort fallback
    assert pool._load_cache(ignore_ttl=True) == ["http://a:1"]


def test_harvest_or_cache_reuses_fresh_cache(tmp_path, monkeypatch):
    import opencode_py.tools.cloudflare_bypass as cfb

    monkeypatch.setattr(cfb, "PROXY_CACHE_PATH", tmp_path / "proxies.json")
    monkeypatch.setattr(cfb, "PROXY_CACHE_TTL", 600)
    pool = cfb.ProxyPool(harvest=False)
    pool._save_cache(["http://cached:1"])

    def boom():
        raise AssertionError("harvest must not run when a fresh cache exists")

    pool._harvest = boom
    assert pool._harvest_or_cache() == ["http://cached:1"]


def test_harvest_or_cache_harvests_when_cache_stale(tmp_path, monkeypatch):
    import opencode_py.tools.cloudflare_bypass as cfb

    monkeypatch.setattr(cfb, "PROXY_CACHE_PATH", tmp_path / "proxies.json")
    monkeypatch.setattr(cfb, "PROXY_CACHE_TTL", 0)  # cache disabled
    pool = cfb.ProxyPool(harvest=False)
    pool._harvest = lambda: ["http://fresh:1", "http://fresh:2"]
    got = pool._harvest_or_cache()
    assert got == ["http://fresh:1", "http://fresh:2"]
    # a fresh harvest is written back to disk for the next process
    assert pool._load_cache(ignore_ttl=True) == ["http://fresh:1", "http://fresh:2"]


# --------------------------------------------------------------------------
# Self-healing: a proxy that fails N consecutive fetches is demoted and skipped,
# a success resets its streak, and a fully-demoted pool refills itself from the
# cache/refresh path instead of staying dead for the rest of the process.
# --------------------------------------------------------------------------

def test_proxy_pool_demotes_dead_proxy_and_revives_on_success():
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(proxies=["http://a:1", "http://b:2"], harvest=False)
    pool.max_fails = 2
    pool.mark_failure("http://a:1")
    pool.mark_failure("http://a:1")
    # 'a' is demoted — next() must skip it and hand out 'b'
    assert pool.next() == "http://b:2"
    # a success clears the streak and revives 'a' into rotation
    pool.mark_success("http://a:1")
    assert pool.next() == "http://a:1"


def test_proxy_pool_all_dead_flags_and_refills():
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(proxies=["http://a:1", "http://b:2"], harvest=False)
    pool.max_fails = 1
    assert pool.all_dead() is False
    pool.mark_failure("http://a:1")
    pool.mark_failure("http://b:2")
    assert pool.all_dead() is True
    # refill pulls a fresh list (cache/refresh) and clears the demotions
    pool._harvest_or_cache = lambda: ["http://fresh:1"]
    assert pool.refill() is True
    assert pool.next() == "http://fresh:1"
    assert pool.all_dead() is False


def test_fetch_demotes_failed_proxy_and_refills_when_pool_all_dead():
    """A full-cascade failure through a proxy must demote it; once every proxy
    is demoted the pool refills instead of leaving the fetch to die."""
    from opencode_py.tools.cloudflare_bypass import ProxyPool, UltimateBypass

    pool = ProxyPool(proxies=["http://p1:1"], harvest=False)
    pool.max_fails = 1
    ub = UltimateBypass(timeout=1, proxy_pool=pool, ip_retries=1)
    ub.try_basic_request = lambda url: {"success": False, "error": "Status: 403"}
    for m in ("cloudscraper", "curl_cffi", "curl", "httpx", "wget"):
        setattr(ub, f"try_{m}", lambda url: {"success": False, "error": "skip"})

    refilled = []
    pool.refill = lambda: refilled.append(True) or False
    result = ub.fetch("https://example.com")
    assert result["success"] is False
    # the lone proxy was demoted, and the pool tried to refill itself
    assert pool._dead == {"http://p1:1"}
    assert refilled == [True]


def test_fetch_success_marks_proxy_healthy():
    """A successful fetch through a proxy must reset its failure streak."""
    from opencode_py.tools.cloudflare_bypass import ProxyPool, UltimateBypass

    pool = ProxyPool(proxies=["http://p1:1"], harvest=False)
    pool.max_fails = 1
    pool.mark_failure("http://p1:1")  # would be dead, but a success revives it
    ub = UltimateBypass(timeout=1, proxy_pool=pool, ip_retries=0)
    ub.try_basic_request = lambda url: {"success": True, "content": "<h1>ok</h1>"}
    for m in ("cloudscraper", "curl_cffi", "curl", "httpx", "wget"):
        setattr(ub, f"try_{m}", lambda url: {"success": False, "error": "skip"})
    result = ub.fetch("https://example.com")
    assert result["success"] is True
    assert "http://p1:1" not in pool._dead
    assert pool._fails.get("http://p1:1") is None


def test_fetch_retries_with_fresh_proxy_when_blocked():
    from opencode_py.tools.cloudflare_bypass import ProxyPool, UltimateBypass

    pool = ProxyPool(proxies=["http://p1:1", "http://p2:2"], harvest=False)
    ub = UltimateBypass(timeout=1, proxy_pool=pool, ip_retries=2)

    def fake_methods(url):
        ub.stats["methods_tried"] += 1
        if ub.proxy == "http://p2:2":
            return {"success": True, "method": "requests", "content": "bypassed via p2"}
        return {"success": False, "method": "requests", "error": "Status: 403"}

    with mock.patch.object(ub, "try_basic_request", side_effect=fake_methods):
        # stub the rest so only requests runs
        for m in ("cloudscraper", "curl_cffi", "curl", "httpx", "wget"):
            mock.patch.object(
                ub, f"try_{m}", return_value={"success": False, "error": "skip"}
            ).start()
        result = ub.fetch("https://example.com")
        assert result["success"] is True
        assert result["proxy"] == "http://p2:2"
        assert result["stats"]["success_method"] == "requests"


def test_rotation_wired_through_bypass_fetch():
    srv, t = _start_server()
    try:
        with mock.patch.dict(
            os.environ,
            {"OPENCODE_PROXY_POOL": "http://127.0.0.1:1"},  # invalid but non-empty
            clear=False,
        ):
            with mock.patch.object(wf, "UltimateBypass") as UB:
                UB.return_value.fetch.return_value = {
                    "success": True,
                    "method": "requests",
                    "content": "<h1>ok</h1>",
                }
                url = f"http://127.0.0.1:{srv.server_port}/blocked"
                r = wf._webfetch(url, "markdown", 10)
                assert r.get("error") is None
                # a pool was constructed and handed to the bypass
                pool = UB.call_args.kwargs["proxy_pool"]
                assert pool.next() == "http://127.0.0.1:1"
    finally:
        srv.shutdown()


def test_session_gateway_rotates_token_per_fetch():
    """A session-based gateway entry must exit from a fresh IP per fetch:
    begin_fetch() mints a new token, next() rewrites the URL to carry it,
    and the token is stable across one fetch's attempts."""
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(
        proxies=["http://user-sess-XXX:pass@gw:8080", "http://plain:1"], harvest=False
    )
    # before begin_fetch no session is active -> returned as-is
    assert pool.next() == "http://user-sess-XXX:pass@gw:8080"
    # after begin_fetch the session gateway gets a NEW random token
    pool.begin_fetch()
    seen = [pool.next() for _ in range(3)]
    sess = {u for u in seen if "-sess-" in u}
    assert sess and all(u.startswith("http://user-sess-") and "-XXX" not in u for u in sess)
    # plain proxies still pass through untouched
    assert "http://plain:1" in seen
    # the session is pinned for the duration of one fetch
    token = None
    for _ in range(3):
        u = pool.next()
        if "-sess-" not in u:
            continue
        tok = u.split("-sess-")[1].split(":")[0]
        token = token or tok
        assert tok == token
    # the next fetch rotates again
    first_url = sess.pop()
    first_token = first_url.split("-sess-")[1].split(":")[0]
    pool.begin_fetch()
    new_tokens = {
        u.split("-sess-")[1].split(":")[0]
        for _ in range(6)
        if "-sess-" in (u := pool.next())
    }
    assert new_tokens and first_token not in new_tokens


def test_session_gateway_passes_through_when_no_session_marker():
    """Plain proxies must be returned untouched even when the pool also holds a
    session gateway."""
    from opencode_py.tools.cloudflare_bypass import ProxyPool

    pool = ProxyPool(proxies=["http://a:1", "http://b:2"], harvest=False)
    pool.begin_fetch()
    assert pool.next() == "http://a:1"
    assert pool.next() == "http://b:2"


def test_fetch_uses_fresh_session_per_call():
    """Each UltimateBypass.fetch() must begin a new session for the gateway,
    so consecutive fetches leave from different IPs."""
    from opencode_py.tools.cloudflare_bypass import UltimateBypass

    pool = mock.MagicMock()
    pool.next.return_value = "http://user-sess-TOK:pass@gw:8080"
    ub = UltimateBypass(timeout=5, ip_retries=0, proxy_pool=pool)

    def fake_methods(url):
        return {"success": True, "method": "requests", "content": "<h1>ok</h1>"}

    with mock.patch.object(ub, "try_basic_request", side_effect=fake_methods):
        for m in ("cloudscraper", "curl_cffi", "curl", "httpx", "wget"):
            mock.patch.object(
                ub, f"try_{m}", return_value={"success": False, "error": "skip"}
            ).start()
        ub.fetch("https://example.com")
        ub.fetch("https://example.com")

    assert pool.begin_fetch.call_count == 2
    assert pool.end_fetch.call_count == 2