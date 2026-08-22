"""webfetch tool: fetch URL -> markdown/text with size caps."""

from __future__ import annotations

import re
from typing import Any, Callable

from .registry import Tool, schema_with

# The Cloudflare bypass module pulls `requests` (and, when present on the
# phone, optional curl_cffi/cloudscraper), so it must NOT load at startup —
# opencode_py boots in headless/TUI mode without it until a fetch is actually
# blocked. `_ensure_bypass()` imports it lazily on first blocked fetch. The
# attributes stay module-level so tests that patch `webfetch.UltimateBypass`
# keep working (a patch makes it truthy before any lazy import runs).
UltimateBypass = None  # type: ignore[assignment]
CLOUDFLARE_BYPASS_AVAILABLE = False


def _ensure_bypass() -> bool:
    """Import the Cloudflare-bypass module on first use; True when usable."""
    global UltimateBypass, CLOUDFLARE_BYPASS_AVAILABLE
    if CLOUDFLARE_BYPASS_AVAILABLE or UltimateBypass is not None:
        # already imported, or the tests patched it — either way it's usable
        return True
    try:
        from .cloudflare_bypass import UltimateBypass as _UB  # type: ignore

        UltimateBypass = _UB
        CLOUDFLARE_BYPASS_AVAILABLE = True
        return True
    except Exception:  # pragma: no cover - library must not hard-fail
        return False

MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5 MB
DEFAULT_TIMEOUT = 30

MAX_BATCH_URLS = 50
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_BATCH_LIMIT = 8000  # chars of content returned per URL in a batch

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


FORMATS = ("markdown", "text", "html")

# content-types that are never worth decoding to text (binary mojibake)
NON_TEXT_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "font/",
    "application/octet-stream",
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/vnd.rar",
    "application/x-bzip2",
    "application/x-lzma",
    "application/xz",
)


def _clamp_int(value, default: int, lo: int, hi: int) -> int:
    """Coerce a caller-supplied integer while clamping to a sane range.

    Calls reach tools with anything the model types (strings, None, "abc"),
    so trusting ``int(value)``/``min(max)`` directly can crash a fetch. Returns
    ``default`` for non-numeric input instead of raising.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(v, hi))


def _is_non_text(content_type: str) -> bool:
    ct = (content_type or "").lower()
    for prefix in NON_TEXT_PREFIXES:
        if ct.startswith(prefix):
            return True
    return False


def _html_to_text(html: str) -> str:
    """Crude HTML -> text (strip tags/scripts/styles). Good enough for v1."""
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", "", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|li|h[1-6]|tr|pre|blockquote)>", "\n", html)
    html = re.sub(r"(?is)<[^>]+>", "", html)
    import html as h

    text = h.unescape(html)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _html_to_markdown(html: str) -> str:
    """Approximate HTML -> markdown. A real turndown port is Phase 2 polish."""
    text = _html_to_text(html)
    return text


def _looks_like_block(content: str, status: int) -> bool:
    """Heuristic detection of a Cloudflare / challenge page body.

    A status code alone isn't enough and neither is a keyword: the whole point
    is to NOT route a perfectly good 200 page (e.g. an article about Cloudflare
    or a demo captcha) through the slow bypass cascade. Strong markers are
    challenge-specific and almost never appear in legit content; weak markers
    (the brand/tech words) only count on a non-200 status or when the page is
    small — challenge pages are small, full docs are not.
    """
    if not content:
        return False
    lower = content.lower()
    strong = [
        "checking your browser",
        "just a moment",
        "verifying you are human",
        "attention required",
        "cf-challenge",
        "cf-chl",
    ]
    weak = ["cloudflare", "captcha", "enable javascript", "cf-mitigated"]
    small = len(content) < 256 * 1024
    hit_strong = any(m in lower for m in strong)
    hit_weak = any(m in lower for m in weak)
    if hit_strong:
        return True
    if status != 200 and small and hit_weak:
        return True
    if status in (401, 403, 429, 503, 504) and hit_weak:
        return True
    return False


def _convert_body(body: str, format: str, content_type: str) -> str:
    """Apply the requested format conversion to a body string (text or HTML)."""
    if format == "text":
        if "text/html" in content_type:
            return _html_to_text(body)
        return body
    if format == "html":
        return body
    # markdown (default)
    if "text/html" in content_type:
        return _html_to_markdown(body)
    return body


def _webfetch(
    url: str,
    format: str = "markdown",
    timeout: int = DEFAULT_TIMEOUT,
    is_interrupted: Callable[[], bool] | None = None,
    registry: Any | None = None,
) -> dict:
    if not re.match(r"^https?://", url):
        return {"output": "URL must start with http:// or https://", "error": True}
    if format not in FORMATS:
        format = "markdown"
    timeout = _clamp_int(timeout, DEFAULT_TIMEOUT, 1, 120)
    # Local hosts may serve plain HTTP; don't force-upgrade them.
    local = url.startswith(("http://localhost", "http://127.0.0.1", "http://[::1]"))
    upgraded = url.startswith("http://") and not local
    if upgraded:
        url = "https://" + url[len("http://"):]
    headers = {
        "User-Agent": BROWSER_UA,
        "Accept": {
            "markdown": "text/markdown, text/plain;q=0.9, text/html;q=0.5, */*;q=0.1",
            "text": "text/plain, text/markdown;q=0.9, text/html;q=0.5, */*;q=0.1",
            "html": "text/html, */*;q=0.8",
        }.get(format, "*/*"),
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    }
    truncated_note = ""

    def interrupted() -> dict:
        return {
            "output": "(interrupted)",
            "error": True,
            "interrupted": True,
            "stopped": True,
            "metadata": {"upgraded_to_https": upgraded},
        }

    def _wants_stop() -> bool:
        return is_interrupted is not None and bool(is_interrupted())

    try:
        import httpx

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url, headers=headers) as resp:
                content_type = resp.headers.get("content-type", "")
                # Don't decode images/binaries into 5 MB of mojibake: a quick
                # content-type check (before streaming the body) short-circuits
                # media, PDFs and archives with a short notice.
                if _is_non_text(content_type):
                    resp.close()
                    return {
                        "output": (
                            f"Remote content is non-text ({content_type or 'unknown type'}). "
                            "The body was not fetched."
                        ),
                        "metadata": {"content_type": content_type, "upgraded_to_https": upgraded},
                    }
                if _wants_stop():
                    resp.close()
                    return interrupted()
                parts = []
                size = 0
                hit_cap = False
                # Register the response so an interrupt (2nd ESC / Ctrl+C) can
                # force-close the socket and wake a blocked read instantly;
                # the tool then polls the shared interrupt flag and aborts.
                registered = registry is not None
                if registered:
                    try:
                        registry.register_fetch(resp)  # type: ignore[attr-defined]
                    except Exception:
                        registered = False
                try:
                    for chunk in resp.iter_bytes():
                        if _wants_stop():
                            resp.close()
                            return interrupted()
                        room = MAX_RESPONSE_SIZE - size
                        if room <= 0:
                            hit_cap = True
                            truncated_note = f"\n\n[Response truncated at {MAX_RESPONSE_SIZE} bytes]"
                            # Stop reading and release the connection immediately.
                            # Draining the remainder (old behavior) lets an endless
                            # streaming response hang the tool forever and defeats
                            # the byte cap. The httpx context manager also closes on
                            # exit; resp.close() frees the socket right now.
                            resp.close()
                            break
                        parts.append(chunk[:room])
                        size += len(chunk[:room])
                finally:
                    if registered:
                        try:
                            registry.unregister_fetch(resp)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                status = resp.status_code
        body = b"".join(parts).decode("utf-8", errors="replace")
    except httpx.HTTPError as e:
        # A forced abort (registry.abort_fetches) closed the response while we
        # were blocked reading — that's an interrupt, not a network failure.
        if _wants_stop():
            return interrupted()
        msg = f"Fetch failed: {e}"
        if upgraded:
            msg += " (the http:// URL was upgraded to https://)"
        return {"output": msg, "error": True}

    if status == 200 and not _looks_like_block(body, status):
        return {"output": _convert_body(body, format, content_type) + truncated_note,
                "metadata": {"upgraded_to_https": upgraded}}
    # Try the cascade for ANY failed status (not just challenge-looking bodies):
    # a bare 403 page is often only a bot wall that a different method or UA
    # (requests vs httpx vs curl) slips past (e.g. wikipedia/reddit 403 on
    # httpx but 200 via requests/curl).
    if _ensure_bypass():
        return _bypass_fetch(url, format, timeout, upgraded, hint=status, is_interrupted=is_interrupted)
    return {"output": f"Fetch failed: HTTP {status}", "error": True}


def _bypass_fetch(
    url: str,
    format: str,
    timeout: int,
    upgraded: bool,
    hint: int = 0,
    is_interrupted: Callable[[], bool] | None = None,
) -> dict:
    """Try the Cloudflare bypass cascade with optional per-attempt IP rotation.

    Rotation sources (no Tor required):
      * OPENCODE_PROXY_POOL  — a comma/space list of proxy URLs.
      * OPENCODE_HARVEST_PROXIES=1 — auto-pull free public proxies.
    Rotating means a blocked attempt retries via a fresh exit IP, which dodges
    per-IP rate limiting; it does not change the blocked-JS-challenge outcome.
    """
    if is_interrupted is not None and is_interrupted():
        return {"output": "(interrupted)", "error": True, "interrupted": True, "stopped": True}
    try:
        from .cloudflare_bypass import ProxyPool

        # Shared process-wide pool: parallel fetches (webfetch_many) all rotate
        # through the same proxy list and the free-proxy harvest runs once.
        # Falls back to an empty pool when no rotation is configured.
        ub = UltimateBypass(timeout=timeout, proxy_pool=ProxyPool.shared())
        result = ub.fetch(url)
    except Exception as e:  # pragma: no cover
        return {"output": f"Fetch failed (bypass): {e}", "error": True}
    if is_interrupted is not None and is_interrupted():
        return {"output": "(interrupted)", "error": True, "interrupted": True, "stopped": True}

    if not result.get("success"):
        err = result.get("error") or "Unknown"
        msg = f"Fetch blocked (HTTP {hint}) and bypass failed: {err}"
        if upgraded:
            msg += " (the http:// URL was upgraded to https://)"
        return {"output": msg, "error": True}

    content = result.get("content", "")
    content_type = "text/html; charset=utf-8"
    truncated_note = ""
    if len(content) > MAX_RESPONSE_SIZE:
        content = content[:MAX_RESPONSE_SIZE]
        truncated_note = f"\n\n[Response truncated at {MAX_RESPONSE_SIZE} bytes]"
    return {
        "output": _convert_body(content, format, content_type) + truncated_note,
        "metadata": {
            "bypassed": True,
            "bypass_method": result.get("method"),
            "upgraded_to_https": upgraded,
        },
    }


def tool(registry: Any | None = None) -> Tool:
    description = """- Fetches content from a specified URL
- Takes a URL and optional format as input
- Fetches the URL content, converts to requested format (markdown by default)
- Returns the content in the specified format
- Use this tool when you need to retrieve and analyze web content

Usage notes:
  - IMPORTANT: if another tool is present that offers better web fetching capabilities, is more targeted to the task, or has fewer restrictions, prefer using that tool instead of this one.
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - Format options: "markdown" (default), "text", or "html"
  - If the initial fetch is blocked (any 4xx/5xx or JS-challenge page), webfetch
    retries with a Cloudflare-bypass cascade (requests, cloudscraper, curl-cffi,
    curl, httpx, wget) and returns the bypassed content.
  - IP rotation (no Tor needed): set OPENCODE_PROXY_POOL to a comma/space list
    of proxy URLs, or OPENCODE_HARVEST_PROXIES=1 to auto-pull free public
    proxies; each blocked attempt then retries from a fresh exit IP.
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large"""

    def run(input: dict) -> dict:
        fmt = input.get("format", "markdown")
        if fmt not in FORMATS:
            fmt = "markdown"
        # Read the engine's interrupt callback at call time (the registry hook
        # is installed by AgentLoop.__init__) so ESC/Ctrl+C aborts an in-flight
        # fetch instead of letting it run to its timeout. Each engine (main +
        # sub-agents) owns its own registry, so workers always see their own
        # turn's interrupt state.
        is_interrupted: Callable[[], bool] | None = None
        if registry is not None:
            checker = getattr(registry, "interrupt_check", None)
            if callable(checker):
                is_interrupted = checker
        return _webfetch(
            input["url"],
            fmt,
            input.get("timeout", DEFAULT_TIMEOUT),
            is_interrupted=is_interrupted,
            registry=registry,
        )

    return Tool(
        name="webfetch",
        description=description,
        parameters=schema_with(
            {
                "url": {"type": "string", "description": "The URL to fetch content from"},
                "format": {
                    "type": "string",
                    "description": "The format to return the content in",
                    "enum": ["markdown", "text", "html"],
                    "optional": True,
                },
                "timeout": {"type": "integer", "description": "Timeout in seconds (max 120)", "optional": True},
            },
            ["url"],
        ),
        run=run,
        permission="webfetch",
    )


def _fetch_one(
    url: str,
    format: str,
    timeout: int,
    content_limit: int,
    is_interrupted: Callable[[], bool] | None = None,
    registry: Any | None = None,
) -> tuple:
    """Fetch a single URL via the shared _webfetch cascade; returns
    (url, ok, content, truncated). Runs inside a worker thread."""
    r = _webfetch(url, format, timeout, is_interrupted=is_interrupted, registry=registry)
    if r.get("error"):
        return (url, False, str(r.get("output", "Unknown error")), False)
    out = str(r.get("output", "") or "")
    truncated = False
    if len(out) > content_limit:
        out = out[:content_limit].rstrip()
        truncated = True
    return (url, True, out, truncated)


def webfetch_many(
    urls: list[str],
    format: str = "markdown",
    timeout: int = DEFAULT_TIMEOUT,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    content_limit: int = DEFAULT_BATCH_LIMIT,
    is_interrupted: Callable[[], bool] | None = None,
    registry: Any | None = None,
) -> dict:
    """Fetch many URLs concurrently and return each result keyed by URL.

    Wall time is bounded by the slowest fetch (not the sum), up to
    ``max_concurrent`` workers. Every URL goes through the same primary fetch
    and Cloudflare-bypass cascade as ``_webfetch``, and all parallel fetches
    share the process-wide proxy pool so IP rotation still works under load.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not isinstance(urls, list) or not urls:
        return {"output": "webfetch_many requires a non-empty 'urls' array.", "error": True}
    if not all(isinstance(u, str) for u in urls):
        return {"output": "webfetch_many 'urls' must be an array of strings.", "error": True}

    # Dedupe keeping order, cap the count so the result stays usable.
    seen: list[str] = []
    for u in urls:
        if u not in seen:
            seen.append(u)
    dropped = seen[MAX_BATCH_URLS:]
    urls = seen[:MAX_BATCH_URLS]

    workers = _clamp_int(max_concurrent, DEFAULT_MAX_CONCURRENT, 1, min(10, len(urls)))
    limit = _clamp_int(content_limit, DEFAULT_BATCH_LIMIT, 1, MAX_RESPONSE_SIZE)
    timeout = _clamp_int(timeout, DEFAULT_TIMEOUT, 1, 120)

    results: list = [None] * len(urls)  # deterministic (submission) order
    interrupted = False
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_fetch_one, u, format, timeout, limit, is_interrupted, registry): i
            for i, u in enumerate(urls)
        }
        # Each fetch polls the shared interrupt flag (2nd ESC / Ctrl+C) and
        # aborts itself; an interrupt stops the turn without waiting for every
        # slow URL. Remaining in-flight workers check the flag on their own and
        # return immediately.
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            if is_interrupted is not None and is_interrupted():
                interrupted = True
                break

    if interrupted:
        return {"output": "(interrupted)", "error": True, "interrupted": True, "stopped": True}

    ok = sum(1 for r in results if r[1])
    lines = [f"# Batch fetch results ({len(urls)} urls, {workers} workers)"]
    for i, (url, success, content, truncated) in enumerate(results, 1):
        if success:
            lines.append(f"\n## {i}. {url}  (OK{' [truncated]' if truncated else ''})\n{content}")
        else:
            lines.append(f"\n## {i}. {url}  (FAILED)\n{content}")
    if dropped:
        lines.append(f"\n[note: {len(dropped)} urls beyond the {MAX_BATCH_URLS} cap were dropped]")

    metadata = {
        "count": len(urls),
        "succeeded": ok,
        "failed": len(urls) - ok,
        "dropped": len(dropped),
        "concurrency": workers,
    }
    if dropped:
        metadata["dropped_urls"] = dropped
    return {"output": "\n".join(lines), "metadata": metadata}


def batch_tool(registry: Any | None = None) -> Tool:
    description = """- Fetches MANY URLs in PARALLEL (webfetch fetches one at a time).
- Input is an array of URLs; they are fetched concurrently (default 5 workers)
  and each result is returned prefixed by its number and URL, so wall time is
  the slowest fetch, not the sum.
- Use webfetch_many instead of several sequential webfetch calls whenever you
  need multiple pages at once (research, comparison, multi-page scraping).
- Every URL uses the same fetch stack and Cloudflare-bypass cascade as
  webfetch, and parallel fetches share the IP-rotation pool.
- Each URL's content is capped to content_limit chars (default 8000); raise it
  to get fuller pages, or use the single-URL webfetch for a complete page.
- URLs are auto-deduped and the batch is capped at 50 URLs. This tool is
  read-only and does not modify any files."""

    def run(input: dict) -> dict:
        is_interrupted: Callable[[], bool] | None = None
        if registry is not None:
            checker = getattr(registry, "interrupt_check", None)
            if callable(checker):
                is_interrupted = checker
        return webfetch_many(
            input.get("urls", []),
            format=input.get("format", "markdown"),
            timeout=input.get("timeout") or DEFAULT_TIMEOUT,
            max_concurrent=input.get("max_concurrent") or DEFAULT_MAX_CONCURRENT,
            content_limit=input.get("content_limit") or DEFAULT_BATCH_LIMIT,
            is_interrupted=is_interrupted,
            registry=registry,
        )

    return Tool(
        name="webfetch_many",
        description=description,
        parameters=schema_with(
            {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs to fetch in parallel (auto-deduped, max 50)",
                },
                "format": {
                    "type": "string",
                    "description": "The format to return the content in",
                    "enum": ["markdown", "text", "html"],
                    "optional": True,
                },
                "timeout": {"type": "integer", "description": "Per-fetch timeout in seconds (max 120)", "optional": True},
                "max_concurrent": {
                    "type": "integer",
                    "description": "Max concurrent fetches (1-10, default 5)",
                    "optional": True,
                },
                "content_limit": {
                    "type": "integer",
                    "description": "Max chars of content returned per URL (default 8000)",
                    "optional": True,
                },
            },
            ["urls"],
        ),
        run=run,
        permission="webfetch",
    )
