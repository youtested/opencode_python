"""Measure the webfetch fix: before-fix vs after-fix success rate over a
real URL set. Not part of `pytest` (needs network); run directly:

    python tests/measure_webfetch_value.py

Reports per-URL ok/-- plus overall % for each mode and the improvement the
fix provides. The OLD mode is simulated faithfully: primary UA="opencode",
httpx dead (removed `proxies=` kwarg), wget dead (GNU-only flags rejected by
BusyBox), and UA-only request headers.
"""
import os
import sys
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencode_py.tools import webfetch as wf
from opencode_py.tools import cloudflare_bypass as cb

URLS = [
    "https://example.com",
    "https://news.ycombinator.com",
    "https://github.com",
    "https://en.wikipedia.org/wiki/HTTP",
    "https://lite.cnn.com/en",
    "https://www.bing.com",
    "https://lichess.org",
    "https://httpbin.org/html",
    "https://stackoverflow.com",
    "https://www.reddit.com",
    "https://www.quora.com",
    "https://medium.com",
    "https://www.instagram.com",
    "https://www.tiktok.com",
    "https://www.cnn.com",
    "https://www.politico.com",
]
TIMEOUT = 8


def blocked_or_empty(out: str) -> bool:
    if not out or len(out.strip()) < 60:
        return True
    low = out.lower()
    return any(
        m in low
        for m in [
            "checking your browser",
            "just a moment",
            "enable javascript",
            "verifying you are human",
            "access denied",
            "you need to enable js",
            "cf-chl",
            "attention required",
            "we've detected unusual",
        ]
    )


def run() -> dict:
    results = {}
    for url in URLS:
        try:
            r = wf._webfetch(url, "text", TIMEOUT)
        except Exception as e:  # pragma: no cover
            r = {"error": True, "output": f"exc: {e}"}
        ok = (not r.get("error")) and not blocked_or_empty(r.get("output", ""))
        results[url] = ok
    return results


class OldUltimateBypass(cb.UltimateBypass):
    """Replicates the pre-fix broken state."""

    def _get_headers(self):
        return {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }

    def try_httpx(self, url):
        return {"success": False, "method": "httpx", "error": "TypeError: 'proxies'"}

    def try_wget(self, url):
        return {"success": False, "method": "wget", "error": "wget: no such option: -q"}


def main():
    print("== OLD (before fix) ==")
    with mock.patch.object(cb, "UltimateBypass", OldUltimateBypass), mock.patch.object(
        wf, "BROWSER_UA", "opencode"
    ):
        old = run()

    print("== NEW (after fix) ==")
    new = run()

    print(f"\n{'URL':24} {'old':6} {'new':6}")
    for url in URLS:
        print(
            f"{url.replace('https://', '').replace('www.', '')[:24]:24} "
            f"{'ok' if old[url] else '--':6} {'ok' if new[url] else '--':6}"
        )
    p_old = 100 * sum(old.values()) / len(URLS)
    p_new = 100 * sum(new.values()) / len(URLS)
    rel = (p_new / p_old - 1) * 100 if p_old else float("nan")
    print(f"\nold={p_old:.0f}%  new={p_new:.0f}%  improvement={p_new - p_old:+.0f} pts "
          f"({rel:+.0f}%)")


if __name__ == "__main__":
    main()
