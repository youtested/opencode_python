"""Low-level network helpers for interrupting stubborn streamed reads."""

from __future__ import annotations

import socket

# Traversal into httpx/httpcore internals is brittle across versions; every
# step is guarded so a layout change just degrades to the old close() behaviour
# instead of raising. The socket is what recv() actually blocks on, and
# closing it from another thread does NOT wake that recv on Linux — only an
# explicit shutdown() does. Reaching the socket lets the interrupt path
# genuinely unblock a stream that's idle mid-"thinking".
_SOCKET_ATTR_CHAIN = (
    "stream._stream._httpcore_stream._stream._connection._network_stream._sock",
)


def socket_of(resp) -> socket.socket | None:
    """Best-effort extraction of the live socket for an in-flight httpx response."""
    target = resp
    for attr in _SOCKET_ATTR_CHAIN[0].split("."):
        try:
            target = getattr(target, attr)
        except Exception:
            return None
    if isinstance(target, socket.socket):
        return target
    return None


def force_close_response(resp) -> None:
    """Close an httpx response AND unblock a reader stuck in ``iter_bytes``.

    ``Response.close()`` alone leaves a blocked recv hanging until the read
    timeout on this platform (the close removes the fd from the table but the
    in-flight syscall keeps its reference). ``socket.shutdown(SHUT_RDWR)``
    resets the connection, which wakes the reader with an EOF/connection-reset
    that the provider's error handling turns into ``StreamInterrupted`` when an
    interrupt is pending — so ESC aborts instantly even mid-thought.

    The shutdown must run BEFORE ``Response.close()``: closing first makes the
    fd invalid so the later ``shutdown()`` is a no-op and the recv is never
    woken.
    """
    if resp is None:
        return
    sock = socket_of(resp)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    try:
        resp.close()
    except Exception:
        pass
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass