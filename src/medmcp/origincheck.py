"""Decide whether a browser request reached the workspace from its own page.

The workspace server binds loopback and has no authentication, which is safe
against the network but not against the *browser*: a page on any site the
operator visits can issue cross-origin requests to ``127.0.0.1``, and a
WebSocket upgrade is not covered by the same-origin policy at all — the browser
connects and the server is the only thing that can refuse. Since ``read_file``
and ``grep`` need no approval, an unrefused socket is enough to have the agent
read workspace files and stream them back to that page, with no dialog ever
shown. ``/ws/replay`` is worse: it runs a saved workflow with no permission flow
at all.

Two headers settle it, and both are needed:

``Origin``
    Names the page making the request. Browsers always send it on WebSocket
    upgrades and on cross-origin requests, and cannot be made to forge it.
``Host``
    Names the address the request was sent to. This is what stops DNS
    rebinding, where ``evil.example`` resolves to ``127.0.0.1`` so the page's
    own origin becomes the target and the ``Origin`` check has nothing to catch.

Both must name a loopback address. Anything else is refused unless the operator
listed it in ``MEDMCP_ALLOWED_ORIGINS`` / ``MEDMCP_ALLOWED_HOSTS`` (a deployment
behind a reverse proxy, say) — the default is closed, and a deployment that
needs otherwise says so explicitly.

Loopback is deliberately allowed on *any* port rather than only the server's:
the dev frontend runs on its own port and proxies here, and a page already
served from the operator's own loopback is not the threat this defends against.
"""

from __future__ import annotations

import ipaddress
import os

# The only *name* that means "this machine". Addresses are parsed, not matched
# as strings: "127.0.0.1.evil.example" is a domain an attacker can register, and
# any prefix test on "127." accepts it.
_LOOPBACK_NAMES: frozenset[str] = frozenset({"localhost"})


def _split_host(value: str) -> str:
    """Return the hostname part of a ``host[:port]`` value, without brackets.

    IPv6 literals arrive bracketed (``[::1]:8100``), so the last colon is only a
    port separator when it follows the closing bracket or the value holds one
    colon at most.
    """
    value = value.strip()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _origin_host(origin: str) -> str:
    """Return the host of an ``Origin`` value, or ``""`` when it has none.

    ``null`` (a sandboxed iframe, a ``file://`` page) has no host and is never
    equal to a loopback name, so it falls through to refusal.
    """
    origin = origin.strip()
    if "://" not in origin:
        return ""
    return _split_host(origin.split("://", 1)[1])


def is_loopback(hostname: str) -> bool:
    """Whether *hostname* names this machine.

    An address is parsed and asked whether it is loopback, which covers all of
    127.0.0.0/8 and ``::1`` without string matching. A name is accepted only
    when it is exactly ``localhost``: anything longer is a registrable domain,
    and a prefix or suffix test on it is how lookalikes get through.
    """
    hostname = hostname.strip().lower().rstrip(".")
    if hostname in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _configured(var: str) -> frozenset[str]:
    """Read a comma-separated allowlist from the environment, lowercased."""
    raw = os.environ.get(var, "")
    return frozenset(item.strip().lower() for item in raw.split(",") if item.strip())


def reject_reason(host: str, origin: str) -> str | None:
    """Return why this request must be refused, or ``None`` to let it through.

    *host* and *origin* are the raw header values (empty string when absent).
    A missing ``Origin`` is not a refusal: non-browser clients (the container
    healthcheck, ``curl``, the CLI) send none, and a browser that omits it is
    making a same-origin top-level request. A missing ``Host`` is likewise let
    through — HTTP/1.1 requires it, so its absence means a client that is not a
    browser.
    """
    if host:
        hostname = _split_host(host).lower()
        if not is_loopback(hostname) and hostname not in _configured("MEDMCP_ALLOWED_HOSTS"):
            return f"unexpected Host {host!r}"
    if origin:
        if origin.strip().lower() in _configured("MEDMCP_ALLOWED_ORIGINS"):
            return None
        origin_host = _origin_host(origin)
        if not origin_host or not is_loopback(origin_host):
            return f"cross-origin request from {origin!r}"
    return None
