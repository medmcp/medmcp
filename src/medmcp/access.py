"""Who may use the workspace's API: an optional password and the agent's network.

Binding to loopback keeps the network out, but not the other accounts on the
same machine: any of them can open ``127.0.0.1:8100``. On a shared machine,
``MEDMCP_PASSWORD`` closes that — every ``/api`` and ``/ws`` request must then
carry it, as the cookie a browser receives from ``POST /api/login`` or as
``Authorization: Bearer`` for a script. Unset, the API is open to whoever can
reach the port, which on a single-user machine is only its owner.

The isolated agent is a different matter and never depends on a password: it
shares a docker network with the core, so it could reach the UI port and drive
the server that is meant to confine it (replay runs stack tools without asking).
Requests from that network are refused outright (:func:`refuse_network`).
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import os
from functools import lru_cache
from http.cookies import CookieError, SimpleCookie

log: logging.Logger = logging.getLogger(__name__)

PASSWORD_ENV = "MEDMCP_PASSWORD"
COOKIE_NAME = "medmcp_access"
# Browsers cap a cookie's lifetime at 400 days; logging in again after that is fine.
COOKIE_MAX_AGE_SEC = 400 * 24 * 3600
_SHORT_PASSWORD = 12

# Paths that answer without the password: logging in itself. Everything outside
# /api and /ws (the page, its bundle, /healthz) carries nothing to protect.
_OPEN_API_PATHS: frozenset[str] = frozenset({"/api/login"})

_refused: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []


@lru_cache(maxsize=1)
def password() -> str:
    """The configured password, or ``""`` when the workspace has none."""
    value = os.environ.get(PASSWORD_ENV, "").strip()
    if value and len(value) < _SHORT_PASSWORD:
        log.warning("%s is shorter than %d characters", PASSWORD_ENV, _SHORT_PASSWORD)
    return value


def required() -> bool:
    """Whether API requests must carry the password."""
    return bool(password())


def cookie_value() -> str:
    """What a logged-in browser's cookie holds: derived from the password, not the password."""
    return hmac.new(password().encode(), b"medmcp-access-cookie", hashlib.sha256).hexdigest()


def password_matches(candidate: str) -> bool:
    """Whether *candidate* is the password (constant-time)."""
    return bool(candidate) and hmac.compare_digest(candidate.encode(), password().encode())


def needs_password(path: str) -> bool:
    """Whether a request to *path* must carry the password."""
    return required() and path.startswith(("/api/", "/ws/")) and path not in _OPEN_API_PATHS


def authorized(headers: dict[str, str]) -> bool:
    """Whether a request carries the password: as a bearer header or the login cookie."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return password_matches(auth[7:].strip())
    try:
        jar = SimpleCookie(headers.get("cookie", ""))
    except CookieError:
        return False
    morsel = jar.get(COOKIE_NAME)
    return morsel is not None and hmac.compare_digest(
        morsel.value.encode(), cookie_value().encode()
    )


def refuse_network(cidr: str) -> None:
    """Refuse every request from *cidr* (the isolated agent's network)."""
    _refused.append(ipaddress.ip_network(cidr, strict=False))


def peer_refused(host: str | None) -> bool:
    """Whether a request from *host* comes from a refused network."""
    if not host or not _refused:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in net for net in _refused)
