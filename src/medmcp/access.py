"""The workspace's access token: who may use the API on this machine.

Binding to loopback keeps the network out, but not the other accounts on the
same machine, nor anything else running locally — a request with no ``Origin``
is a non-browser client, which the origin check lets through by design. So every
``/api`` and ``/ws`` request must also carry this token: as the cookie a browser
receives when it is paired (``POST /api/pair``), or as ``Authorization: Bearer``
for a script.

The token is random, generated once and kept in ``.vibe/state/ui_token.json``
(0600, on the persisted state volume in a container), so a restart or an update
does not unpair a browser. ``MEDMCP_UI_TOKEN`` sets it instead, for a deployment
or a harness that needs to know it up front. ``medmcp url`` prints the pairing
link, and the server logs it at start.
"""

from __future__ import annotations

import logging
import os
import secrets
from functools import lru_cache
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from urllib.parse import quote

from medmcp.acp import VIBE_HOME
from medmcp.statefile import atomic_write_json, read_json

log: logging.Logger = logging.getLogger(__name__)

TOKEN_PATH: Path = VIBE_HOME / "state" / "ui_token.json"
TOKEN_ENV = "MEDMCP_UI_TOKEN"
COOKIE_NAME = "medmcp_access"
# Browsers cap a cookie's lifetime at 400 days; re-pairing after that is fine.
COOKIE_MAX_AGE_SEC = 400 * 24 * 3600
# A shorter value from the environment is a configuration mistake, not a secret.
_MIN_TOKEN_LEN = 16

# Paths that answer without a token: the pairing call itself. Everything outside
# /api and /ws (the page, its bundle, /healthz) carries nothing to protect.
_OPEN_API_PATHS: frozenset[str] = frozenset({"/api/pair"})


@lru_cache(maxsize=1)
def token() -> str:
    """The access token: from the environment, else the state file, else new."""
    configured = os.environ.get(TOKEN_ENV, "").strip()
    if len(configured) >= _MIN_TOKEN_LEN:
        return configured
    if configured:
        log.warning("%s is shorter than %d characters; ignoring it", TOKEN_ENV, _MIN_TOKEN_LEN)
    stored = (read_json(TOKEN_PATH) or {}).get("token")
    if isinstance(stored, str) and len(stored) >= _MIN_TOKEN_LEN:
        return stored
    fresh = secrets.token_urlsafe(32)
    atomic_write_json(TOKEN_PATH, {"token": fresh})
    return fresh


def matches(candidate: str) -> bool:
    """Whether *candidate* is the token (constant-time)."""
    return bool(candidate) and secrets.compare_digest(candidate.encode(), token().encode())


def needs_token(path: str) -> bool:
    """Whether a request to *path* must carry the token."""
    return path.startswith(("/api/", "/ws/")) and path not in _OPEN_API_PATHS


def presented(headers: dict[str, str]) -> str:
    """The token a request carries — bearer header first, then the cookie — or ``""``."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    try:
        jar = SimpleCookie(headers.get("cookie", ""))
    except CookieError:
        return ""
    morsel = jar.get(COOKIE_NAME)
    return morsel.value if morsel is not None else ""


def pairing_url(base: str) -> str:
    """The link that pairs a browser: *base* (``http://localhost:8100``) with the token."""
    return f"{base.rstrip('/')}/?token={quote(token())}"


def public_base_url(port: int) -> str:
    """Where people open the workspace (``MEDMCP_PUBLIC_URL``, else localhost:*port*)."""
    return os.environ.get("MEDMCP_PUBLIC_URL", "").strip() or f"http://localhost:{port}"
