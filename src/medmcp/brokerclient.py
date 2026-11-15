"""Ask the core's broker one question from an agent-side helper (stdlib only).

The path guard runs as a ``pre_tool`` hook in the agent's process tree (in the
agent's container when it is isolated) and the LLM shim runs beside the core;
neither holds the core's state. Both reach it over the broker's unix socket
with an op the server answers itself (see the ``ops`` of
:class:`medmcp.backend_broker.BackendBroker`). Every failure -- no socket, no
answer in time, a refusal -- is ``None``: a helper that cannot ask the core goes
on as if the core had nothing to say.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

# Mirrors medmcp.backend_broker.STREAM_LIMIT, without importing the pool.
_STREAM_LIMIT = 32 * 1024 * 1024


def default_socket() -> Path:
    """The broker socket as an agent process finds it.

    ``MEDMCP_BROKER_SOCK`` when set (the proxy's own variable), else
    ``$VIBE_HOME/backend.sock``: the agent's home when it is isolated, the
    core's vibe home otherwise -- in both cases where the broker binds.
    """
    explicit = os.environ.get("MEDMCP_BROKER_SOCK", "").strip()
    if explicit:
        return Path(explicit)
    home = os.environ.get("VIBE_HOME", "").strip()
    base = Path(home) if home else Path(__file__).resolve().parents[2] / ".vibe"
    return base / "backend.sock"


async def ask(
    op: str, payload: JsonDict, *, timeout: float = 3.0, socket: Path | None = None
) -> JsonDict | None:
    """Send one ``op`` with *payload* to the broker; its answer, or ``None``."""
    path = socket or default_socket()
    if not path.exists():
        return None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path), limit=_STREAM_LIMIT), timeout
        )
    except (OSError, TimeoutError):
        return None
    try:
        request = {**payload, "op": op, "id": 1}
        writer.write(json.dumps(request, default=str).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
    except (OSError, TimeoutError):
        return None
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    try:
        response: object = json.loads(line)
    except ValueError:
        return None
    if not isinstance(response, dict) or not cast("JsonDict", response).get("ok"):
        return None
    return cast("JsonDict", response)


def ask_sync(
    op: str, payload: JsonDict, *, timeout: float = 3.0, socket: Path | None = None
) -> JsonDict | None:
    """:func:`ask` for a synchronous caller (a hook process); never raises."""
    try:
        return asyncio.run(ask(op, payload, timeout=timeout, socket=socket))
    except Exception:
        return None
