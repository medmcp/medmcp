"""Stack calls the person approved, waiting to be made.

The broker runs stack tools for whoever connects to its socket, and the agent's
shell can connect to it as easily as the per-call proxy vibe spawns — the two run
as the same user, in the same container. So the broker does not take a
connection's word for anything: a ``call_tool`` runs only if it matches, stack,
tool and arguments, a call the person approved in the chat moments before, and
each approval is spent by the one call it was for.

In-memory and per process: the server records the decision and the broker, in
the same process, consumes it.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

JsonDict = dict[str, Any]

# Long enough for a tool call to follow its approval through vibe's queue; short
# enough that an approval nobody used does not linger as an open door.
APPROVAL_TTL_SEC = 600.0


def _key(stack: str, tool: str, args: JsonDict) -> str:
    return json.dumps([stack, tool, args], sort_keys=True, separators=(",", ":"), default=str)


class ApprovalLedger:
    """A multiset of approved ``(stack, tool, args)`` calls, each good for one use."""

    def __init__(
        self, ttl_sec: float = APPROVAL_TTL_SEC, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Approvals expire *ttl_sec* after they are granted."""
        self._ttl = ttl_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._grants: dict[str, list[float]] = {}

    def grant(self, stack: str, tool: str, args: JsonDict) -> None:
        """Record that the person approved this exact call."""
        with self._lock:
            self._grants.setdefault(_key(stack, tool, args), []).append(self._clock())

    def consume(self, stack: str, tool: str, args: JsonDict) -> bool:
        """Spend one approval for this exact call; ``False`` when there is none."""
        now = self._clock()
        key = _key(stack, tool, args)
        with self._lock:
            live = [t for t in self._grants.get(key, []) if now - t < self._ttl]
            if not live:
                self._grants.pop(key, None)
                return False
            live.pop(0)
            if live:
                self._grants[key] = live
            else:
                self._grants.pop(key, None)
            return True
