"""Preview grants: a workflow run starts only from the preview that showed it.

Replay calls stack tools directly, with no permission prompt between the inputs
and the calls, so the run editor's preview — every resolved step, every row
checked — is the confirmation. A grant makes that a property of the server
rather than of the page: ``replay-preview`` issues one for exactly what it
showed, and ``/ws/replay`` starts a run only by redeeming it.

A grant is an unguessable token kept here, in memory, and bound to

- the workflow's name,
- the SHA-256 of its ``recipe.yaml`` as the preview read it, and
- the exact list of input bindings the preview found runnable.

It lives :data:`DEFAULT_TTL_S` seconds and is spent by its first presentation,
whether that start goes ahead or is refused — a refused attempt cannot be
retried with the same token, and nothing can be learned by probing one. A
restart forgets every grant, which costs at most a fresh preview.

The ``medmcp replay`` command line takes no grant: the operator types the
command in a shell, which is its own confirmation, and its run record says
``previewed: false``.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

DEFAULT_TTL_S = 15 * 60
"""How long a preview may be run from. The editor previews again before then."""

MAX_GRANTS = 256
"""Outstanding grants kept at most; the oldest is dropped first."""


class GrantRefusedError(Exception):
    """A run start whose preview grant is missing, spent, expired or for something else."""


@dataclass(frozen=True)
class Grant:
    """What one preview showed, and until when it may be run."""

    name: str
    recipe_sha256: str
    bindings: tuple[tuple[tuple[str, str], ...], ...]
    issued_at: str
    """Wall-clock time of the preview (ISO 8601), for the run record."""
    expires: float
    """Deadline on the store's clock."""


def _frozen(bindings: list[dict[str, str]]) -> tuple[tuple[tuple[str, str], ...], ...]:
    """An immutable copy of *bindings*; key order within a row does not matter."""
    return tuple(tuple(sorted(row.items())) for row in bindings)


class PreviewGrants:
    """The outstanding grants of this process (thread-safe)."""

    def __init__(
        self,
        *,
        ttl_s: float = DEFAULT_TTL_S,
        max_grants: int = MAX_GRANTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create an empty store.

        Args:
            ttl_s: Seconds a grant stays redeemable.
            max_grants: Outstanding grants kept at most.
            clock: Monotonic time source (injectable for tests).
        """
        self.ttl_s = ttl_s
        self._max = max_grants
        self._clock = clock
        self._grants: dict[str, Grant] = {}
        self._lock = threading.Lock()

    def issue(self, name: str, recipe_sha256: str, bindings: list[dict[str, str]]) -> str:
        """Record what a preview showed and return the token that may run it."""
        token = secrets.token_urlsafe(32)
        now = self._clock()
        grant = Grant(
            name=name,
            recipe_sha256=recipe_sha256,
            bindings=_frozen(bindings),
            issued_at=datetime.now(UTC).isoformat(timespec="seconds"),
            expires=now + self.ttl_s,
        )
        with self._lock:
            for key in [k for k, g in self._grants.items() if g.expires <= now]:
                del self._grants[key]
            self._grants[token] = grant
            while len(self._grants) > self._max:
                del self._grants[next(iter(self._grants))]
        return token

    def redeem(
        self, token: str, name: str, recipe_sha256: str, bindings: list[dict[str, str]]
    ) -> Grant:
        """Spend *token* for a start of *name* over *bindings*; return its grant.

        The token is spent whatever the outcome.

        Raises:
            GrantRefusedError: no token, an unknown or spent one, an expired one,
                or one issued for another workflow, another version of its
                recipe, or other inputs. The message says which.
        """
        if not token:
            raise GrantRefusedError("this run was not previewed; check its steps in the run editor")
        with self._lock:
            grant = self._grants.pop(token, None)
        if grant is None:
            raise GrantRefusedError("the preview is unknown or was already used; check again")
        if grant.expires <= self._clock():
            raise GrantRefusedError("the preview has expired; check the steps again")
        if grant.name != name:
            raise GrantRefusedError(f"the preview was for workflow {grant.name!r}, not {name!r}")
        if grant.recipe_sha256 != recipe_sha256:
            raise GrantRefusedError(
                f"workflow {name!r} changed after the preview; check the steps again"
            )
        if grant.bindings != _frozen(bindings):
            raise GrantRefusedError("the inputs differ from the ones the preview showed")
        return grant
