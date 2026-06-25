"""Persistent MCP backend pool — keeps stack servers warm across tool calls.

Today every stack tool call spawns a fresh MCP stdio server (``docker run`` +
import + CUDA init), runs one tool, and tears it down. This module holds each
stack's server **persistent** so that cost is paid once (ideally pre-warmed at
activation) instead of on every call. It is the UI/vibe-agnostic core of the
stack pre-warm design (``docs/stack-prewarm-proxy.md``, Layer 1); the broker and
proxy shim that connect vibe-acp to this pool live in separate modules.

A :class:`Backend` wraps one long-lived ``stdio_client`` + ``ClientSession`` (the
same primitives as :mod:`medmcp.replay`'s ``mcp_caller``) but does **not** tear
the session down after a call. Because the MCP/anyio session must be entered and
exited in the *same* task, each backend runs a dedicated *runner* task that owns
the session for its whole lifetime: it opens the session, signals readiness, then
waits for a close signal before unwinding. Calls from other tasks use the live
``ClientSession`` object (safe — the SDK multiplexes requests by id); calls into a
single backend are serialized so a stack keeps its one-call-at-a-time assumption.

:class:`BackendPool` owns the set of warm backends and enforces two policies: an
idle-TTL reaper evicts backends unused past their TTL, and a GPU LRU cap bounds
how many VRAM-holding backends stay warm at once (they compete with the LLM).

Sampling relay is intentionally **not** implemented in v1 (no current stack uses
MCP sampling); a backend that issues a sampling request gets the SDK's default
rejection. See the design doc's open decisions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log: logging.Logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

# Name of the optional, best-effort tool a stack may expose so the pool can make
# it load heavy resources (model weights, CUDA context) at pre-warm time rather
# than on the first real call. Unmodified stacks simply don't expose it.
WARMUP_TOOL: str = "warmup"

# How often the idle reaper sweeps for TTL-expired / dead backends.
_REAPER_INTERVAL_SEC: float = 15.0
# Liveness ping bound — a pre-warmed backend may have died while idle (OOM, the
# container daemon restarting); ``ensure`` probes before handing it back.
_PING_TIMEOUT_SEC: float = 2.0
# Grace period for a backend to unwind cleanly before it is cancelled.
_TEARDOWN_TIMEOUT_SEC: float = 10.0


class BackendError(Exception):
    """Raised when a backend cannot be started or is not running."""


@dataclass(frozen=True)
class BackendSpec:
    """Immutable launch recipe for one stack's persistent MCP server.

    Produced by discovery (``settings.load_mcp_servers`` → ``.vibe/backends.json``)
    and resolved by name through the pool's ``resolve_spec`` callback.

    Attributes:
        name: Stack name (e.g. ``"medmcp-neuro"``).
        command: Executable to launch (``"docker"`` for container stacks, or the
            absolute uv-tool binary host-native).
        args: Arguments to *command*.
        env: Extra environment variables, merged over ``os.environ``.
        gpu: Whether this stack holds VRAM; counts against the GPU LRU cap.
        idle_ttl_sec: Evict the backend after this many seconds idle.
        startup_timeout_sec: Max time to reach the MCP ``initialize`` handshake.
        tool_timeout_sec: Per-call read timeout for ``call_tool``.
    """

    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    gpu: bool
    idle_ttl_sec: float
    startup_timeout_sec: float
    tool_timeout_sec: float


class Backend:
    """One long-lived MCP stdio session to a stack server, reused across calls."""

    def __init__(self, spec: BackendSpec, *, cwd: str | None = None) -> None:
        """Create a backend for *spec*; call :meth:`start` to launch it.

        Args:
            spec: The launch recipe.
            cwd: Working directory for the server process (the workspace root, so
                a stack's tools resolve paths the same way the live chat does).
        """
        self.spec = spec
        self._cwd = cwd
        self._session: ClientSession | None = None
        self._runner: asyncio.Task[None] | None = None
        self._started: asyncio.Event = asyncio.Event()
        self._closing: asyncio.Event = asyncio.Event()
        self._start_error: BaseException | None = None
        self._lock: asyncio.Lock = asyncio.Lock()  # serializes call_tool
        self._tools: list[mcp_types.Tool] | None = None
        self._last_used: float = time.monotonic()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _server_params(self) -> StdioServerParameters:
        env = {**os.environ, **self.spec.env}
        return StdioServerParameters(
            command=self.spec.command,
            args=list(self.spec.args),
            env=env,
            cwd=self._cwd,
        )

    async def _run(self) -> None:
        """Own the session for its whole lifetime (enter + exit in one task).

        Opens the stdio session, runs ``initialize``, publishes the live session,
        then blocks until :meth:`aclose` signals close — so the session's context
        managers unwind in the same task that entered them. A failure before the
        ready signal is stored for :meth:`start` to raise; a failure *after* (the
        process dying) just ends the task, so :attr:`alive` flips to ``False``.
        """
        init_timeout = timedelta(seconds=self.spec.startup_timeout_sec)
        try:
            async with (
                stdio_client(self._server_params()) as (read, write),
                ClientSession(read, write, read_timeout_seconds=init_timeout) as session,
            ):
                await session.initialize()
                self._session = session
                self._started.set()
                await self._closing.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._started.is_set():
                self._start_error = exc
                self._started.set()  # unblock start()
            else:
                log.warning("backend %s ended unexpectedly: %s", self.spec.name, exc)
        finally:
            self._session = None

    async def start(self) -> None:
        """Launch the server and wait until it is initialized.

        Raises:
            BackendError: the server failed to start or did not initialize within
                its startup timeout.
        """
        if self._runner is not None:
            return
        self._runner = asyncio.create_task(self._run(), name=f"backend:{self.spec.name}")
        try:
            await asyncio.wait_for(
                self._started.wait(), timeout=self.spec.startup_timeout_sec + 5.0
            )
        except TimeoutError as exc:
            await self.aclose()
            raise BackendError(
                f"stack {self.spec.name!r} did not start within "
                f"{self.spec.startup_timeout_sec:.0f}s"
            ) from exc
        if self._start_error is not None or self._session is None:
            err = self._start_error
            await self.aclose()
            raise BackendError(f"stack {self.spec.name!r} failed to start: {err}") from err

    async def aclose(self) -> None:
        """Stop the server; idempotent and safe to call mid-call (cancels it)."""
        self._closing.set()
        runner, self._runner = self._runner, None
        if runner is None:
            self._session = None
            return
        try:
            await asyncio.wait_for(runner, timeout=_TEARDOWN_TIMEOUT_SEC)
        except TimeoutError:
            log.warning("backend %s did not stop in time; cancelled", self.spec.name)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("backend %s teardown error: %s", self.spec.name, exc)
        finally:
            self._session = None

    # ── use ──────────────────────────────────────────────────────────────────

    async def call(self, tool: str, args: JsonDict) -> mcp_types.CallToolResult:
        """Call *tool* on this backend (serialized with other calls to it).

        A transport-level failure (the process having died) propagates so the pool
        can evict and respawn; a tool that merely *reports* failure returns a
        normal :class:`~mcp.types.CallToolResult` with ``isError`` set.
        """
        session = self._session
        if session is None or not self.alive:
            raise BackendError(f"backend {self.spec.name!r} is not running")
        timeout = timedelta(seconds=self.spec.tool_timeout_sec)
        async with self._lock:
            self._last_used = time.monotonic()
            try:
                return await session.call_tool(tool, args, read_timeout_seconds=timeout)
            finally:
                self._last_used = time.monotonic()

    async def list_tools(self) -> list[mcp_types.Tool]:
        """Return the backend's tools (cached after the first call)."""
        if self._tools is not None:
            return self._tools
        session = self._session
        if session is None or not self.alive:
            raise BackendError(f"backend {self.spec.name!r} is not running")
        resp = await session.list_tools()
        self._tools = list(resp.tools)
        self._last_used = time.monotonic()
        return self._tools

    async def healthy(self) -> bool:
        """Return whether the backend answers a bounded ping (does not touch idle)."""
        session = self._session
        if session is None or not self.alive:
            return False
        try:
            await asyncio.wait_for(session.send_ping(), timeout=_PING_TIMEOUT_SEC)
        except Exception:
            return False
        return True

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def alive(self) -> bool:
        """Whether the runner is still running and the session is live."""
        return (
            self._runner is not None
            and not self._runner.done()
            and self._session is not None
            and not self._closing.is_set()
        )

    @property
    def busy(self) -> bool:
        """Whether a call is currently in flight (its lock is held)."""
        return self._lock.locked()

    @property
    def last_used(self) -> float:
        """Monotonic timestamp of the last call (drives idle-TTL and LRU)."""
        return self._last_used

    def idle_seconds(self, now: float | None = None) -> float:
        """Seconds since the backend was last used."""
        return (now if now is not None else time.monotonic()) - self._last_used


class BackendPool:
    """Owns warm backends; enforces idle-TTL eviction and a GPU LRU cap."""

    def __init__(
        self,
        *,
        resolve_spec: Callable[[str], BackendSpec | None],
        max_warm_gpu: int = 1,
        cwd: str | None = None,
        reaper_interval_sec: float = _REAPER_INTERVAL_SEC,
    ) -> None:
        """Create an empty pool.

        Args:
            resolve_spec: Maps a stack name to its :class:`BackendSpec`, or ``None``
                if the stack is not installed.
            max_warm_gpu: Max number of GPU (VRAM-holding) backends kept warm at
                once; starting another evicts the least-recently-used GPU backend.
                Clamped to at least 1.
            cwd: Working directory passed to every backend (the workspace root).
            reaper_interval_sec: How often the idle reaper sweeps.
        """
        self._resolve = resolve_spec
        self._max_warm_gpu = max(1, max_warm_gpu)
        self._cwd = cwd
        self._reaper_interval = reaper_interval_sec
        self._backends: dict[str, Backend] = {}
        self._name_locks: dict[str, asyncio.Lock] = {}
        self._reaper: asyncio.Task[None] | None = None
        self._closed = False

    # ── public API ───────────────────────────────────────────────────────────

    async def ensure(self, name: str) -> Backend:
        """Return a live backend for *name*, starting (or restarting) it if needed.

        A pre-existing backend is probed with a bounded ping first, so a stack that
        died while idle is transparently replaced. Concurrent ``ensure`` calls for
        the same name are serialized so the server is started only once.
        """
        existing = self._backends.get(name)
        if existing is not None and existing.alive and await existing.healthy():
            return existing

        async with self._name_lock(name):
            existing = self._backends.get(name)
            if existing is not None and existing.alive:
                return existing
            if existing is not None:
                await self._drop(name)  # dead — make way for a fresh one

            spec = self._resolve(name)
            if spec is None:
                raise BackendError(f"stack {name!r} is not installed")
            if spec.gpu:
                await self._enforce_gpu_cap(exclude=name)

            backend = Backend(spec, cwd=self._cwd)
            await backend.start()
            self._backends[name] = backend
            self._ensure_reaper()
            log.info("backend %s warm (gpu=%s)", name, spec.gpu)
            return backend

    async def call(self, name: str, tool: str, args: JsonDict) -> mcp_types.CallToolResult:
        """Call *tool* on stack *name*, warming the backend if needed.

        On a transport failure the backend is evicted so the next call respawns it;
        the error is **not** retried here (re-running a side-effecting medical tool
        is worse than surfacing the failure).
        """
        backend = await self.ensure(name)
        try:
            return await backend.call(tool, args)
        except BackendError:
            raise
        except Exception:
            await self._drop(name)
            raise

    async def list_tools(self, name: str) -> list[mcp_types.Tool]:
        """Return the tools advertised by stack *name*, warming it if needed."""
        backend = await self.ensure(name)
        return await backend.list_tools()

    async def prewarm(self, names: Iterable[str]) -> dict[str, str | None]:
        """Warm each named stack now (activation hook); never raises.

        Calls each stack's optional ``warmup`` tool so heavy init happens here
        rather than on the first real call. Returns a per-stack result mapping with
        ``None`` on success or an error string on failure.
        """
        results: dict[str, str | None] = {}
        for name in names:
            try:
                backend = await self.ensure(name)
                await self._maybe_warmup(backend)
                results[name] = None
            except Exception as exc:
                log.warning("pre-warm of %s failed: %s", name, exc)
                results[name] = str(exc)
        return results

    async def evict(self, name: str) -> None:
        """Tear down a backend now (deactivation), freeing its RAM/VRAM."""
        await self._drop(name)

    def warm_names(self) -> list[str]:
        """Names of currently-live backends."""
        return [n for n, b in self._backends.items() if b.alive]

    async def aclose(self) -> None:
        """Stop the reaper and tear down every backend."""
        self._closed = True
        reaper, self._reaper = self._reaper, None
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
        for name in list(self._backends):
            await self._drop(name)

    # ── internals ────────────────────────────────────────────────────────────

    def _name_lock(self, name: str) -> asyncio.Lock:
        lock = self._name_locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._name_locks[name] = lock
        return lock

    async def _drop(self, name: str) -> None:
        backend = self._backends.pop(name, None)
        if backend is not None:
            await backend.aclose()
            log.info("backend %s evicted", name)

    async def _enforce_gpu_cap(self, *, exclude: str) -> None:
        """Evict least-recently-used non-busy GPU backends to make room for one more."""

        def warm_gpu() -> list[Backend]:
            return [b for n, b in self._backends.items() if b.spec.gpu and b.alive and n != exclude]

        while len(warm_gpu()) >= self._max_warm_gpu:
            evictable = [b for b in warm_gpu() if not b.busy]
            if not evictable:
                log.warning(
                    "GPU warm cap (%d) reached but all backends busy; "
                    "temporarily exceeding it for %s",
                    self._max_warm_gpu,
                    exclude,
                )
                return
            lru = min(evictable, key=lambda b: b.last_used)
            await self._drop(lru.spec.name)

    async def _maybe_warmup(self, backend: Backend) -> None:
        try:
            tools = await backend.list_tools()
        except Exception as exc:
            log.debug("warmup tool-list for %s failed: %s", backend.spec.name, exc)
            return
        if any(t.name == WARMUP_TOOL for t in tools):
            try:
                await backend.call(WARMUP_TOOL, {})
            except Exception as exc:
                log.debug("warmup call for %s failed: %s", backend.spec.name, exc)

    def _ensure_reaper(self) -> None:
        if self._reaper is None and not self._closed:
            self._reaper = asyncio.create_task(self._reap_loop(), name="backend-pool-reaper")

    async def _reap_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._reaper_interval)
            now = time.monotonic()
            for name, backend in list(self._backends.items()):
                if not backend.alive:
                    await self._drop(name)
                elif not backend.busy and backend.idle_seconds(now) >= backend.spec.idle_ttl_sec:
                    log.info("backend %s idle %.0fs; reaping", name, backend.idle_seconds(now))
                    await self._drop(name)
