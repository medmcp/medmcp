"""Persistent MCP backend pool — keeps stack servers warm across tool calls.

Today every stack tool call spawns a fresh MCP stdio server (``docker run`` +
import + CUDA init), runs one tool, and tears it down. This module holds each
stack's server **persistent** so that cost is paid once (ideally pre-warmed at
activation) instead of on every call. It is the UI/vibe-agnostic core of the
stack pre-warm design (Layer 1); the broker and
proxy shim that connect vibe-acp to this pool live in separate modules.

A :class:`Backend` wraps one long-lived ``stdio_client`` + ``ClientSession`` (the
same primitives as :mod:`medmcp.replay`'s ``mcp_caller``) but does **not** tear
the session down after a call. Because the MCP/anyio session must be entered and
exited in the *same* task, each backend runs a dedicated *runner* task that owns
the session for its whole lifetime: it opens the session, signals readiness, then
waits for a close signal before unwinding. Concurrent calls share that one live
``ClientSession`` and run in parallel — the SDK multiplexes requests by id and the
stack server handles each as its own task — so a long-running tool call (e.g. a
multi-minute skull-strip) no longer blocks shorter calls to the same stack. An
in-flight counter keeps :attr:`Backend.busy` truthful so the pool never evicts or
reaps a backend that still has work running.

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
from pathlib import Path
from typing import Any, cast

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from medmcp import tooleffect
from medmcp.statefile import atomic_write_json, read_json

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
        self._inflight: int = 0  # in-flight call_tool count (drives `busy`)
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
        """Call *tool* on this backend; concurrent calls run in parallel.

        Calls share the one live ``ClientSession`` (the SDK multiplexes requests by
        id and the stack server handles each in its own task), so a long-running
        tool call does not block other calls to the same backend. The in-flight
        count keeps :attr:`busy` truthful so the pool will not evict or reap a
        backend with work still running.

        A transport-level failure (the process having died) propagates so the pool
        can evict and respawn; a tool that merely *reports* failure returns a
        normal :class:`~mcp.types.CallToolResult` with ``isError`` set. Stopping
        the backend while the call runs fails it with :class:`BackendError`.
        """
        session = self._session
        if session is None or not self.alive:
            raise BackendError(f"backend {self.spec.name!r} is not running")
        timeout = timedelta(seconds=self.spec.tool_timeout_sec)
        self._inflight += 1
        self._last_used = time.monotonic()
        # Raced against the close signal: when the backend is stopped (evicted,
        # replaced, the pool shutting down) the session is torn down by
        # cancellation, which never answers a request still waiting for its
        # response — the call would otherwise sit until the tool timeout (hours).
        call = asyncio.ensure_future(session.call_tool(tool, args, read_timeout_seconds=timeout))
        closing = asyncio.ensure_future(self._closing.wait())
        try:
            done, _ = await asyncio.wait({call, closing}, return_when=asyncio.FIRST_COMPLETED)
            if call in done:
                return call.result()
            raise BackendError(f"stack {self.spec.name!r} was stopped while {tool!r} was running")
        finally:
            closing.cancel()
            if not call.done():
                call.cancel()
            self._inflight -= 1
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
        """Whether at least one call is currently in flight."""
        return self._inflight > 0

    @property
    def inflight(self) -> int:
        """How many calls are currently in flight."""
        return self._inflight

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
        schema_cache: Path | None = None,
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
            schema_cache: A JSON file that keeps each stack's input schemas and
                declared effects across restarts; see :meth:`known_tools`.
        """
        self._resolve = resolve_spec
        self._schema_cache = schema_cache
        # Each stack's input schemas as last listed, kept after its backend is
        # reaped (vibe keeps using a tool list for a day without asking again),
        # and the effect each tagged tool declares (see medmcp.tooleffect).
        self._known: dict[str, dict[str, JsonDict]] = {}
        self._effects: dict[str, dict[str, str]] = {}
        self._load_schemas()
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

        A pre-existing idle backend is probed with a bounded ping first, so a stack
        that died while idle is transparently replaced; a busy one is never replaced
        while its process lives. Concurrent ``ensure`` calls for
        the same name are serialized so the server is started only once.
        """
        probed = self._backends.get(name)
        # A busy backend is not pinged: a stack in the middle of heavy work may not
        # answer within the ping bound, and replacing it would kill the call that is
        # running. Calls share its session, so the new one is simply served later.
        if probed is not None and probed.alive and (probed.busy or await probed.healthy()):
            return probed

        async with self._name_lock(name):
            current = self._backends.get(name)
            if current is not None and current.alive and current is not probed:
                return current  # another ensure replaced it while we waited
            if current is not None:
                await self._drop(name)  # dead, or hung (failed its ping) — make way

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
        tools = await backend.list_tools()
        self._remember(name, tools)
        return tools

    def known_tools(self, name: str) -> dict[str, JsonDict] | None:
        """Stack *name*'s tools as last listed (tool → input schema), never warming it.

        ``None`` when the pool has never listed them. Survives the backend's
        eviction and, with a ``schema_cache`` file, a restart; refreshed each time
        the stack's tools are listed.
        """
        known = self._known.get(name)
        return dict(known) if known is not None else None

    def known_stacks(self) -> list[str]:
        """The stacks whose tool lists the pool holds (see :meth:`known_tools`)."""
        return list(self._known)

    def known_effects(self, name: str) -> dict[str, str] | None:
        """Stack *name*'s declared effects as last listed (tool → effect), never warming it.

        Only tagged tools appear (see :mod:`medmcp.tooleffect`). ``None`` when the
        pool has never listed the stack; kept, refreshed and dropped together with
        :meth:`known_tools`.
        """
        if name not in self._known:
            return None
        return dict(self._effects.get(name, {}))

    def tool_effect(self, name: str, tool: str) -> str | None:
        """The effect stack *name*'s *tool* declared when last listed, or ``None``."""
        return self._effects.get(name, {}).get(tool)

    def _remember(self, name: str, tools: list[mcp_types.Tool]) -> None:
        listed = {t.name: dict(t.inputSchema) for t in tools if t.name != WARMUP_TOOL}
        effects: dict[str, str] = {}
        for t in tools:
            effect = tooleffect.tool_effect(t)
            if t.name != WARMUP_TOOL and effect is not None:
                effects[t.name] = effect
        if self._known.get(name) == listed and self._effects.get(name) == effects:
            return
        self._known[name] = listed
        self._effects[name] = effects
        self._save_schemas()

    def _save_schemas(self) -> None:
        if self._schema_cache is None:
            return
        stacks: JsonDict = {
            stack: {
                tool: {"schema": schema, "effect": self._effects.get(stack, {}).get(tool)}
                for tool, schema in tools.items()
            }
            for stack, tools in self._known.items()
        }
        try:
            atomic_write_json(self._schema_cache, {"version": 2, "stacks": stacks})
        except OSError as exc:
            log.debug("could not write the tool schema cache: %s", exc)

    def _load_schemas(self) -> None:
        """Read the cache: ``{"version": 2, "stacks": {stack: {tool: {schema, effect}}}}``.

        A file from before effects were kept (``{stack: {tool: schema}}``) still
        gives the schemas; its effects are unknown until the stack is listed again.
        """
        data = (read_json(self._schema_cache) if self._schema_cache is not None else None) or {}
        versioned = data.get("version") == 2
        stacks = data.get("stacks") if versioned else data
        if not isinstance(stacks, dict):
            return
        for stack, raw_tools in cast("JsonDict", stacks).items():
            if not isinstance(raw_tools, dict):
                continue
            schemas: dict[str, JsonDict] = {}
            effects: dict[str, str] = {}
            for tool, raw in cast("JsonDict", raw_tools).items():
                if not isinstance(raw, dict):
                    continue
                entry = cast("JsonDict", raw)
                schema = entry.get("schema") if versioned else entry
                if not isinstance(schema, dict):
                    continue
                schemas[str(tool)] = cast("JsonDict", schema)
                effect = entry.get("effect") if versioned else None
                if isinstance(effect, str) and effect in tooleffect.EFFECTS:
                    effects[str(tool)] = effect
            self._known[str(stack)] = schemas
            self._effects[str(stack)] = effects

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
        """Tear down a backend now (deactivation), freeing its RAM/VRAM.

        Its remembered tool list goes too: a stack that comes back, possibly as a
        new image, is listed again before its schemas and effects are trusted.
        """
        await self._drop(name)
        self._effects.pop(name, None)
        if self._known.pop(name, None) is not None:
            self._save_schemas()

    async def stop(self, name: str) -> None:
        """Tear down *name*'s backend now, killing any call still running on it.

        The evict path without forgetting the stack's tool list: a cancelled task
        (:mod:`medmcp.tasks`) stops its work this way, and the stack itself is
        unchanged, so its schemas and effects stay valid.
        """
        await self._drop(name)

    def inflight(self, name: str) -> int:
        """How many calls are running on *name*'s backend right now."""
        backend = self._backends.get(name)
        return backend.inflight if backend is not None else 0

    def is_gpu(self, name: str) -> bool:
        """Whether stack *name* holds VRAM (from its spec; ``False`` when not installed)."""
        backend = self._backends.get(name)
        if backend is not None:
            return backend.spec.gpu
        spec = self._resolve(name)
        return spec is not None and spec.gpu

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
        self._remember(backend.spec.name, tools)
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
