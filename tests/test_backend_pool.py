"""Lifecycle tests for the persistent MCP backend pool.

These spawn a real (tiny) FastMCP stdio server (``fake_stack_server.py``) so the
subtle part — a session held open across calls in a dedicated runner task — is
exercised for real, not mocked.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from medmcp import replay
from medmcp.backend_pool import Backend, BackendError, BackendPool, BackendSpec

_FAKE_SERVER: Path = Path(__file__).parent / "fake_stack_server.py"


def _spec(name: str, *, gpu: bool = False, idle_ttl: float = 300.0) -> BackendSpec:
    """Build a spec that launches the fake stack server."""
    return BackendSpec(
        name=name,
        command=sys.executable,
        args=[str(_FAKE_SERVER)],
        env={},
        gpu=gpu,
        idle_ttl_sec=idle_ttl,
        startup_timeout_sec=30.0,
        tool_timeout_sec=30.0,
    )


def _resolver(specs: dict[str, BackendSpec]) -> Callable[[str], BackendSpec | None]:
    """Return a resolve_spec callback backed by *specs*."""

    def resolve(name: str) -> BackendSpec | None:
        return specs.get(name)

    return resolve


@pytest.mark.asyncio
async def test_backend_start_call_and_close() -> None:
    """A backend starts, serves calls, lists tools, pings, then closes cleanly."""
    backend = Backend(_spec("fake"))
    await backend.start()
    try:
        assert backend.alive
        names = {t.name for t in await backend.list_tools()}
        assert {"echo", "warmup", "crash"} <= names
        result = await backend.call("echo", {"text": "hi"})
        assert replay.extract_structured(result) == {"text": "hi"}
        assert await backend.healthy()
    finally:
        await backend.aclose()
    assert not backend.alive


@pytest.mark.asyncio
async def test_concurrent_calls_overlap_on_one_backend() -> None:
    """Two slow calls to one warm backend run in parallel, not back-to-back."""
    pool = BackendPool(resolve_spec=_resolver({"fake": _spec("fake")}))
    try:
        backend = await pool.ensure("fake")
        assert not backend.busy
        started = time.monotonic()
        results = await asyncio.gather(
            pool.call("fake", "sleep", {"seconds": 0.5}),
            pool.call("fake", "sleep", {"seconds": 0.5}),
        )
        elapsed = time.monotonic() - started
        # Serialized these would take ~1.0s; multiplexed they finish in ~0.5s.
        assert elapsed < 0.9, f"concurrent calls serialized ({elapsed:.2f}s)"
        assert all(replay.extract_structured(r) == {"slept": 0.5} for r in results)
        assert not backend.busy  # in-flight count returns to zero
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_pool_ensure_reuses_then_evicts() -> None:
    """Ensure returns the same warm backend; evict tears it down."""
    pool = BackendPool(resolve_spec=_resolver({"fake": _spec("fake")}))
    try:
        first = await pool.ensure("fake")
        second = await pool.ensure("fake")
        assert first is second
        assert pool.warm_names() == ["fake"]
        await pool.evict("fake")
        assert pool.warm_names() == []
        assert not first.alive
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_prewarm_invokes_warmup_and_state_persists() -> None:
    """Prewarm calls the warmup hook once; the same process serves later calls."""
    pool = BackendPool(resolve_spec=_resolver({"fake": _spec("fake")}))
    try:
        errors = await pool.prewarm(["fake"])
        assert errors == {"fake": None}
        # warmup ran exactly once at pre-warm, and the counter survives — proof
        # the call below hit the same warm process, not a fresh one.
        result = await pool.call("fake", "warmup_count", {})
        assert replay.extract_structured(result) == {"count": 1}
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_gpu_lru_cap_evicts_least_recently_used() -> None:
    """With max_warm_gpu=1, warming a second GPU stack evicts the first."""
    specs = {"a": _spec("a", gpu=True), "b": _spec("b", gpu=True)}
    pool = BackendPool(resolve_spec=_resolver(specs), max_warm_gpu=1)
    try:
        first = await pool.ensure("a")
        second = await pool.ensure("b")
        assert second.alive
        assert not first.alive
        assert pool.warm_names() == ["b"]
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_idle_backend_is_reaped() -> None:
    """A backend idle past its TTL is evicted by the reaper."""
    pool = BackendPool(
        resolve_spec=_resolver({"fake": _spec("fake", idle_ttl=0.05)}),
        reaper_interval_sec=0.05,
    )
    try:
        await pool.ensure("fake")
        deadline = time.monotonic() + 5.0
        while pool.warm_names() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert pool.warm_names() == []
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_call_failure_evicts_and_ensure_respawns() -> None:
    """A backend that dies mid-call is dropped; ensure spawns a fresh one."""
    pool = BackendPool(resolve_spec=_resolver({"fake": _spec("fake")}))
    try:
        first = await pool.ensure("fake")
        with pytest.raises(Exception):  # noqa: B017 - transport death surfaces
            await pool.call("fake", "crash", {})
        assert pool.warm_names() == []
        second = await pool.ensure("fake")
        assert second is not first
        assert second.alive
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_ensure_unknown_stack_raises() -> None:
    """Ensure raises BackendError when the stack is not installed."""
    pool = BackendPool(resolve_spec=_resolver({}))
    try:
        with pytest.raises(BackendError):
            await pool.ensure("nope")
    finally:
        await pool.aclose()
