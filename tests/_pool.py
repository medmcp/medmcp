"""Shared harness for the backend-pool tests: a spec for the fake stack server."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool, BackendSpec

FAKE_STACK_SERVER: Path = Path(__file__).parent / "fake_stack_server.py"


def fake_spec(name: str, *, gpu: bool = False, idle_ttl: float = 300.0) -> BackendSpec:
    """Build a spec that launches the fake stack server."""
    return BackendSpec(
        name=name,
        command=sys.executable,
        args=[str(FAKE_STACK_SERVER)],
        env={},
        gpu=gpu,
        idle_ttl_sec=idle_ttl,
        startup_timeout_sec=30.0,
        tool_timeout_sec=30.0,
    )


def resolver(specs: dict[str, BackendSpec]) -> Callable[[str], BackendSpec | None]:
    """Return a resolve_spec callback backed by *specs*."""

    def resolve(name: str) -> BackendSpec | None:
        return specs.get(name)

    return resolve


@asynccontextmanager
async def broker(
    tmp_path: Path, specs: dict[str, BackendSpec] | None = None
) -> AsyncGenerator[Path]:
    """Run a broker over a real socket against a pool (the fake stack by default)."""
    if specs is None:
        specs = {"fake": fake_spec("fake")}
    pool = BackendPool(resolve_spec=resolver(specs))
    broker = BackendBroker(pool, tmp_path / "broker.sock")
    await broker.start()
    try:
        yield broker.socket_path
    finally:
        await broker.aclose()
        await pool.aclose()
