"""Tests for the stack-pool wiring in the workspace server (no vibe-acp)."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from medmcp import server, settings
from medmcp.backend_pool import BackendPool, BackendSpec

# pyright: reportPrivateUsage=false

JsonDict = dict[str, Any]

_FAKE_SERVER: Path = Path(__file__).parent / "fake_stack_server.py"


def _spec(name: str) -> BackendSpec:
    return BackendSpec(
        name=name,
        command=sys.executable,
        args=[str(_FAKE_SERVER)],
        env={},
        gpu=False,
        idle_ttl_sec=300.0,
        startup_timeout_sec=30.0,
        tool_timeout_sec=30.0,
    )


def _resolver(specs: dict[str, BackendSpec]) -> Callable[[str], BackendSpec | None]:
    def resolve(name: str) -> BackendSpec | None:
        return specs.get(name)

    return resolve


def test_resolve_backend_spec_maps_active_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool callback turns an active stack into a BackendSpec, None if unknown."""
    neuro: JsonDict = {
        "name": "medmcp-neuro",
        "command": "docker",
        "args": ["run", "--device", "nvidia.com/gpu=all", "neuro:dev"],
        "env": {},
        "tool_timeout_sec": 7200.0,
    }
    monkeypatch.setattr(settings, "active_servers", lambda: [neuro])

    spec = server._resolve_backend_spec("medmcp-neuro")
    assert spec is not None
    assert spec.command == "docker"
    assert spec.gpu is True
    assert spec.tool_timeout_sec == 7200.0
    assert server._resolve_backend_spec("ghost") is None


@pytest.mark.asyncio
async def test_lifespan_disabled_creates_no_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag off, the lifespan leaves the pool/broker unset."""
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    monkeypatch.setattr(server, "_pool", None)
    monkeypatch.setattr(server, "_broker", None)
    async with server._lifespan(server.app):
        assert server._pool is None
        assert server._broker is None


@pytest.mark.asyncio
async def test_lifespan_enabled_starts_and_stops_broker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the flag on, the broker socket is bound on entry and removed on exit."""
    monkeypatch.setenv("MEDMCP_STACK_POOL", "1")
    monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
    monkeypatch.setattr(settings, "WORKFLOWS_ENABLED_PATH", tmp_path / "workflows_enabled.json")
    monkeypatch.setattr(settings, "ACTIVE_WORKFLOWS_PATH", tmp_path / "active_workflows.json")
    monkeypatch.setattr(settings, "get_uv_tool_dir", lambda: None)
    settings.load_mcp_servers.cache_clear()
    monkeypatch.setattr(server, "_pool", None)
    monkeypatch.setattr(server, "_broker", None)

    sock = tmp_path / "backend.sock"
    async with server._lifespan(server.app):
        assert server._pool is not None
        assert server._broker is not None
        assert sock.exists()
        # the startup sync wrote the (empty) registry + proxied config
        assert (tmp_path / "backends.json").exists()
    assert server._pool is None
    assert server._broker is None
    assert not sock.exists()
    settings.load_mcp_servers.cache_clear()


@pytest.mark.asyncio
async def test_apply_pool_changes_prewarms_and_evicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activation pre-warms newly-active stacks and evicts deactivated ones."""
    pool = BackendPool(resolve_spec=_resolver({"fake": _spec("fake")}))
    monkeypatch.setattr(server, "_pool", pool)
    try:
        before = set(server._background_tasks)
        await server._apply_pool_changes({"fake"}, set())
        # pre-warm is fire-and-forget; await the task it scheduled
        await asyncio.gather(*(set(server._background_tasks) - before))
        assert "fake" in pool.warm_names()

        await server._apply_pool_changes(set(), {"fake"})
        assert "fake" not in pool.warm_names()
    finally:
        await pool.aclose()
