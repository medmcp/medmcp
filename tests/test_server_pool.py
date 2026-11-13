"""Tests for the stack-pool wiring in the workspace server (no vibe-acp)."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from medmcp import jobs, server, settings
from medmcp.backend_pool import BackendPool

from ._pool import fake_spec, resolver

JsonDict = dict[str, Any]


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


async def test_lifespan_disabled_creates_no_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag off, the lifespan leaves the pool/broker unset."""
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    monkeypatch.setattr(server, "_pool", None)
    monkeypatch.setattr(server, "_broker", None)
    async with server._lifespan(server.app):
        assert server._pool is None
        assert server._broker is None


async def test_lifespan_enabled_starts_and_stops_broker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the flag on, the broker socket is bound on entry and removed on exit."""
    monkeypatch.setenv("MEDMCP_STACK_POOL", "1")
    monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
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


async def test_apply_pool_changes_prewarms_and_evicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activation pre-warms newly-active stacks and evicts deactivated ones."""
    pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}))
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


async def test_reinstall_replaces_a_running_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reinstall drops the backend on the old image before warming the new one."""
    pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}))
    monkeypatch.setattr(server, "_pool", pool)
    try:
        before = set(server._background_tasks)
        await server._apply_pool_changes({"fake"}, set())
        await asyncio.gather(*(set(server._background_tasks) - before))
        old = pool._backends["fake"]

        before = set(server._background_tasks)
        await server._apply_pool_changes({"fake"}, {"fake"})
        await asyncio.gather(*(set(server._background_tasks) - before))
        assert "fake" in pool.warm_names()
        assert pool._backends["fake"] is not old
        assert not old.alive
    finally:
        await pool.aclose()


# ── background jobs ──────────────────────────────────────────────────────────


def _manager_stub(listed: list[jobs.Job]) -> jobs.JobManager:
    """A job manager whose daemon is a list of jobs."""

    class _Stub(jobs.JobManager):
        def list(self) -> list[jobs.Job]:
            return listed

        def logs(self, job_id: str, tail: int = 200) -> str:
            return "epoch 3\n"

        def stop(self, job_id: str) -> jobs.Job:
            job = self.get(job_id)
            listed[:] = [dataclasses.replace(job, state="exited", exit_code=143)]
            return listed[0]

        def remove(self, job_id: str) -> None:
            listed.clear()

    return _Stub(workspace="/ws", resolve_server=lambda _n: None, max_jobs=lambda: 1)


def _a_job(state: str = "running") -> jobs.Job:
    return jobs.Job(
        id="0123456789ab",
        stack="medmcp-nnunet",
        tool="start_training",
        session="chat-1",
        name="d4 fold 0",
        gpu=True,
        image="nnunet:dev",
        image_id="sha256:aa",
        state=state,
        exit_code=None,
        created_at="2026-11-13T10:00:00Z",
        started_at="2026-11-13T10:00:01Z",
        finished_at=None,
        arguments={"dataset": "4"},
    )


def test_only_the_pooled_backend_of_a_stack_with_a_job_runner_gets_the_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offer is made where its answer will be acted on, and only there."""
    declaration: JsonDict = {"command": ["medmcp-nnunet-job", "call"], "tools": ["start_training"]}
    nnunet: JsonDict = {
        "name": "medmcp-nnunet",
        "command": "docker",
        "args": ["run", "--rm", "-i", "nnunet:dev"],
        "env": {},
        "job": declaration,
    }
    neuro: JsonDict = {**nnunet, "name": "medmcp-neuro"}
    del neuro["job"]
    monkeypatch.setattr(settings, "active_servers", lambda: [nnunet, neuro])

    monkeypatch.setattr(server, "_jobs", None)
    spec = server._resolve_backend_spec("medmcp-nnunet")
    assert spec is not None and "MEDMCP_JOBS=1" not in spec.args

    monkeypatch.setattr(server, "_jobs", _manager_stub([]))
    spec = server._resolve_backend_spec("medmcp-nnunet")
    assert spec is not None and spec.args[:3] == ["run", "-e", "MEDMCP_JOBS=1"]
    plain = server._resolve_backend_spec("medmcp-neuro")
    assert plain is not None and "MEDMCP_JOBS=1" not in plain.args
    # backends.json (the proxy's direct-spawn fallback) never carries it.
    registry = settings.build_backend_registry([nnunet])
    assert "MEDMCP_JOBS=1" not in registry["medmcp-nnunet"]["args"]


def test_jobs_api_lists_stops_and_forgets(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """``/api/jobs``: list, one job with its output, stop, then remove."""
    monkeypatch.setattr(server, "_jobs", None)
    assert client.get("/api/jobs").json() == {"enabled": False, "jobs": [], "max": 0}
    assert client.post("/api/jobs/0123456789ab/stop").status_code == 404

    monkeypatch.setattr(server, "_jobs", _manager_stub([_a_job()]))
    body = client.get("/api/jobs").json()
    assert body["enabled"] is True and body["max"] == 1
    assert body["jobs"][0]["status"] == "running" and body["jobs"][0]["session"] == "chat-1"
    assert client.get("/api/jobs/0123456789ab").json()["logs"] == "epoch 3\n"
    assert client.get("/api/jobs/ffffffffffff").status_code == 404
    assert client.delete("/api/jobs/0123456789ab").status_code == 409  # still running
    assert client.post("/api/jobs/0123456789ab/stop").json()["status"] == "stopped"
    assert client.delete("/api/jobs/0123456789ab").json() == {"ok": True}
    assert client.get("/api/jobs").json()["jobs"] == []
