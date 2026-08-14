"""Tests for the pre-warm proxy interception in settings (Layer 1 wiring)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

from medmcp import settings
from medmcp.settings import sync_servers_to_vibe_config

JsonDict = dict[str, Any]

_NEURO: JsonDict = {
    "name": "medmcp-neuro",
    "command": "docker",
    "args": ["run", "--rm", "-i", "--device", "nvidia.com/gpu=all", "-v", "/ws:/ws", "neuro:dev"],
    "env": {},
    "skills_path": "/skills/neuro",
    "tool_timeout_sec": 7200.0,
}
_DICOM: JsonDict = {
    "name": "medmcp-dicom",
    "command": "/abs/bin/medmcp-dicom",
    "args": [],
    "env": {"FOO": "bar"},
}


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, pool: bool) -> None:
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
    if pool:
        monkeypatch.setenv("MEDMCP_STACK_POOL", "1")
    else:
        monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)


def _entries_by_name(tmp_path: Path) -> dict[str, JsonDict]:
    with (tmp_path / "config.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    entries = cast("list[JsonDict]", cfg["mcp_servers"])
    return {str(e["name"]): e for e in entries}


# ── feature flag ──────────────────────────────────────────────────────────────


def test_stack_pool_enabled_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag is off by default and reads common truthy spellings."""
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    assert settings.stack_pool_enabled() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MEDMCP_STACK_POOL", truthy)
        assert settings.stack_pool_enabled() is True
    monkeypatch.setenv("MEDMCP_STACK_POOL", "0")
    assert settings.stack_pool_enabled() is False


# ── backend registry ────────────────────────────────────────────────────────


def test_build_backend_registry_captures_specs_and_infers_gpu() -> None:
    """Registry captures real command/env and infers gpu from the CDI arg."""
    reg = settings.build_backend_registry([_NEURO, _DICOM])
    neuro = reg["medmcp-neuro"]
    assert neuro["command"] == "docker"
    assert neuro["gpu"] is True  # inferred from the CDI --device arg
    assert neuro["tool_timeout_sec"] == 7200.0
    assert neuro["idle_ttl_sec"] == settings.DEFAULT_IDLE_TTL_SEC
    dicom = reg["medmcp-dicom"]
    assert dicom["gpu"] is False
    assert dicom["env"] == {"FOO": "bar"}
    assert dicom["tool_timeout_sec"] == settings.DEFAULT_TOOL_TIMEOUT_SEC


def test_build_backend_registry_honors_explicit_gpu_and_ttl() -> None:
    """An explicit gpu flag and idle_ttl_sec override the defaults."""
    srv: JsonDict = {
        "name": "x",
        "command": "/bin/x",
        "args": [],
        "gpu": True,
        "idle_ttl_sec": 30.0,
    }
    reg = settings.build_backend_registry([srv])
    assert reg["x"]["gpu"] is True
    assert reg["x"]["idle_ttl_sec"] == 30.0


# ── sync interception ────────────────────────────────────────────────────────


def test_sync_disabled_writes_real_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the pool off, sync writes the real commands and no registry."""
    _isolate(monkeypatch, tmp_path, pool=False)
    sync_servers_to_vibe_config([_NEURO, _DICOM])
    by = _entries_by_name(tmp_path)
    assert by["medmcp-neuro"]["command"] == "docker"
    assert by["medmcp-dicom"]["command"] == "/abs/bin/medmcp-dicom"
    assert not (tmp_path / "backends.json").exists()


def test_sync_enabled_routes_through_proxy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With the pool on, entries point at the proxy and backends.json holds the real specs."""
    _isolate(monkeypatch, tmp_path, pool=True)
    sync_servers_to_vibe_config([_NEURO, _DICOM])

    neuro = _entries_by_name(tmp_path)["medmcp-neuro"]
    assert Path(str(neuro["command"])).name == "medmcp-mcp-proxy"
    assert neuro["args"] == ["medmcp-neuro"]
    assert neuro["env"]["MEDMCP_BROKER_SOCK"] == str(tmp_path / "backend.sock")
    # discovery-owned fields are preserved so vibe still loads skills / waits long enough
    assert neuro["skills_path"] == "/skills/neuro"
    assert neuro["tool_timeout_sec"] == 7200.0

    reg = cast("JsonDict", json.loads((tmp_path / "backends.json").read_text()))
    assert reg["medmcp-neuro"]["command"] == "docker"  # the REAL spec
    assert reg["medmcp-neuro"]["gpu"] is True
    assert reg["medmcp-dicom"]["command"] == "/abs/bin/medmcp-dicom"


def test_toggle_off_restores_real_command_and_strips_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Toggling the pool off restores real commands and removes proxy env keys."""
    _isolate(monkeypatch, tmp_path, pool=True)
    sync_servers_to_vibe_config([_NEURO])
    # The first sync proxied the config; now disable and re-sync.
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    sync_servers_to_vibe_config([_NEURO])

    neuro = _entries_by_name(tmp_path)["medmcp-neuro"]
    assert neuro["command"] == "docker"  # real command restored
    assert neuro["args"][0] == "run"
    assert "MEDMCP_BROKER_SOCK" not in cast("JsonDict", neuro.get("env", {}))


# ── startup timeout floor ────────────────────────────────────────────────────


def test_sync_writes_startup_timeout_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every entry gets a startup_timeout_sec; containers get the larger budget.

    vibe's own default is 10s, which a container stack cold-starts past — and a
    discovery miss drops the whole stack for the session.
    """
    _isolate(monkeypatch, tmp_path, pool=False)
    sync_servers_to_vibe_config([_NEURO, _DICOM])
    by = _entries_by_name(tmp_path)
    assert by["medmcp-neuro"]["startup_timeout_sec"] == settings.DEFAULT_STACK_STARTUP_TIMEOUT_SEC
    assert by["medmcp-dicom"]["startup_timeout_sec"] == settings.DEFAULT_STARTUP_TIMEOUT_SEC


def test_sync_preserves_explicit_startup_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A value supplied by the manifest/package is never overwritten by the floor."""
    _isolate(monkeypatch, tmp_path, pool=False)
    sync_servers_to_vibe_config([{**_NEURO, "startup_timeout_sec": 15.0}])
    assert _entries_by_name(tmp_path)["medmcp-neuro"]["startup_timeout_sec"] == 15.0


def test_sync_repairs_existing_entry_without_startup_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stack already in config.toml with no startup_timeout_sec is given one.

    The preserve-branch keeps vibe-owned fields as they are, so without the floor
    an already-installed stack would stay pinned to vibe's 10s default forever —
    no reinstall would fix it. This is the regression that silently dropped a
    container stack on first run.
    """
    _isolate(monkeypatch, tmp_path, pool=False)
    # Simulate a config written before the floor existed.
    sync_servers_to_vibe_config([_NEURO])
    config = tmp_path / "config.toml"
    config.write_text(
        config.read_text().replace(
            f"startup_timeout_sec = {settings.DEFAULT_STACK_STARTUP_TIMEOUT_SEC}\n", ""
        )
    )
    assert "startup_timeout_sec" not in config.read_text()

    sync_servers_to_vibe_config([_NEURO])
    neuro = _entries_by_name(tmp_path)["medmcp-neuro"]
    assert neuro["startup_timeout_sec"] == settings.DEFAULT_STACK_STARTUP_TIMEOUT_SEC
    assert neuro["tool_timeout_sec"] == 7200.0  # other preserved fields survive
