"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from medmcp import provenance, server, settings

JsonDict = dict[str, Any]

# The server's OriginGuard admits only a loopback Host + Origin pair: the client's
# base_url and these WebSocket headers exist to satisfy it.
WS_HEADERS = {"host": "127.0.0.1:8100", "origin": "http://127.0.0.1:8100"}

# A local-only vibe config.toml, the starting point for the model-switch tests.
BASE_CONFIG = """\
active_model = "local"
system_prompt_id = "medmcp"

[[providers]]
name = "ollama"
api_base = "http://localhost:11434/v1"
api_key_env_var = ""
api_style = "openai"
backend = "generic"

[[models]]
name = "muse-medmcp"
provider = "ollama"
alias = "local"
temperature = 0.3
"""


@pytest.fixture(autouse=True)
def _isolate_stacks_d(  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Stop tests from discovering the repo's real ``stacks.d/*.toml`` manifests.

    ``load_mcp_servers`` reads ``settings.STACKS_D_PATH`` (the committed
    ``stacks.d/`` dir) as a real on-disk source; without this, any test that
    exercises discovery would pick up the shipped container manifests. Points it
    at a non-existent dir by default; the stacks.d tests override it themselves.
    The discovery cache is cleared around every test so no result leaks between.
    """
    absent = tmp_path_factory.mktemp("no_stacks") / "absent"
    clear_cache = settings.load_mcp_servers.cache_clear  # a test may monkeypatch the function
    clear_cache()
    with patch("medmcp.settings.STACKS_D_PATH", absent):
        yield
    clear_cache()


@pytest.fixture(autouse=True)
def _isolate_model_choice(  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Stop tests from reading this checkout's real model choice.

    Every config sync consults the cloud-model and local-model state files, so a
    developer who has switched models in a host-native workspace would otherwise
    change what unrelated tests see written to ``config.toml``.
    """
    state = tmp_path_factory.mktemp("model_state")
    with (
        patch("medmcp.settings.CLOUD_MODEL_PATH", state / "cloud_model.json"),
        patch("medmcp.settings.CLOUD_MODEL_SECRET_PATH", state / "cloud_model_secret.json"),
        patch("medmcp.settings.LOCAL_MODEL_PATH", state / "local_model.json"),
    ):
        yield


@pytest.fixture
def vibe_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """``tmp_path`` as an isolated VIBE_HOME: base prompt in place, every state path inside.

    Discovery sees no uv-tool stacks, no ``stacks.d`` manifests and no pool; the
    ``load_mcp_servers`` cache is cleared before and after.
    """
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "medmcp.md").write_text(
        f"You are MedMCP.\n\n**Operational rules**\n{settings.ONPREM_RULE}\n- Other rule.\n"
    )
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
    monkeypatch.setattr(provenance, "VIBE_HOME", tmp_path)
    for attr, filename in (
        ("EXTERNAL_MCP_PATH", "external_mcp.json"),
        ("EXTERNAL_SECRETS_PATH", "external_secrets.json"),
        ("ACTIVE_STACKS_PATH", "active_stacks.json"),
        ("CLOUD_MODEL_PATH", "cloud_model.json"),
        ("CLOUD_MODEL_SECRET_PATH", "cloud_model_secret.json"),
        ("LOCAL_MODEL_PATH", "local_model.json"),
    ):
        monkeypatch.setattr(settings, attr, tmp_path / filename)
    monkeypatch.setattr(settings, "STACKS_D_PATH", tmp_path / "absent-stacks.d")
    # Source #1 too, or the machine's real uv-tool stacks turn up in discovery.
    monkeypatch.setattr(settings, "get_uv_tool_dir", lambda: None)
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    monkeypatch.delenv(settings.CLOUD_MODEL_KEY_ENV, raising=False)
    clear_cache = settings.load_mcp_servers.cache_clear  # a test may monkeypatch the function
    clear_cache()
    yield tmp_path
    clear_cache()


def synced_config(vibe_home: Path, servers: list[JsonDict] | None = None) -> JsonDict:
    """Sync *servers* (default: the active set, freshly discovered) and load config.toml."""
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers() if servers is None else servers)
    with (vibe_home / "config.toml").open("rb") as fh:
        return tomllib.load(fh)


def entries_by_name(cfg: JsonDict) -> dict[str, JsonDict]:
    """The ``[[mcp_servers]]`` entries of a loaded config, keyed by name."""
    return {str(e["name"]): e for e in cast("list[JsonDict]", cfg["mcp_servers"])}


@pytest.fixture
def client() -> TestClient:
    """A client for the workspace app, presenting a loopback origin (see ``WS_HEADERS``)."""
    return TestClient(server.app, base_url="http://127.0.0.1:8100")


def make_vibe_session(
    root: Path,
    session_id: str,
    *,
    messages: Iterable[JsonDict] = (),
    parent_id: str | None = None,
    timestamp: str = "20260101_000000",
) -> Path:
    """Create a vibe-style transcript dir under *root* (a VIBE_HOME) and return it."""
    d = root / "logs" / "session" / f"session_{timestamp}_{session_id[:8]}"
    d.mkdir(parents=True)
    meta: JsonDict = {"session_id": session_id}
    if parent_id is not None:
        meta["parent_session_id"] = parent_id
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "messages.jsonl").write_text("".join(json.dumps(m) + "\n" for m in messages))
    return d
