"""Tests for the local model switch — catalog state, the Ollama side, the sync output.

Driven against a fake Ollama behind an ``httpx.MockTransport``, so the requests
made (and the ones deliberately not made) are asserted directly: nothing is
downloaded before a licence is accepted, the choice is recorded only once the
model is usable, and only one model is left in memory.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

# pyright: reportPrivateUsage=false
from medmcp import localmodels, server, settings

JsonDict = dict[str, Any]

_WS_HEADERS = {"host": "127.0.0.1:8100", "origin": "http://127.0.0.1:8100"}

_BASE_CONFIG = """\
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
thinking = "medium"
"""


def _info(
    context: int = 262_144,
    capabilities: tuple[str, ...] = ("completion", "tools"),
    params: str = "",
) -> JsonDict:
    return {
        "capabilities": list(capabilities),
        "model_info": {"general.architecture": "x", "x.context_length": context},
        "parameters": params,
    }


class FakeOllama:
    """The slice of Ollama's HTTP API a model switch uses."""

    def __init__(self) -> None:
        """Start with the default model present and loaded, as a fresh install has it."""
        self.library: dict[str, JsonDict] = {m.tag: _info() for m in localmodels.CATALOG}
        self.installed: dict[str, JsonDict] = {
            "muse-glimmer:30b": _info(),
            "muse-medmcp:latest": _info(),
        }
        self.loaded: list[str] = ["muse-medmcp:latest"]
        self.created: dict[str, JsonDict] = {}
        self.calls: list[str] = []
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Answer one request the way Ollama would, and remember that it was made."""
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        body = cast("JsonDict", json.loads(request.content)) if request.content else {}
        model = str(body.get("model", ""))
        self.calls.append(f"{request.method} {path} {model}".strip())
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.installed]})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.loaded]})
        if path == "/api/pull":
            if model not in self.library:
                return httpx.Response(
                    200, text='{"error":"pull model manifest: file does not exist"}\n'
                )
            self.installed[model] = self.library[model]
            lines = [
                {"status": "pulling manifest"},
                {"status": "pulling aaa", "digest": "sha256:aaa", "total": 1000, "completed": 400},
                {"status": "pulling aaa", "digest": "sha256:aaa", "total": 1000, "completed": 1000},
                {"status": "pulling bbb", "digest": "sha256:bbb", "total": 50, "completed": 50},
                {"status": "success"},
            ]
            return httpx.Response(200, text="".join(json.dumps(line) + "\n" for line in lines))
        if path == "/api/show":
            if model not in self.installed:
                return httpx.Response(404, json={"error": f"model '{model}' not found"})
            return httpx.Response(200, json=self.installed[model])
        if path == "/api/create":
            self.created[model] = body
            self.installed[model] = self.installed[str(body["from"])]
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/generate":
            if body.get("keep_alive") == 0:
                self.loaded = [n for n in self.loaded if n != model]
            elif model not in self.loaded:
                self.loaded.append(model)
            return httpx.Response(200, json={"done": True})
        if path == "/api/delete":
            if model not in self.installed:
                return httpx.Response(404, json={"error": "model not found"})
            del self.installed[model]
            return httpx.Response(200)
        return httpx.Response(404)


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeOllama:
    """A fake Ollama wired into ``localmodels``, with all on-disk state in a tmp dir."""
    fake = FakeOllama()

    def _client(read_timeout: float | None = 30.0) -> httpx.AsyncClient:
        del read_timeout
        return httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), base_url="http://o")

    monkeypatch.setattr(localmodels, "_client", _client)
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "medmcp.md").write_text(f"You are MedMCP.\n{settings.ONPREM_RULE}\n")
    (tmp_path / "config.toml").write_text(_BASE_CONFIG)
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "muse-medmcp")
    monkeypatch.setattr(settings, "EXTERNAL_MCP_PATH", tmp_path / "external_mcp.json")
    monkeypatch.setattr(settings, "ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json")
    monkeypatch.setattr(settings, "get_uv_tool_dir", lambda: None)
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    settings.load_mcp_servers.cache_clear()
    return fake


def _sync(tmp_path: Path) -> JsonDict:
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    with (tmp_path / "config.toml").open("rb") as fh:
        return tomllib.load(fh)


def _row(listing: JsonDict, model_id: str) -> JsonDict:
    return next(r for r in cast("list[JsonDict]", listing["models"]) if r["id"] == model_id)


# ── catalog state ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_fresh_install_runs_the_default(ollama: FakeOllama) -> None:
    """Nothing selected: Muse Glimmer is in use, the rest are offered for download."""
    del ollama
    listing = await localmodels.list_models()
    assert listing["active"] == localmodels.DEFAULT_MODEL_ID
    assert listing["reachable"] is True
    muse = _row(listing, "muse-glimmer")
    assert (muse["default"], muse["active"], muse["downloaded"], muse["deletable"]) == (
        True,
        True,
        True,
        False,
    )
    others = [r for r in listing["models"] if not r["default"]]
    assert len(others) == 6
    assert not any(r["downloaded"] or r["active"] or r["deletable"] for r in others)
    assert settings.local_model_name() == "muse-medmcp"


@pytest.mark.asyncio
async def test_an_unreachable_ollama_is_reported_not_raised(ollama: FakeOllama) -> None:
    """The window still opens and says what is selected."""
    ollama.down = True
    listing = await localmodels.list_models()
    assert listing["reachable"] is False
    assert listing["active"] == localmodels.DEFAULT_MODEL_ID


# ── switching ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_selecting_downloads_prepares_and_records(ollama: FakeOllama) -> None:
    """A model that is not there yet is pulled, given its context, then recorded."""
    ollama.library["gemma4:31b"] = _info(
        context=262_144,
        capabilities=("completion", "tools", "thinking", "vision"),
        params="temperature                    1\ntop_k                          64",
    )
    frames: list[JsonDict] = []
    entry = await localmodels.select("gemma4", on_progress=frames.append)

    assert entry == {
        "id": "gemma4",
        "model": "medmcp-gemma4",
        "base": "gemma4:31b",
        "temperature": 1.0,
        "thinking": "medium",
        "num_ctx": localmodels.TARGET_NUM_CTX,
        "compact_threshold": int(localmodels.TARGET_NUM_CTX * localmodels.COMPACT_FRACTION),
    }
    assert ollama.created["medmcp-gemma4"]["from"] == "gemma4:31b"
    assert ollama.created["medmcp-gemma4"]["parameters"] == {"num_ctx": 131_072}
    assert settings.load_local_model() == entry
    assert settings.local_model_name() == "medmcp-gemma4"

    stages = [f["stage"] for f in frames]
    assert stages[0] == "download"
    assert stages[-2:] == ["prepare", "switch"]
    assert max(f.get("completed", 0) for f in frames) == 1050
    assert (await localmodels.list_models())["active"] == "gemma4"


@pytest.mark.asyncio
async def test_a_downloaded_model_is_not_pulled_again(ollama: FakeOllama) -> None:
    """Switching back to a model already on disk makes no download request."""
    ollama.installed["qwen3.8:27b"] = _info()
    await localmodels.select("qwen3.8")
    assert not any(c.startswith("POST /api/pull") for c in ollama.calls)


@pytest.mark.asyncio
async def test_settings_are_read_from_the_model(ollama: FakeOllama) -> None:
    """A smaller context is kept.

    No vendor temperature means the low default, no thinking means off.
    """
    ollama.library["devstral-small-2:24b"] = _info(context=40_000)
    entry = await localmodels.select("devstral-small-2")
    assert entry["num_ctx"] == 40_000
    assert entry["compact_threshold"] == 35_000
    assert entry["temperature"] == localmodels.DEFAULT_TEMPERATURE
    assert entry["thinking"] == "off"


@pytest.mark.asyncio
async def test_only_the_selected_model_stays_loaded(ollama: FakeOllama) -> None:
    """The previous model is unloaded: Ollama never does that on its own here."""
    await localmodels.select("granite4.2")
    assert ollama.loaded == []
    ollama.loaded = ["medmcp-granite4.2:latest", "muse-medmcp:latest"]
    await localmodels.unload_others("medmcp-granite4.2")
    assert ollama.loaded == ["medmcp-granite4.2:latest"]


@pytest.mark.asyncio
async def test_returning_to_the_default_clears_the_selection(ollama: FakeOllama) -> None:
    """Muse Glimmer is the absence of a choice, not another derived model."""
    await localmodels.select("granite4.2")
    ollama.loaded = ["medmcp-granite4.2:latest"]
    entry = await localmodels.select(localmodels.DEFAULT_MODEL_ID)
    assert entry == {"id": "muse-glimmer", "model": "muse-medmcp"}
    assert settings.load_local_model() is None
    assert ollama.loaded == []
    assert "muse-medmcp" not in ollama.created


@pytest.mark.asyncio
async def test_a_model_without_tool_calling_is_refused(ollama: FakeOllama) -> None:
    """The agent could do nothing with it, so it never becomes the selection."""
    ollama.library["qwen3.8:27b"] = _info(capabilities=("completion",))
    with pytest.raises(localmodels.ModelError, match="tool calling"):
        await localmodels.select("qwen3.8")
    assert settings.load_local_model() is None
    assert "medmcp-qwen3.8" not in ollama.created


@pytest.mark.asyncio
async def test_a_context_too_small_for_the_agent_is_refused(ollama: FakeOllama) -> None:
    """The system prompt and tool list alone would fill it."""
    ollama.library["qwen3.8:27b"] = _info(context=8_192)
    with pytest.raises(localmodels.ModelError, match="too small"):
        await localmodels.select("qwen3.8")
    assert settings.load_local_model() is None


@pytest.mark.asyncio
async def test_licence_is_asked_before_anything_is_downloaded(ollama: FakeOllama) -> None:
    """A model that is not Apache-2.0 makes no request at all until accepted."""
    with pytest.raises(localmodels.LicenseConsentRequiredError) as raised:
        await localmodels.select("nemotron-3.5-lightning")
    assert raised.value.model.license == "NVIDIA Open Model License"
    assert ollama.calls == []

    entry = await localmodels.select("nemotron-3.5-lightning", accept_license=True)
    assert entry["model"] == "medmcp-nemotron-3.5-lightning"


@pytest.mark.asyncio
async def test_a_failed_download_leaves_the_previous_model_selected(ollama: FakeOllama) -> None:
    """An error mid-switch changes nothing the agent reads."""
    await localmodels.select("granite4.2")
    del ollama.library["gemma4:31b"]
    with pytest.raises(localmodels.ModelError, match="download of gemma4:31b failed"):
        await localmodels.select("gemma4")
    assert settings.local_model_name() == "medmcp-granite4.2"


@pytest.mark.asyncio
async def test_an_unreachable_ollama_fails_the_switch_cleanly(ollama: FakeOllama) -> None:
    """A connection error becomes a message for the operator, not a traceback."""
    ollama.down = True
    with pytest.raises(localmodels.ModelError, match="could not reach Ollama"):
        await localmodels.select("gemma4")
    assert settings.load_local_model() is None


@pytest.mark.asyncio
async def test_unknown_model_is_not_found(ollama: FakeOllama) -> None:
    """Only catalog entries can be selected — never an arbitrary tag from a request."""
    with pytest.raises(FileNotFoundError):
        await localmodels.select("evil/model:latest")
    assert ollama.calls == []


# ── deleting ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_removes_the_model_and_its_derived_model(ollama: FakeOllama) -> None:
    """Both names go, so the disk space actually comes back."""
    await localmodels.select("gemma4")
    await localmodels.select(localmodels.DEFAULT_MODEL_ID)
    assert _row(await localmodels.list_models(), "gemma4")["deletable"] is True

    await localmodels.delete("gemma4")
    assert "gemma4:31b" not in ollama.installed
    assert "medmcp-gemma4" not in ollama.installed
    assert _row(await localmodels.list_models(), "gemma4")["downloaded"] is False


@pytest.mark.asyncio
async def test_the_model_in_use_and_the_default_cannot_be_deleted(ollama: FakeOllama) -> None:
    """Neither would leave the workspace with a model to answer."""
    await localmodels.select("gemma4")
    with pytest.raises(localmodels.ModelError, match="in use"):
        await localmodels.delete("gemma4")
    with pytest.raises(localmodels.ModelError, match="default"):
        await localmodels.delete(localmodels.DEFAULT_MODEL_ID)
    assert "gemma4:31b" in ollama.installed
    assert "muse-medmcp:latest" in ollama.installed


@pytest.mark.asyncio
async def test_deleting_a_model_that_is_not_downloaded(ollama: FakeOllama) -> None:
    """Reported as not found rather than as a success."""
    del ollama
    with pytest.raises(FileNotFoundError, match="not downloaded"):
        await localmodels.delete("qwen3.8")


# ── what the agent's config gets ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_adds_the_selection_and_leaves_the_default_entry_alone(
    ollama: FakeOllama, tmp_path: Path
) -> None:
    """A second model entry is selected; the shipped one is byte-for-byte what it was."""
    from vibe.core.config.models import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
        ModelConfig,
    )

    del ollama
    await localmodels.select("mistral-small3.2")
    cfg = _sync(tmp_path)

    assert cfg["active_model"] == settings.LOCAL_SELECTED_ALIAS
    assert [m["alias"] for m in cfg["models"]] == ["local", settings.LOCAL_SELECTED_ALIAS]
    assert cfg["models"][0] == {
        "name": "muse-medmcp",
        "provider": "ollama",
        "alias": "local",
        "temperature": 0.3,
        "thinking": "medium",
    }
    selected = ModelConfig.model_validate(cfg["models"][1])
    assert selected.name == "medmcp-mistral-small3.2"
    assert selected.provider == "ollama"
    assert selected.thinking == "off"
    assert selected.auto_compact_threshold == int(131_072 * localmodels.COMPACT_FRACTION)
    assert [p["name"] for p in cfg["providers"]] == ["ollama"]


@pytest.mark.asyncio
async def test_returning_to_the_default_restores_the_config(
    ollama: FakeOllama, tmp_path: Path
) -> None:
    """The owned entry is removed and ``active_model`` moves back."""
    del ollama
    await localmodels.select("mistral-small3.2")
    _sync(tmp_path)
    await localmodels.select(localmodels.DEFAULT_MODEL_ID)
    cfg = _sync(tmp_path)
    assert cfg["active_model"] == "local"
    assert [m["alias"] for m in cfg["models"]] == ["local"]


@pytest.mark.asyncio
async def test_a_cloud_model_wins_over_the_local_selection(
    ollama: FakeOllama, tmp_path: Path
) -> None:
    """Chats go to the cloud model.

    The local choice stays for the helper passes and the way back.
    """
    del ollama
    await localmodels.select("mistral-small3.2")
    settings.configure_cloud_model("anthropic", "claude-opus-5-5", api_key="sk-test")
    settings.acknowledge_cloud_model()
    settings.set_cloud_model_enabled(True)

    cfg = _sync(tmp_path)
    assert cfg["active_model"] == settings.CLOUD_MODEL_ALIAS
    assert [m["alias"] for m in cfg["models"]] == ["local", "local-selected", "cloud"]
    assert settings.active_model_name() == "claude-opus-5-5"
    assert settings.local_model_name() == "medmcp-mistral-small3.2"

    settings.set_cloud_model_enabled(False)
    assert _sync(tmp_path)["active_model"] == settings.LOCAL_SELECTED_ALIAS


@pytest.mark.asyncio
async def test_helper_passes_follow_the_selection(ollama: FakeOllama) -> None:
    """Explanations and titles use the one model that is loaded."""
    assert settings.local_helper_request() == {"model": "muse-medmcp", "think": False}

    await localmodels.select("mistral-small3.2")
    assert settings.local_helper_request() == {"model": "medmcp-mistral-small3.2"}

    ollama.library["gemma4:31b"] = _info(capabilities=("completion", "tools", "thinking"))
    await localmodels.select("gemma4")
    assert settings.local_helper_request() == {"model": "medmcp-gemma4", "think": False}


def test_a_malformed_selection_falls_back_to_the_default(tmp_path: Path) -> None:
    """A hand-edited state file cannot point the agent at an arbitrary string."""
    del tmp_path
    settings.LOCAL_MODEL_PATH.write_text(
        json.dumps({"selected": {"id": "x", "model": "bad name; rm -rf", "thinking": "off"}})
    )
    assert settings.load_local_model() is None
    assert settings.local_model_name() == settings.OLLAMA_MODEL


# ── API ──────────────────────────────────────────────────────────────────────


def _patch_restart(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(server, "_apply_stack_change", lambda: calls.append("sync"))

    async def _restart() -> None:
        calls.append("restart")

    async def _preload(name: str) -> None:
        calls.append(f"preload {name}")

    monkeypatch.setattr(server, "_restart_vibe", _restart)
    monkeypatch.setattr(localmodels, "preload", _preload)
    return calls


def test_switch_socket_streams_progress_then_restarts_the_agent(
    ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The browser sees the download, then one ``done``; the agent restarts on the new model."""
    del ollama
    calls = _patch_restart(monkeypatch)
    client = TestClient(server.app, base_url="http://127.0.0.1:8100")
    frames: list[JsonDict] = []
    with client.websocket_connect("/ws/models/select", headers=_WS_HEADERS) as ws:
        ws.send_json({"id": "gemma4"})
        while True:
            frame = cast("JsonDict", ws.receive_json())
            frames.append(frame)
            if frame["type"] != "progress":
                break

    assert frames[-1] == {"type": "done", "id": "gemma4", "model": "medmcp-gemma4"}
    assert {f["stage"] for f in frames[:-1]} == {"download", "prepare", "switch"}
    assert calls[:2] == ["sync", "restart"]
    assert settings.local_model_name() == "medmcp-gemma4"

    listing = cast("JsonDict", client.get("/api/models").json())
    assert listing["active"] == "gemma4"


def test_switch_socket_asks_for_the_licence(
    ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dialog is the explanation; the refusal to download is the enforcement."""
    calls = _patch_restart(monkeypatch)
    client = TestClient(server.app, base_url="http://127.0.0.1:8100")
    with client.websocket_connect("/ws/models/select", headers=_WS_HEADERS) as ws:
        ws.send_json({"id": "nemotron-3.5-lightning"})
        frame = cast("JsonDict", ws.receive_json())
    assert frame == {
        "type": "needs_license",
        "id": "nemotron-3.5-lightning",
        "label": "Nemotron 3.5 Lightning",
        "license": "NVIDIA Open Model License",
    }
    assert ollama.calls == []
    assert calls == []


def test_delete_endpoint_maps_errors(ollama: FakeOllama) -> None:
    """Unknown or absent is 404; the default or the model in use is 400."""
    del ollama
    client = TestClient(server.app, base_url="http://127.0.0.1:8100")
    assert client.delete("/api/models/nope").status_code == 404
    assert client.delete("/api/models/qwen3.8").status_code == 404
    assert client.delete("/api/models/muse-glimmer").status_code == 400


def test_usage_meter_uses_the_selected_models_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fetched window describes the default model, not the one selected."""
    monkeypatch.setattr(settings, "_context_window_tokens", 131_072)
    assert server._usage_window({"size": 35_000}, None, 40_000) == 40_000
    assert server._usage_window({"size": 180_000}, 180_000, None) == 180_000
