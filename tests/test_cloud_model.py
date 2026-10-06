"""Tests for the cloud model — the double gate, the key's path, and the sync output.

With a cloud model in force the whole conversation leaves the machine, so most of
what is asserted here is what does *not* happen: no cloud provider in the agent's
config without an acknowledgement, no key written anywhere the agent's config or
a response body can see, no key delivered to a host it was not entered for, and
no chat taken down because a key is missing.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

# pyright: reportPrivateUsage=false
from medmcp import provenance, server, settings

JsonDict = dict[str, Any]

_KEY = "sk-test-s3cret-value"

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
"""


@pytest.fixture(autouse=True)
def _isolate(  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Point every piece of on-disk state at a tmp dir, with a local-only config."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "medmcp.md").write_text(
        f"You are MedMCP.\n\n**Operational rules**\n{settings.ONPREM_RULE}\n- Other rule.\n"
    )
    (tmp_path / "config.toml").write_text(_BASE_CONFIG)
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
    monkeypatch.setattr(provenance, "VIBE_HOME", tmp_path)
    monkeypatch.setattr(settings, "CLOUD_MODEL_PATH", tmp_path / "cloud_model.json")
    monkeypatch.setattr(settings, "CLOUD_MODEL_SECRET_PATH", tmp_path / "cloud_model_secret.json")
    monkeypatch.setattr(settings, "EXTERNAL_MCP_PATH", tmp_path / "external_mcp.json")
    monkeypatch.setattr(settings, "EXTERNAL_SECRETS_PATH", tmp_path / "external_secrets.json")
    monkeypatch.setattr(settings, "ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json")
    monkeypatch.setattr(settings, "STACKS_D_PATH", tmp_path / "absent-stacks.d")
    monkeypatch.setattr(settings, "get_uv_tool_dir", lambda: None)
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    monkeypatch.delenv(settings.CLOUD_MODEL_KEY_ENV, raising=False)
    settings.load_mcp_servers.cache_clear()


def _configure(**overrides: object) -> JsonDict:
    kwargs: JsonDict = {"provider": "anthropic", "model": "claude-opus-5-5", "api_key": _KEY}
    kwargs.update(overrides)
    return settings.configure_cloud_model(**kwargs)


def _enable() -> None:
    settings.acknowledge_cloud_model()
    settings.set_cloud_model_enabled(True)


def _sync(tmp_path: Path) -> JsonDict:
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    with (tmp_path / "config.toml").open("rb") as fh:
        return tomllib.load(fh)


def _client() -> TestClient:
    return TestClient(server.app, base_url="http://127.0.0.1:8100")


# ── the double gate ──────────────────────────────────────────────────────────


def test_default_state_is_local() -> None:
    """Absent state file means disabled, unacknowledged, nothing configured."""
    assert settings.load_cloud_model() == {
        "enabled": False,
        "acknowledged_at": None,
        "model": None,
    }
    assert settings.active_cloud_model() is None
    assert settings.active_model_name() == settings.OLLAMA_MODEL


def test_enable_without_acknowledgement_is_refused() -> None:
    """The acknowledgement is a precondition, not a UI-only formality."""
    _configure()
    with pytest.raises(ValueError, match="acknowledged"):
        settings.set_cloud_model_enabled(True)
    assert settings.active_cloud_model() is None


def test_enable_without_a_model_is_refused() -> None:
    """There is nothing to switch to until a model is configured."""
    settings.acknowledge_cloud_model()
    with pytest.raises(ValueError, match="configure"):
        settings.set_cloud_model_enabled(True)


def test_configuring_alone_changes_nothing(tmp_path: Path) -> None:
    """A stored configuration is inert until the feature is switched on."""
    _configure()
    cfg = _sync(tmp_path)
    assert cfg["active_model"] == "local"
    assert [p["name"] for p in cfg["providers"]] == ["ollama"]
    assert settings.agent_secret_env() == {}


def test_acknowledge_configure_enable() -> None:
    """With all three in place the cloud model is in force."""
    _configure()
    _enable()
    cloud = settings.active_cloud_model()
    assert cloud is not None
    assert cloud["model"] == "claude-opus-5-5"
    assert settings.active_model_name() == "claude-opus-5-5"


def test_disabling_clears_the_acknowledgement() -> None:
    """Consent covers one activation, so re-enabling goes through it again."""
    _configure()
    _enable()
    settings.set_cloud_model_enabled(False)
    assert settings.load_cloud_model()["acknowledged_at"] is None
    with pytest.raises(ValueError, match="acknowledged"):
        settings.set_cloud_model_enabled(True)


def test_enabled_flag_alone_does_not_open_the_gate(tmp_path: Path) -> None:
    """A state file carrying ``enabled`` without an acknowledgement is treated as off."""
    entry = _configure()
    settings.CLOUD_MODEL_PATH.write_text(
        json.dumps({"enabled": True, "acknowledged_at": None, "model": entry})
    )
    assert settings.active_cloud_model() is None
    assert _sync(tmp_path)["active_model"] == "local"


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"provider": "bedrock"}, "invalid provider"),
        ({"model": ""}, "invalid model id"),
        ({"model": "claude opus"}, "invalid model id"),
        ({"api_key": ""}, "needs an API key"),
        ({"api_key_env": "MY_KEY"}, "not both"),
        ({"api_key": "", "api_key_env": "9BAD"}, "invalid environment variable"),
        ({"compact_threshold": 10}, "context budget"),
        ({"provider": "openai-compatible", "model": "m"}, "invalid endpoint"),
        (
            {"provider": "openai-compatible", "model": "m", "api_base": "ftp://x.example"},
            "invalid endpoint",
        ),
        (
            {
                "provider": "openai-compatible",
                "model": "m",
                "api_base": "https://user:pw@gw.example/v1",
            },
            "must not carry credentials",
        ),
        (
            {
                "provider": "openai-compatible",
                "model": "m",
                "api_base": "https://gw.example/v1?api-key=abc",
            },
            "must not carry credentials",
        ),
    ],
)
def test_rejects_bad_input(kwargs: dict[str, Any], match: str) -> None:
    """Each unusable field is refused before anything is stored."""
    with pytest.raises(ValueError, match=match):
        _configure(**kwargs)
    assert settings.load_cloud_model()["model"] is None
    assert settings.load_cloud_model_key() == ""


def test_a_preset_endpoint_cannot_be_overridden() -> None:
    """Choosing Anthropic means Anthropic's endpoint, whatever else is sent along."""
    entry = _configure(api_base="https://evil.example")
    assert entry["api_base"] == "https://api.anthropic.com"


def test_a_hand_edited_state_file_cannot_repoint_a_preset() -> None:
    """The endpoint in force for a preset comes from the preset, not the file."""
    entry = _configure()
    _enable()
    state = settings.load_cloud_model()
    state["model"] = {**entry, "api_base": "https://evil.example"}
    settings.CLOUD_MODEL_PATH.write_text(json.dumps(state))
    cloud = settings.active_cloud_model()
    assert cloud is not None
    assert cloud["api_base"] == "https://api.anthropic.com"


def test_a_gateway_may_need_no_key(tmp_path: Path) -> None:
    """An OpenAI-compatible endpoint can be configured without a credential."""
    _configure(
        provider="openai-compatible",
        model="org/model:v1",
        api_base="https://gw.example/v1/",
        api_key="",
    )
    _enable()
    cfg = _sync(tmp_path)
    cloud = next(p for p in cfg["providers"] if p["name"] == "cloud")
    assert cloud["api_base"] == "https://gw.example/v1"
    assert cloud["api_key_env_var"] == ""


# ── sync output ──────────────────────────────────────────────────────────────


def test_sync_selects_the_cloud_model(tmp_path: Path) -> None:
    """In force: a cloud provider and model after the local ones, and selected."""
    _configure()
    _enable()
    cfg = _sync(tmp_path)

    assert cfg["active_model"] == "cloud"
    assert [p["name"] for p in cfg["providers"]] == ["ollama", "cloud"]
    assert [m["alias"] for m in cfg["models"]] == ["local", "cloud"]
    # The local entries are untouched, and still first: the container entrypoint
    # rewrites the first provider's endpoint and the first model's name.
    assert cfg["providers"][0]["api_base"] == "http://localhost:11434/v1"
    assert cfg["models"][0] == {
        "name": "muse-medmcp",
        "provider": "ollama",
        "alias": "local",
        "temperature": 0.3,
    }
    assert cfg["providers"][1] == {
        "name": "cloud",
        "api_base": "https://api.anthropic.com",
        "api_key_env_var": settings.CLOUD_MODEL_KEY_ENV,
        "api_style": "anthropic",
        "backend": "generic",
    }
    assert cfg["models"][1]["name"] == "claude-opus-5-5"
    assert cfg["models"][1]["auto_compact_threshold"] == 180_000


@pytest.mark.parametrize("provider_id", list(settings.CLOUD_PROVIDERS))
def test_generated_entries_validate_against_vibes_own_models(
    tmp_path: Path, provider_id: str
) -> None:
    """What we write must satisfy vibe's schema and name an adapter vibe has.

    An ``api_style`` vibe does not know is a ``KeyError`` on the first request,
    not at load — so every preset's style is checked against vibe's own table.
    """
    from vibe.core.config.models import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
        ModelConfig,
        ProviderConfig,
    )
    from vibe.core.llm.backend.generic import (  # pyright: ignore[reportMissingTypeStubs]
        _ADAPTERS,  # pyright: ignore[reportUnknownVariableType]
    )

    _configure(provider=provider_id, model="some-model", api_base="https://gw.example/v1")
    _enable()
    cfg = _sync(tmp_path)

    provider = ProviderConfig.model_validate(cfg["providers"][-1])
    model = ModelConfig.model_validate(cfg["models"][-1])
    assert provider.api_style in _ADAPTERS
    assert model.provider == provider.name
    assert model.alias == cfg["active_model"]


def test_disabling_restores_the_local_model(tmp_path: Path) -> None:
    """Switching off removes both entries and moves ``active_model`` back."""
    _configure()
    _enable()
    _sync(tmp_path)
    settings.set_cloud_model_enabled(False)
    cfg = _sync(tmp_path)

    assert cfg["active_model"] == "local"
    assert [p["name"] for p in cfg["providers"]] == ["ollama"]
    assert [m["alias"] for m in cfg["models"]] == ["local"]
    assert "anthropic" not in (tmp_path / "config.toml").read_text()


def test_a_missing_key_falls_back_to_the_local_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named variable that is unset must not take every chat down.

    vibe refuses to build a session when the active provider's key variable is
    empty, so the cloud model is only selected once the key is actually there.
    """
    _configure(api_key="", api_key_env="SITE_LLM_KEY")
    _enable()
    monkeypatch.delenv("SITE_LLM_KEY", raising=False)
    assert settings.active_cloud_model() is None
    assert _sync(tmp_path)["active_model"] == "local"

    monkeypatch.setenv("SITE_LLM_KEY", "from-the-deployment")
    assert settings.active_cloud_model() is not None
    cfg = _sync(tmp_path)
    assert cfg["active_model"] == "cloud"
    assert cfg["providers"][-1]["api_key_env_var"] == "SITE_LLM_KEY"
    # A variable the deployment owns is inherited, not injected by this process.
    assert settings.agent_secret_env() == {}


# ── the key ──────────────────────────────────────────────────────────────────


def test_the_key_is_never_written_to_config_or_state(tmp_path: Path) -> None:
    """Only the variable's name reaches config.toml; the state file holds no key."""
    _configure()
    _enable()
    _sync(tmp_path)
    assert _KEY not in (tmp_path / "config.toml").read_text()
    assert _KEY not in settings.CLOUD_MODEL_PATH.read_text()
    for prompt in (tmp_path / "prompts").iterdir():
        assert _KEY not in prompt.read_text()


def test_the_key_file_is_not_world_readable() -> None:
    """The store holds a real credential; nothing else may read it."""
    _configure()
    assert settings.CLOUD_MODEL_SECRET_PATH.stat().st_mode & 0o077 == 0


def test_the_key_reaches_the_agent_only_while_in_force() -> None:
    """Off means the agent process does not hold the credential at all."""
    _configure()
    assert settings.agent_secret_env() == {}
    _enable()
    assert settings.agent_secret_env() == {settings.CLOUD_MODEL_KEY_ENV: _KEY}
    settings.set_cloud_model_enabled(False)
    assert settings.agent_secret_env() == {}


def test_editing_the_model_keeps_the_key() -> None:
    """Correcting a model id does not mean pasting the key again."""
    _configure()
    entry = _configure(model="claude-sonnet-5-5", api_key="")
    assert entry["api_key_env"] == settings.CLOUD_MODEL_KEY_ENV
    assert settings.load_cloud_model_key() == _KEY


def test_the_key_does_not_follow_to_another_endpoint() -> None:
    """Re-pointing the endpoint must not deliver a stored key somewhere else."""
    _configure(provider="openai-compatible", model="m", api_base="https://gw.example/v1")
    entry = _configure(
        provider="openai-compatible", model="m", api_base="https://evil.example/v1", api_key=""
    )
    assert entry["api_key_env"] == ""
    assert settings.load_cloud_model_key() == ""

    # And a provider that needs a key refuses the switch outright, leaving the
    # previous configuration and its key in place.
    _configure()
    with pytest.raises(ValueError, match="needs an API key"):
        _configure(provider="openai", model="some-model", api_key="")
    assert settings.load_cloud_model()["model"]["provider"] == "anthropic"
    assert settings.load_cloud_model_key() == _KEY


def test_naming_a_variable_drops_the_stored_key() -> None:
    """Exactly one source is in play; the other is not kept around unused."""
    _configure()
    _configure(api_key="", api_key_env="SITE_LLM_KEY")
    assert settings.load_cloud_model_key() == ""


def test_removing_forgets_the_key_and_switches_off() -> None:
    """Remove is the whole way out: configuration, key, toggle, acknowledgement."""
    _configure()
    _enable()
    settings.remove_cloud_model()
    assert settings.load_cloud_model() == {
        "enabled": False,
        "acknowledged_at": None,
        "model": None,
    }
    assert not settings.CLOUD_MODEL_SECRET_PATH.exists()
    with pytest.raises(FileNotFoundError):
        settings.remove_cloud_model()


# ── the system prompt ────────────────────────────────────────────────────────


def test_prompt_follows_the_posture(tmp_path: Path) -> None:
    """A cloud model is never told it runs on-premise, with or without external MCP."""
    _configure()
    _enable()
    assert _sync(tmp_path)["system_prompt_id"] == settings.CLOUD_SYSTEM_PROMPT_ID
    variant = (tmp_path / "prompts" / f"{settings.CLOUD_SYSTEM_PROMPT_ID}.md").read_text()
    assert settings.ONPREM_RULE not in variant
    assert settings.CLOUD_RULE in variant
    assert "- Other rule." in variant

    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    settings.add_external_server("pubmed", "streamable-http", "https://example.org/mcp")
    assert _sync(tmp_path)["system_prompt_id"] == settings.CLOUD_EXTERNAL_SYSTEM_PROMPT_ID
    both = (tmp_path / "prompts" / f"{settings.CLOUD_EXTERNAL_SYSTEM_PROMPT_ID}.md").read_text()
    assert settings.CLOUD_EXTERNAL_RULE in both

    settings.set_cloud_model_enabled(False)
    assert _sync(tmp_path)["system_prompt_id"] == settings.EXTERNAL_SYSTEM_PROMPT_ID
    settings.set_external_mcp_enabled(False)
    assert _sync(tmp_path)["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID


def test_custom_prompt_id_is_left_alone(tmp_path: Path) -> None:
    """A hand-set prompt id is not replaced by a variant this module owns."""
    (tmp_path / "config.toml").write_text(
        _BASE_CONFIG.replace('system_prompt_id = "medmcp"', 'system_prompt_id = "mine"')
    )
    _configure()
    _enable()
    assert _sync(tmp_path)["system_prompt_id"] == "mine"


# ── provenance + API ─────────────────────────────────────────────────────────


def test_the_manifest_records_where_the_model_ran(tmp_path: Path) -> None:
    """A session's record names the model and the endpoint it was sent to."""
    _configure()
    _enable()
    _sync(tmp_path)
    model = provenance.build_manifest("s1", servers=[], model_name=settings.active_model_name())[
        "model"
    ]
    assert model["name"] == "claude-opus-5-5"
    assert model["endpoint"] == "https://api.anthropic.com"


def test_the_key_never_appears_in_a_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither the configure response nor the state listing carries the value."""
    monkeypatch.setattr(server, "_apply_stack_change", lambda: None)
    client = _client()
    saved = client.put(
        "/api/cloud-model/config",
        json={"provider": "anthropic", "model": "claude-opus-5-5", "api_key": _KEY},
    )
    assert saved.status_code == 200
    assert _KEY not in saved.text

    listed = client.get("/api/cloud-model")
    assert _KEY not in listed.text
    body = cast("JsonDict", listed.json())
    assert body["model"]["key_managed"] is True
    assert body["model"]["key_present"] is True
    assert body["active"] is False
    assert {p["id"] for p in body["providers"]} == set(settings.CLOUD_PROVIDERS)


def test_api_refuses_enabling_without_consent() -> None:
    """The dialog is the explanation; the endpoint is the enforcement."""
    _configure()
    resp = _client().put("/api/cloud-model", json={"enabled": True})
    assert resp.status_code == 400
    assert settings.active_cloud_model() is None


def test_the_agent_restarts_only_when_the_model_in_force_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving a configuration while off leaves open chats alone; switching does not."""
    calls: list[str] = []
    monkeypatch.setattr(server, "_apply_stack_change", lambda: calls.append("sync"))

    async def _restart() -> None:
        calls.append("restart")

    monkeypatch.setattr(server, "_restart_vibe", _restart)
    client = _client()

    saved = client.put(
        "/api/cloud-model/config",
        json={"provider": "anthropic", "model": "claude-opus-5-5", "api_key": _KEY},
    )
    assert saved.json()["restarted"] is False
    assert calls == []

    assert client.post("/api/cloud-model/acknowledge").status_code == 200
    assert client.put("/api/cloud-model", json={"enabled": True}).json()["restarted"] is True
    assert calls == ["sync", "restart"]

    # The off switch has to reach the agent already running on the cloud model.
    assert client.put("/api/cloud-model", json={"enabled": False}).json()["restarted"] is True
    assert calls == ["sync", "restart", "sync", "restart"]
    assert settings.active_cloud_model() is None


def test_usage_meter_uses_the_cloud_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama's window describes the local model, not the one answering."""
    monkeypatch.setattr(settings, "_context_window_tokens", 131072)
    assert server._usage_window({"size": 180_000}, 180_000) == 180_000
    assert server._usage_window({}, 100_000) == 100_000
    assert server._usage_window({"size": 200_000}) == 131072
