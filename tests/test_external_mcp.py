"""Tests for external MCP servers — the double gate, validation, and sync output.

This feature deliberately crosses the on-premise boundary the rest of the product
guarantees, so most of what is asserted here is what does *not* happen: no
discovery without an acknowledgement, no secret written to disk, no remote entry
shadowing a local stack, and no relaxed system prompt unless one is actually in use.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

# pyright: reportPrivateUsage=false
from medmcp import server, settings

JsonDict = dict[str, Any]


@pytest.fixture(autouse=True)
def _isolate(  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Point every piece of on-disk state at a tmp dir."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "medmcp.md").write_text(
        f"You are MedMCP.\n\n**Operational rules**\n{settings.ONPREM_RULE}\n- Other rule.\n"
    )
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
    monkeypatch.setattr(settings, "EXTERNAL_MCP_PATH", tmp_path / "external_mcp.json")
    monkeypatch.setattr(settings, "EXTERNAL_SECRETS_PATH", tmp_path / "external_secrets.json")
    monkeypatch.setattr(settings, "ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json")
    monkeypatch.setattr(settings, "STACKS_D_PATH", tmp_path / "absent-stacks.d")
    # Source #1 too, or the machine's real uv-tool stacks turn up in discovery.
    monkeypatch.setattr(settings, "get_uv_tool_dir", lambda: None)
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    settings.load_mcp_servers.cache_clear()


def _entries(tmp_path: Path) -> dict[str, JsonDict]:
    with (tmp_path / "config.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    return {str(e["name"]): e for e in cast("list[JsonDict]", cfg["mcp_servers"])}


def _add(
    name: str = "pubmed",
    transport: str = "streamable-http",
    url: str = "https://example.org/mcp",
    api_key_env: str = "",
    api_key_header: str = "",
    api_key_format: str = "",
) -> JsonDict:
    return settings.add_external_server(
        name,
        transport,
        url,
        api_key_env,
        True,
        api_key_header,
        api_key_format,
    )


def _add_with_token(name: str, token: str) -> JsonDict:
    return settings.add_external_server(
        name, "streamable-http", "https://example.org/mcp", token=token
    )


# ── the double gate ──────────────────────────────────────────────────────────


def test_default_state_is_off() -> None:
    """Absent state file means disabled, unacknowledged, no servers."""
    state = settings.load_external_mcp()
    assert state == {"enabled": False, "acknowledged_at": None, "servers": []}
    assert settings.external_mcp_enabled() is False


def test_enable_without_acknowledgement_is_refused() -> None:
    """The acknowledgement is a precondition, not a UI-only formality."""
    with pytest.raises(ValueError, match="acknowledged"):
        settings.set_external_mcp_enabled(True)
    assert settings.external_mcp_enabled() is False


def test_acknowledge_then_enable() -> None:
    """With the acknowledgement recorded, the toggle takes effect."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    assert settings.external_mcp_enabled() is True


def test_acknowledgement_is_idempotent() -> None:
    """Re-acknowledging keeps the original timestamp rather than resetting it."""
    first = settings.acknowledge_external_mcp()["acknowledged_at"]
    assert settings.acknowledge_external_mcp()["acknowledged_at"] == first


def test_disabling_clears_the_acknowledgement() -> None:
    """Consent covers one activation, so turning it off withdraws it."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    settings.set_external_mcp_enabled(False)
    assert settings.external_mcp_acknowledged() is False
    assert settings.load_external_mcp()["acknowledged_at"] is None


def test_re_enabling_needs_fresh_consent() -> None:
    """Off and on again must go through the dialog, not inherit the old decision."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    settings.set_external_mcp_enabled(False)
    with pytest.raises(ValueError, match="acknowledged"):
        settings.set_external_mcp_enabled(True)
    assert settings.external_mcp_enabled() is False
    # …and it works again once the operator has re-read and accepted it.
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    assert settings.external_mcp_enabled() is True


def test_enabled_flag_alone_does_not_open_the_gate() -> None:
    """A hand-edited state file claiming enabled without an ack is still off."""
    settings.EXTERNAL_MCP_PATH.write_text(
        json.dumps(
            {"enabled": True, "acknowledged_at": None, "servers": [{"name": "x", "active": True}]}
        )
    )
    assert settings.external_mcp_enabled() is False
    assert settings.external_servers() == []


def test_servers_hidden_until_enabled() -> None:
    """A configured, active server is not discovered while the feature is off."""
    settings.acknowledge_external_mcp()
    _add()
    assert settings.external_servers() == []
    settings.set_external_mcp_enabled(True)
    assert [s["name"] for s in settings.external_servers()] == ["pubmed"]


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "Bad Name"}, "invalid server name"),
        ({"transport": "stdio"}, "invalid transport"),
        ({"url": "ftp://example.org"}, "invalid url"),
        ({"url": "not-a-url"}, "invalid url"),
        ({"api_key_env": "not valid!"}, "invalid environment variable"),
    ],
)
def test_rejects_bad_input(kwargs: dict[str, Any], match: str) -> None:
    """Each field is validated before anything is written."""
    settings.acknowledge_external_mcp()
    with pytest.raises(ValueError, match=match):
        _add(**kwargs)
    assert settings.load_external_mcp()["servers"] == []


def test_stdio_is_not_an_allowed_transport() -> None:
    """Stdio would mean launching an arbitrary binary on the host — not offered."""
    assert "stdio" not in settings.EXTERNAL_MCP_TRANSPORTS


def test_duplicate_name_rejected() -> None:
    """Two servers cannot share a name."""
    settings.acknowledge_external_mcp()
    _add()
    with pytest.raises(ValueError, match="already exists"):
        _add()


def test_cannot_shadow_an_installed_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote entry must not be able to take an audited local stack's name."""
    monkeypatch.setattr(
        settings,
        "load_mcp_servers",
        lambda: [{"name": "medmcp-neuro", "command": "/bin/medmcp-neuro"}],
    )
    settings.acknowledge_external_mcp()
    with pytest.raises(ValueError, match="already the name of an installed stack"):
        _add("medmcp-neuro")


# ── discovery + sync ─────────────────────────────────────────────────────────


def test_sync_writes_http_entry_without_command(tmp_path: Path) -> None:
    """The synced entry is an HTTP server: transport + url, and no command/args."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add(api_key_env="PUBMED_TOKEN")
    settings.load_mcp_servers.cache_clear()

    settings.sync_servers_to_vibe_config(settings.active_servers())
    entry = _entries(tmp_path)["pubmed"]
    assert entry["transport"] == "streamable-http"
    assert entry["url"] == "https://example.org/mcp"
    assert "command" not in entry
    assert "args" not in entry
    assert entry["auth"] == {"type": "static", "api_key_env": "PUBMED_TOKEN"}


@pytest.mark.parametrize("env_var", ["", "PUBMED_TOKEN"])
def test_generated_entry_validates_against_vibes_own_model(
    tmp_path: Path, env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What we write must satisfy vibe's schema, not merely look plausible.

    vibe's ``MCPAuth`` is a discriminated union whose members forbid extra keys,
    so a missing ``type`` or a misspelled field is a hard validation error and the
    server silently fails to load. Asserting our own dict shape cannot catch that
    — only vibe's model can, so parse the real synced entry with it.
    """
    from vibe.core.config.models import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
        MCPStreamableHttp,
    )

    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add(api_key_env=env_var)
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())

    parsed = MCPStreamableHttp.model_validate(_entries(tmp_path)["pubmed"])
    assert parsed.url == "https://example.org/mcp"

    # And the token actually reaches the wire, read from the environment at
    # request time rather than from anything we stored.
    if env_var:
        monkeypatch.setenv(env_var, "s3cret")
        assert parsed.http_headers() == {"Authorization": "Bearer s3cret"}
    else:
        assert parsed.http_headers() == {}


def test_custom_header_and_format_reach_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An API-key service (non-bearer) is configurable and sends the right header."""
    from vibe.core.config.models import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
        MCPStreamableHttp,
    )

    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add(api_key_env="SVC_KEY", api_key_header="X-API-Key", api_key_format="{token}")
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())

    monkeypatch.setenv("SVC_KEY", "abc123")
    parsed = MCPStreamableHttp.model_validate(_entries(tmp_path)["pubmed"])
    assert parsed.http_headers() == {"X-API-Key": "abc123"}


def test_default_scheme_writes_no_override_keys(tmp_path: Path) -> None:
    """A bearer-token server produces the same config it did before these fields."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add(api_key_env="TOK")
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    assert _entries(tmp_path)["pubmed"]["auth"] == {"type": "static", "api_key_env": "TOK"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"api_key_env": "T", "api_key_header": "bad header"}, "invalid header name"),
        ({"api_key_env": "T", "api_key_format": "Bearer"}, "must contain"),
        ({"api_key_env": "T", "api_key_format": "{token} {other}"}, "may only reference"),
        ({"api_key_env": "T", "api_key_format": "{{token}}"}, "must contain"),
        ({"api_key_header": "X-API-Key"}, "needs an environment variable"),
    ],
)
def test_rejects_bad_auth_scheme(kwargs: dict[str, Any], match: str) -> None:
    """Header/format are validated the same way vibe validates them.

    The last case matters most: vibe silently ignores these fields without a
    token, so storing them would look configured and do nothing.
    """
    settings.acknowledge_external_mcp()
    with pytest.raises(ValueError, match=match):
        _add(**kwargs)


def test_added_server_is_active_even_with_an_explicit_active_set() -> None:
    """Once active_stacks.json exists it is an explicit list, so add must join it."""
    settings.save_active_server_names({"medmcp-neuro"})
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add()
    settings.load_mcp_servers.cache_clear()
    assert "pubmed" in settings.load_active_server_names()
    assert "pubmed" in [s["name"] for s in settings.active_servers()]


def test_no_token_is_ever_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the env var *name* reaches disk — never the credential it holds.

    Asserts on the secret's value rather than on field names, so it keeps testing
    the actual property as the stored shape grows.
    """
    monkeypatch.setenv("PUBMED_TOKEN", "s3cret-do-not-persist")
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add(api_key_env="PUBMED_TOKEN")
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())

    for path in (tmp_path / "external_mcp.json", tmp_path / "config.toml"):
        text = path.read_text()
        assert "PUBMED_TOKEN" in text  # the name is expected
        assert "s3cret-do-not-persist" not in text  # the value must never be


def test_disabling_removes_the_entries(tmp_path: Path) -> None:
    """Turning the feature off takes the servers out of the vibe config."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add()
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    assert "pubmed" in _entries(tmp_path)

    settings.set_external_mcp_enabled(False)
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    assert "pubmed" not in _entries(tmp_path)


def test_nothing_remote_survives_in_the_config(tmp_path: Path) -> None:
    """After disabling, the written config holds no remote entry of any kind.

    Asserting a known name is gone would miss an entry that lingers under another
    key, so this asks the stronger question: does anything in the file still point
    off this machine? The raw text is checked too — a URL surviving in a comment
    or a stale table is still a thing a reader would have to explain away.
    """
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add("pubmed", url="https://example.org/mcp")
    _add("vendor", url="https://vendor.example/mcp")
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    assert len(_entries(tmp_path)) == 2

    settings.set_external_mcp_enabled(False)
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())

    for name, entry in _entries(tmp_path).items():
        assert "url" not in entry, name
        assert "auth" not in entry, name
        assert str(entry.get("transport", "")) not in settings.EXTERNAL_MCP_TRANSPORTS, name
    raw = (tmp_path / "config.toml").read_text()
    assert "example.org" not in raw
    assert "vendor.example" not in raw


def test_both_active_and_inactive_servers_are_dropped(tmp_path: Path) -> None:
    """The switch is a master switch: per-server state does not survive it."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add("pubmed")
    _add("vendor", url="https://vendor.example/mcp")
    settings.set_external_server_active("vendor", False)

    settings.set_external_mcp_enabled(False)
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    assert _entries(tmp_path) == {}
    assert settings.external_servers() == []


def test_the_off_switch_reaches_the_running_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling re-syncs the config *and* restarts vibe-acp.

    Dropping the entries from config.toml is only half of it — the agent process
    already running was started with them, so without the restart the live
    session would keep talking to whatever it had connected.
    """
    calls: list[str] = []
    monkeypatch.setattr(server, "_apply_stack_change", lambda: calls.append("sync"))

    async def _restart() -> None:
        calls.append("restart")

    monkeypatch.setattr(server, "_restart_vibe", _restart)
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)

    resp = TestClient(server.app).put("/api/external-mcp", json={"enabled": False})

    assert resp.status_code == 200
    assert calls == ["sync", "restart"]
    assert settings.external_mcp_enabled() is False


def test_token_presence_is_reported_without_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API says whether the named variable holds a token — never what it holds."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add("pubmed", api_key_env="PUBMED_TOKEN")
    monkeypatch.setenv("PUBMED_TOKEN", "s3cret-value")

    body = TestClient(server.app).get("/api/external-mcp").json()

    server_entry = cast("list[JsonDict]", body["servers"])[0]
    assert server_entry["token_present"] is True
    assert "s3cret-value" not in json.dumps(body)


def test_missing_token_variable_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A named variable that does not exist is flagged, so the UI can say so.

    Without this the only symptom is the remote service's 401 — vibe attaches the
    header only when the variable is truthy, so the request goes out with no
    credential rather than failing where the cause is visible.
    """
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add("pubmed", api_key_env="PUBMED_TOKEN")
    monkeypatch.delenv("PUBMED_TOKEN", raising=False)

    body = TestClient(server.app).get("/api/external-mcp").json()
    assert cast("list[JsonDict]", body["servers"])[0]["token_present"] is False


def test_empty_token_variable_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty variable is absent for our purposes, matching vibe's own check."""
    monkeypatch.setenv("PUBMED_TOKEN", "")
    assert settings.external_token_present("PUBMED_TOKEN") is False
    assert settings.external_token_present("") is False


# ── tokens entered in the UI ─────────────────────────────────────────────────


def test_stored_token_never_reaches_the_config(tmp_path: Path) -> None:
    """A token typed into the UI is not written to the file vibe reads."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add_with_token("pubmed", "s3cret-value")
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())

    raw = (tmp_path / "config.toml").read_text()
    assert "s3cret-value" not in raw
    assert "MEDMCP_EXT_PUBMED" in raw
    entry = _entries(tmp_path)["pubmed"]
    assert cast("JsonDict", entry["auth"])["api_key_env"] == "MEDMCP_EXT_PUBMED"


def test_stored_token_never_appears_in_a_response() -> None:
    """Neither the add response nor the state listing carries the value."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    client = TestClient(server.app)

    added = client.post(
        "/api/external-mcp/servers",
        json={
            "name": "pubmed",
            "transport": "streamable-http",
            "url": "https://example.org/mcp",
            "token": "s3cret-value",
        },
    )
    assert added.status_code == 200
    assert "s3cret-value" not in added.text

    listed = client.get("/api/external-mcp")
    assert "s3cret-value" not in listed.text
    entry = cast("list[JsonDict]", listed.json()["servers"])[0]
    assert entry["token_managed"] is True
    assert entry["token_present"] is True


def test_stored_token_reaches_the_agent_environment() -> None:
    """The token is handed to the agent subprocess under the managed name."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add_with_token("pubmed", "s3cret-value")
    assert settings.external_secret_env() == {"MEDMCP_EXT_PUBMED": "s3cret-value"}


def test_no_token_is_handed_over_while_the_feature_is_off() -> None:
    """Off means the agent process does not even hold the credential."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add_with_token("pubmed", "s3cret-value")
    settings.set_external_mcp_enabled(False)
    assert settings.external_secret_env() == {}


def test_no_token_is_handed_over_for_a_deactivated_server() -> None:
    """A server switched off individually does not put its token in the agent."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add_with_token("pubmed", "s3cret-value")
    settings.set_external_server_active("pubmed", False)
    assert settings.external_secret_env() == {}


def test_token_and_env_var_together_are_refused() -> None:
    """Two sources would mean one silently loses; say so instead."""
    with pytest.raises(ValueError, match="either a token or an environment variable"):
        settings.add_external_server(
            "pubmed",
            "streamable-http",
            "https://example.org/mcp",
            api_key_env="PUBMED_TOKEN",
            token="s3cret-value",
        )


def test_removing_a_server_forgets_its_token() -> None:
    """Nothing is left behind that a later server of the same name would inherit."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add_with_token("pubmed", "s3cret-value")
    settings.remove_external_server("pubmed")
    assert settings.load_external_secrets() == {}


def test_replacing_a_token_adopts_the_managed_variable() -> None:
    """A hand-plumbed server can move to a stored token, with one source in play."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add("pubmed", api_key_env="PUBMED_TOKEN")
    settings.replace_external_token("pubmed", "s3cret-value")

    stored = cast("list[JsonDict]", settings.load_external_mcp()["servers"])[0]
    assert stored["api_key_env"] == "MEDMCP_EXT_PUBMED"
    assert settings.external_secret_env() == {"MEDMCP_EXT_PUBMED": "s3cret-value"}


def test_secret_file_is_not_world_readable() -> None:
    """The store holds real credentials; it must not be readable by anything else."""
    settings.set_external_secret("pubmed", "s3cret-value")
    assert settings.EXTERNAL_SECRETS_PATH.stat().st_mode & 0o077 == 0


def test_synced_entry_is_not_readopted_as_a_stdio_stack(tmp_path: Path) -> None:
    """Discovery must not pick its own HTTP output back up as a broken local stack."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add()
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())

    settings.set_external_mcp_enabled(False)
    settings.load_mcp_servers.cache_clear()
    assert [s["name"] for s in settings.load_mcp_servers()] == []


def test_deactivating_one_server_leaves_it_configured(tmp_path: Path) -> None:
    """Deactivating hides a server from discovery without forgetting its settings."""
    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add()
    settings.set_external_server_active("pubmed", False)
    assert settings.external_servers() == []
    assert len(cast("list[JsonDict]", settings.load_external_mcp()["servers"])) == 1


def test_remove_unknown_server_raises() -> None:
    """Removing a name that was never registered is a 404, not a silent no-op."""
    with pytest.raises(FileNotFoundError):
        settings.remove_external_server("nope")


# ── system prompt ────────────────────────────────────────────────────────────


def test_prompt_switches_only_while_external_is_in_use(tmp_path: Path) -> None:
    """The relaxed prompt applies when an external server is active, and not otherwise."""
    settings.sync_servers_to_vibe_config([])
    with (tmp_path / "config.toml").open("rb") as fh:
        assert tomllib.load(fh)["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID

    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add()
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())

    with (tmp_path / "config.toml").open("rb") as fh:
        assert tomllib.load(fh)["system_prompt_id"] == settings.EXTERNAL_SYSTEM_PROMPT_ID
    variant = (tmp_path / "prompts" / f"{settings.EXTERNAL_SYSTEM_PROMPT_ID}.md").read_text()
    assert settings.ONPREM_RULE not in variant
    assert settings.EXTERNAL_RULE in variant

    settings.set_external_mcp_enabled(False)
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    with (tmp_path / "config.toml").open("rb") as fh:
        assert tomllib.load(fh)["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID


def test_custom_prompt_id_is_left_alone(tmp_path: Path) -> None:
    """Only a prompt id this sync owns is rewritten."""
    (tmp_path / "config.toml").write_text('system_prompt_id = "my-own"\n')
    settings.sync_servers_to_vibe_config([])
    with (tmp_path / "config.toml").open("rb") as fh:
        assert tomllib.load(fh)["system_prompt_id"] == "my-own"


def test_missing_anchor_keeps_the_base_prompt(tmp_path: Path) -> None:
    """A prompt edit that drops the rule must not silently yield a no-op variant."""
    (tmp_path / "prompts" / "medmcp.md").write_text("You are MedMCP.\n")
    assert settings.write_external_prompt_variant() is False

    settings.acknowledge_external_mcp()
    settings.set_external_mcp_enabled(True)
    _add()
    settings.load_mcp_servers.cache_clear()
    settings.sync_servers_to_vibe_config(settings.active_servers())
    with (tmp_path / "config.toml").open("rb") as fh:
        assert tomllib.load(fh)["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID


def test_shipped_prompt_still_contains_the_anchor() -> None:
    """Guards the real prompt against drift.

    ``write_external_prompt_variant`` degrades to "keep the base prompt" when the
    rule it rewrites is gone, which is the safe direction but also a silent one —
    the feature would just stop relaxing the prompt. This fails loudly instead.
    """
    shipped = Path(__file__).resolve().parents[1] / ".vibe" / "prompts" / "medmcp.md"
    assert settings.ONPREM_RULE in shipped.read_text()
