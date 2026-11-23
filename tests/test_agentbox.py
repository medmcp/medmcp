"""The isolated agent container: what it gets, and what it never gets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medmcp import agentbox, settings

from .conftest import synced_config

WS = "/srv/data"
INTERNAL = {"medmcp_agent"}

CORE: dict[str, Any] = {
    "Id": "0123456789abcdef",
    "Name": "/medmcp-medmcp-1",
    "Image": "sha256:core",
    "NetworkSettings": {
        "Networks": {
            "medmcp_default": {"IPAddress": "172.18.0.5", "IPPrefixLen": 16},
            "medmcp_agent": {"IPAddress": "172.19.0.2", "IPPrefixLen": 16},
        }
    },
    "Mounts": [
        {
            "Type": "bind",
            "Source": "/run/user/1000/docker.sock",
            "Destination": "/var/run/docker.sock",
        },
        {"Type": "volume", "Name": "medmcp_vibe-state", "Destination": "/app/.vibe/state"},
        {"Type": "volume", "Name": "medmcp_vibe-agent", "Destination": str(agentbox.AGENT_HOME)},
        {
            "Type": "volume",
            "Name": "medmcp_vibe-logs",
            "Destination": str(agentbox.VIBE_HOME / "logs"),
        },
        {"Type": "volume", "Name": "medmcp_stacks", "Destination": "/app/stacks.d"},
        {"Type": "bind", "Source": WS, "Destination": WS},
    ],
}


class TestPlan:
    """Read off the running core: four mounts, its network, its image by id."""

    def test_only_the_agents_mounts_are_carried(self) -> None:
        """No docker socket and no state volume; the agent home and stacks.d read-only."""
        box = agentbox.plan(CORE, WS, INTERNAL)
        assert box.owner == "medmcp-medmcp-1"
        assert box.image == "sha256:core"
        assert box.network == "medmcp_agent"
        assert box.gateway_host == "172.19.0.2"
        assert box.subnet == "172.19.0.0/16"
        assert box.mounts == (
            f"{WS}:{WS}",
            f"medmcp_vibe-agent:{agentbox.AGENT_HOME}:ro",
            f"medmcp_vibe-logs:{agentbox.AGENT_HOME / 'logs'}",
            "medmcp_stacks:/app/stacks.d:ro",
        )

    def test_a_core_without_the_volumes_refuses(self) -> None:
        """Isolation never degrades into running the agent with the core's mounts."""
        core = {**CORE, "Mounts": [m for m in CORE["Mounts"] if "vibe-agent" not in str(m)]}
        with pytest.raises(agentbox.IsolationUnavailableError, match="vibe-agent"):
            agentbox.plan(core, WS, INTERNAL)


def test_a_core_on_no_internal_network_refuses() -> None:
    """Without a network that has no route out, the agent would reach the internet."""
    with pytest.raises(agentbox.IsolationUnavailableError, match="internal network"):
        agentbox.plan(CORE, WS, set())


def test_docker_argv_is_hardened() -> None:
    """Capabilities, privileges and the root file system are locked down."""
    argv = agentbox.docker_argv(agentbox.plan(CORE, WS, INTERNAL), ["OLLAMA_BASE_URL"], "agent-x")
    joined = " ".join(argv)
    assert "docker.sock" not in joined
    assert "vibe-state" not in joined
    for flag in (
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--read-only",
        "--network medmcp_agent",
        f"--label {agentbox.OWNER_LABEL}=medmcp-medmcp-1",
    ):
        assert flag in joined
    assert argv[-3:] == ["--entrypoint", "medmcp-agent", "sha256:core"]
    # Values travel in the docker CLI's environment, never in its argv.
    assert "-e OLLAMA_BASE_URL" in joined


def test_forwarded_names() -> None:
    """Configuration and owed credentials go across; the core's own plumbing does not."""
    env = {
        "MEDMCP_SHIM_RELAY": "live",
        "OLLAMA_BASE_URL": "http://llm:11434",
        "LANG": "C.UTF-8",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "PATH": "/usr/bin",
        "PUBMED_TOKEN": "t",
    }
    names = agentbox.forwarded_names(env, {"PUBMED_TOKEN": "t"})
    assert names == ["LANG", "MEDMCP_SHIM_RELAY", "OLLAMA_BASE_URL", "PUBMED_TOKEN"]


def test_publish_home_mirrors_only_what_the_agent_reads(tmp_path: Path) -> None:
    """Config, hooks, profile, prompts; never state, and nothing stale."""
    home, agent = tmp_path / "vibe", tmp_path / "vibe" / "agent-home"
    (home / "agents").mkdir(parents=True)
    (home / "prompts").mkdir()
    (home / "state").mkdir()
    (home / "config.toml").write_text("a = 1\n")
    (home / "hooks.toml").write_text("hooks = []\n")
    (home / "agents" / "medmcp.toml").write_text('safety = "neutral"\n')
    (home / "prompts" / "medmcp.md").write_text("You are MedMCP.\n")
    (home / "state" / "external_secrets.json").write_text('{"x": "secret"}')
    agentbox.publish_home(home, agent)
    (agent / "prompts" / "dropped.md").write_text("old")
    agentbox.publish_home(home, agent)
    assert (agent / "config.toml").read_text() == "a = 1\n"
    assert (agent / "agents" / "medmcp.toml").exists()
    assert sorted(p.name for p in (agent / "prompts").iterdir()) == ["medmcp.md"]
    assert (agent / "trusted_folders.toml").read_text().startswith("trusted = []")
    assert (agent / "logs").is_dir()
    assert not any("secret" in p.read_text() for p in agent.rglob("*") if p.is_file())


class TestIsolationMode:
    """With isolation on, stacks run only through the broker, which sits in the agent home."""

    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
        monkeypatch.setenv(agentbox.ISOLATION_ENV, "container")

    def test_pool_is_forced_on_and_the_socket_is_shared(self) -> None:
        """The agent container has no docker socket; the broker is its only way to a stack."""
        assert settings.stack_pool_enabled() is True
        assert settings.backend_socket_path() == agentbox.AGENT_HOME / "backend.sock"

    def test_sync_publishes_the_agent_home(
        self, vibe_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every sync refreshes the read-only copy the agent runs from."""
        agent_home = vibe_home / "agent-home"
        monkeypatch.setattr(agentbox, "AGENT_HOME", agent_home)
        (vibe_home / "config.toml").write_text('active_model = "local"\n')
        cfg = synced_config(vibe_home, servers=[])
        assert cfg["default_agent"] == "medmcp"
        assert (agent_home / "config.toml").read_text() == (vibe_home / "config.toml").read_text()
        assert (agent_home / "hooks.toml").exists()
