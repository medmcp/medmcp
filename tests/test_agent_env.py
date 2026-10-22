"""The agent's environment carries no credential it was not explicitly given."""

from __future__ import annotations

from pathlib import Path

import pytest

from medmcp import acp


def test_credentials_are_scrubbed_and_the_rest_kept() -> None:
    """Registry tokens, update secrets and anything key- or token-shaped stay out."""
    env = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "MEDMCP_WORKSPACE": "/data",
        "GHCR_TOKEN": "t",
        "GHCR_USER": "u",
        "MEDMCP_EXT_GITHUB": "t",
        "MEDMCP_CLOUD_API_KEY": "k",
        "MEDMCP_UPDATE_ENV": "{}",
        "OPENAI_API_KEY": "k",
        "AWS_SECRET_ACCESS_KEY": "k",
        "HF_TOKEN": "t",
        "DB_PASSWORD": "p",
        "SITE_KEY": "k",
    }
    assert acp.scrub_environment(env) == {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "MEDMCP_WORKSPACE": "/data",
    }


def test_the_agent_gets_exactly_what_the_provider_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A credential the agent needs comes back through extra_env, and nothing else does."""
    monkeypatch.setenv("GHCR_TOKEN", "registry")
    monkeypatch.setenv("MEDMCP_EXT_GITHUB", "stale")
    client = acp.VibeAcpClient(vibe_home=tmp_path, extra_env=lambda: {"MEDMCP_EXT_DOCS": "t"})
    env = client._agent_env()  # pyright: ignore[reportPrivateUsage]
    assert "GHCR_TOKEN" not in env
    assert "MEDMCP_EXT_GITHUB" not in env
    assert env["MEDMCP_EXT_DOCS"] == "t"
    assert env["VIBE_HOME"] == str(tmp_path)
