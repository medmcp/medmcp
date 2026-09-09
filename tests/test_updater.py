"""Tests for the update helper (``medmcp.updater``) with docker and compose stubbed.

The helper is the piece that runs with no core alive to watch it, so what it
records matters most: an ``ok`` after a clean ``up``, a ``rolled_back`` when
the new release did not come up but the previous one did, and a ``failed``
with both reasons when neither did.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

# pyright: reportPrivateUsage=false
from medmcp import updater

COMPOSE_TEXT = """
services:
  medmcp:
    image: ghcr.io/medmcp/core:${MEDMCP_TAG:-main}
    env_file:
      - path: ${MEDMCP_ENV_FILE:-./medmcp.env}
        required: false
    environment:
      OLLAMA_BASE_URL: "http://llm:11434"
      MEDMCP_WORKSPACE: "${MEDMCP_WORKSPACE:?set it}"
      MEDMCP_GPU: "${MEDMCP_GPU:-all}"
"""


def _proc(code: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["docker"], returncode=code, stdout="", stderr=stderr)


class TestEnvFile:
    """Recovering the operator's env_file variables by elimination."""

    def test_compose_service_env_keys(self) -> None:
        """Compose service env keys."""
        assert updater.compose_service_env_keys(COMPOSE_TEXT) == {
            "OLLAMA_BASE_URL",
            "MEDMCP_WORKSPACE",
            "MEDMCP_GPU",
        }
        assert updater.compose_service_env_keys("not: [yaml") == set()

    def test_env_file_entries_are_recovered_by_elimination(self) -> None:
        """Only what neither the image nor the compose block set came from env_file."""
        container = {
            "PATH": "/app/.venv/bin",
            "MEDMCP_BUILD": "v0.2.3",
            "MEDMCP_WORKSPACE": "/data",
            "OLLAMA_BASE_URL": "http://llm:11434",
            "PACS_TOKEN": "s3cret",
            "HOSTNAME": "abc",
        }
        image = {"PATH": "/app/.venv/bin", "MEDMCP_BUILD": "v0.2.3"}
        declared = updater.compose_service_env_keys(COMPOSE_TEXT)
        assert updater.env_file_entries(container, image, declared) == {"PACS_TOKEN": "s3cret"}

    def test_write_env_file_is_private_and_verbatim(self, tmp_path: Path) -> None:
        """Write env file is private and verbatim."""
        path = tmp_path / "medmcp.env"
        updater.write_env_file(path, {"B": "x=y z", "A": "1"})
        assert path.read_text() == "A=1\nB=x=y z\n"
        assert path.stat().st_mode & 0o777 == 0o600


class TestApply:
    """The helper's outcomes with docker and compose stubbed."""

    @pytest.fixture
    def harness(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
        """Stub docker/compose; record every compose invocation's MEDMCP_TAG."""
        compose = tmp_path / "docker-compose.ghcr.yml"
        compose.write_text(COMPOSE_TEXT)
        monkeypatch.setattr(updater, "COMPOSE_PATH", compose)
        monkeypatch.setattr(updater, "RESULT_PATH", tmp_path / "state" / "update_result.json")
        monkeypatch.setattr(updater, "HOST_DOCKER_CONFIG", tmp_path / "absent.json")
        monkeypatch.setattr(updater, "ENV_FILE_PATH", tmp_path / "medmcp.env")
        monkeypatch.setattr(updater, "wait_for_docker", lambda: True)
        old = {
            "Config": {
                "Image": "ghcr.io/medmcp/core:v0.2.3",
                "Env": ["PATH=/x", "MEDMCP_WORKSPACE=/data", "PACS_TOKEN=s3cret"],
            }
        }
        image = {"Config": {"Env": ["PATH=/x"]}}

        def _inspect(ref: str, *, image: bool = False) -> dict[str, Any] | None:
            return {"Config": {"Env": ["PATH=/x"]}} if image else old

        monkeypatch.setattr(updater, "_inspect", _inspect)
        calls: list[dict[str, str]] = []
        outcomes: list[subprocess.CompletedProcess[str]] = []

        def _up(
            *, project: str, project_dir: str, compose_file: Path, env: dict[str, str]
        ) -> subprocess.CompletedProcess[str]:
            content = ""
            path = env.get("MEDMCP_ENV_FILE")
            if path and Path(path).exists():
                content = Path(path).read_text()
            calls.append(
                {"project": project, "dir": project_dir, "env_file_content": content, **env}
            )
            return outcomes.pop(0)

        monkeypatch.setattr(updater, "compose_up", _up)
        del image
        return {"calls": calls, "outcomes": outcomes, "tmp": tmp_path}

    @staticmethod
    def _env() -> dict[str, str]:
        return {
            "MEDMCP_BUILD": "v0.3.0",
            "MEDMCP_UPDATE_REPLACE": "abc123",
            "MEDMCP_UPDATE_PROJECT": "medmcp",
            "MEDMCP_UPDATE_PROJECT_DIR": "/home/op/.cache/docker-compose/x",
            "MEDMCP_UPDATE_PREVIOUS_TAG": "v0.2.3",
            "MEDMCP_UPDATE_ENV": json.dumps({"MEDMCP_WORKSPACE": "/data", "MEDMCP_GPU": "0"}),
            "PATH": "/usr/bin",
        }

    def _result(self, harness: dict[str, Any]) -> dict[str, Any]:
        return json.loads(updater.RESULT_PATH.read_text())

    def test_clean_update(self, harness: dict[str, Any]) -> None:
        """Clean update."""
        harness["outcomes"].append(_proc(0))
        assert updater.apply(self._env()) == 0
        (call,) = harness["calls"]
        assert call["project"] == "medmcp"
        assert call["dir"] == "/home/op/.cache/docker-compose/x"
        assert call["MEDMCP_TAG"] == "v0.3.0"
        assert call["MEDMCP_WORKSPACE"] == "/data"
        assert call["MEDMCP_GPU"] == "0"
        assert "MEDMCP_UPDATE_ENV" not in call
        # The operator's env_file secret travelled with the update, off the host,
        # and was removed again once compose had read it.
        assert call["MEDMCP_ENV_FILE"] == str(updater.ENV_FILE_PATH)
        assert call["env_file_content"] == "PACS_TOKEN=s3cret\n"
        assert not updater.ENV_FILE_PATH.exists()
        result = self._result(harness)
        assert (result["status"], result["from"], result["to"]) == ("ok", "v0.2.3", "v0.3.0")

    def test_failed_up_rolls_back(self, harness: dict[str, Any]) -> None:
        """Failed up rolls back."""
        harness["outcomes"] += [_proc(1, "core failed to start"), _proc(0)]
        assert updater.apply(self._env()) == 1
        tags = [c["MEDMCP_TAG"] for c in harness["calls"]]
        assert tags == ["v0.3.0", "v0.2.3"]
        result = self._result(harness)
        assert result["status"] == "rolled_back"
        assert "core failed to start" in result["detail"]

    def test_failed_rollback_is_reported_with_both_reasons(self, harness: dict[str, Any]) -> None:
        """Failed rollback is reported with both reasons."""
        harness["outcomes"] += [_proc(1, "new broke"), _proc(1, "old broke too")]
        assert updater.apply(self._env()) == 3
        result = self._result(harness)
        assert result["status"] == "failed"
        assert "new broke" in result["detail"]
        assert "old broke too" in result["detail"]

    def test_missing_contract_is_recorded_not_raised(self, harness: dict[str, Any]) -> None:
        """Missing contract is recorded not raised."""
        env = self._env()
        del env["MEDMCP_UPDATE_PROJECT"]
        assert updater.apply(env) == 2
        assert harness["calls"] == []
        assert self._result(harness)["status"] == "failed"
