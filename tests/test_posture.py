"""The approval posture is put back on every sync, whatever was written to the files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from medmcp import posture, settings

REPO = Path(__file__).resolve().parents[1]


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


class TestConfig:
    """``config.toml``: the keys that decide whether a call asks first."""

    @pytest.mark.parametrize("shipped", [".vibe/config.toml", "docker/config.toml"])
    def test_shipped_configs_need_no_correction(self, shipped: str) -> None:
        """Otherwise every sync would log a restoration that never happened."""
        assert posture.enforce_config(_toml(REPO / shipped)) == []

    def test_tampered_config_is_restored(self) -> None:
        """Each way of approving without asking is undone."""
        cfg = _toml(REPO / "docker/config.toml")
        cfg["default_agent"] = "auto-approve"
        cfg["auto_approve"] = True
        cfg["bypass_tool_permissions"] = True
        cfg["applied_migrations"] = []
        cfg["disabled_tools"] = []
        cfg["tools"]["bash"] = {"permission": "always", "allowlist": ["*"]}
        cfg["tools"]["mcp_medmcp-neuro_skull_strip"] = {"permission": "always"}
        assert len(posture.enforce_config(cfg)) == 8
        assert cfg["default_agent"] == "medmcp"
        assert cfg["auto_approve"] is False
        assert cfg["bypass_tool_permissions"] is False
        assert "bash_read_only_defaults_v1" in cfg["applied_migrations"]
        assert "web_search" in cfg["disabled_tools"]
        assert cfg["tools"]["bash"] == {"permission": "ask", "allowlist": ["find"]}
        assert cfg["tools"]["mcp_medmcp-neuro_skull_strip"]["permission"] == "ask"


class TestAgentProfile:
    """The profile's keys sit above config.toml, so only the canonical ones may remain."""

    def test_shipped_profile_is_canonical(self, tmp_path: Path) -> None:
        """The checked-in profile is left byte-for-byte as it is."""
        path = tmp_path / "medmcp.toml"
        text = (REPO / ".vibe/agents/medmcp.toml").read_text()
        path.write_text(text)
        assert posture.enforce_agent_profile(path) == []
        assert path.read_text() == text

    def test_tampered_profile_is_reset_keeping_the_prompt(self, tmp_path: Path) -> None:
        """An override block or a flipped flag is removed; the prompt id is the sync's."""
        path = tmp_path / "medmcp.toml"
        path.write_text(
            'system_prompt_id = "medmcp-external"\nsafety = "yolo"\nenable_subagents = true\n'
            '[tools.bash]\npermission = "always"\n'
        )
        assert posture.enforce_agent_profile(path)
        assert _toml(path) == {
            **posture.CANONICAL_AGENT_PROFILE,
            "system_prompt_id": "medmcp-external",
        }


def test_trusted_folders_are_emptied(tmp_path: Path) -> None:
    """A trusted workspace would load its own .vibe/ config and hooks."""
    path = tmp_path / "trusted_folders.toml"
    path.write_text('trusted = ["/data"]\nuntrusted = []\n')
    assert posture.enforce_trusted_folders(path) == ["/data"]
    assert _toml(path)["trusted"] == []
    assert posture.enforce_trusted_folders(path) == []


class TestStackArgs:
    """A manifest may start a hardened container on the workspace, and nothing else."""

    WS = "/srv/data"

    def _check(self, *flags: str) -> str | None:
        args = ["run", *flags, "-v", f"{self.WS}:{self.WS}", "img:tag", "--serve", "-v", "x"]
        return posture.unsafe_stack_args(posture.split_flag_values(args), self.WS)

    def test_installed_shape_is_allowed(self) -> None:
        """What install_stack_image writes passes, GPU and egress opt-in included."""
        hardened = settings._harden_stack_run_args(
            ["run", "--rm", "-i", "--device", "nvidia.com/gpu=0", "--network", "bridge"]
        )
        assert self._check(*hardened[1:]) is None

    @pytest.mark.parametrize(
        "flags",
        [
            ("--privileged",),
            ("--network", "host"),
            ("--network=host",),
            ("--pid", "host"),
            ("--cap-add", "SYS_ADMIN"),
            ("--security-opt", "seccomp=unconfined"),
            ("-v", "/:/host"),
            ("--mount", "type=bind,src=/,dst=/host"),
            ("--device", "/dev/sda"),
        ],
    )
    def test_escapes_are_refused(self, flags: tuple[str, ...]) -> None:
        """Each of these undoes the sandbox the hardening sets up."""
        assert self._check(*flags) is not None

    def test_container_command_after_the_image_is_not_checked(self) -> None:
        """Arguments after the image go to the stack's own entry point."""
        assert self._check() is None


class TestManifestLoading:
    """A refused manifest is skipped, so the core never runs it."""

    def _load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
    ) -> list[dict[str, Any]]:
        monkeypatch.setattr(settings, "STACKS_D_PATH", tmp_path)
        monkeypatch.setenv("MEDMCP_WORKSPACE", "/srv/data")
        (tmp_path / "s.toml").write_text('name = "s"\n' + body)
        return settings._load_stack_manifests()

    def test_non_docker_command_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shell command in stacks.d would run unhardened in the core."""
        assert self._load(tmp_path, monkeypatch, 'command = "/bin/sh"\nargs = ["-c", "id"]\n') == []

    def test_env_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Manifest env reaches the docker CLI itself (LD_PRELOAD, DOCKER_HOST)."""
        body = (
            'command = "docker"\nargs = ["run", "-i", "img"]\n'
            '[env]\nLD_PRELOAD = "/srv/data/x.so"\n'
        )
        assert self._load(tmp_path, monkeypatch, body) == []

    def test_privileged_manifest_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hardening only adds flags; one that undoes it must stop the load."""
        body = 'command = "docker"\nargs = ["run", "--privileged", "-v", "/:/host", "img"]\n'
        assert self._load(tmp_path, monkeypatch, body) == []
