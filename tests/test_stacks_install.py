"""Tests for container-stack install/uninstall (settings.install_stack_image, …).

Docker is mocked — `_run_docker` and `_extract_image_skills` are patched so no
real images or daemon are needed.
"""

from __future__ import annotations

import subprocess
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from medmcp import settings

LABEL = (
    '{"name": "medmcp-foo", "gpu": false, "tool_timeout_sec": 1800, '
    '"skills_path": "/app/src/medmcp_foo/skills"}'
)


def _fake_docker(label: str | None) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a `_run_docker` stub: `inspect --format` yields *label* (or <no value>)."""

    def fake(args: list[str], *, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        out = ""
        if args[0] == "inspect" and "--format" in args:
            out = label if label is not None else "<no value>"
        elif args[0] == "create":
            out = "deadbeef"
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    return fake


def _fake_extract(image: str, in_image_path: str, into_dir: Path) -> None:
    """Stub extraction: create into_dir/<basename> like a real `docker cp` of a dir."""
    (into_dir / Path(in_image_path).name).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def stacks_dir(tmp_path: Path) -> Iterator[Path]:
    """Point STACKS_D_PATH + ACTIVE_STACKS_PATH at a writable tmp location."""
    d = tmp_path / "stacks.d"
    with (
        patch("medmcp.settings.STACKS_D_PATH", d),
        patch("medmcp.settings.ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json"),
    ):
        yield d


def _read_manifest(stacks_dir: Path, name: str) -> dict[str, Any]:
    """Parse the written stacks.d/<name>.toml manifest."""
    return tomllib.loads((stacks_dir / f"{name}.toml").read_text())


class TestReadStackLabel:
    """read_stack_label parses the org.medmcp.stack image label."""

    def test_parses_label(self) -> None:
        """A well-formed label is parsed into a dict."""
        with patch("medmcp.settings._run_docker", _fake_docker(LABEL)):
            meta = settings.read_stack_label("img:tag")
        assert meta["name"] == "medmcp-foo"
        assert meta["gpu"] is False

    def test_missing_label_raises(self) -> None:
        """An image without the label raises FileNotFoundError."""
        with (
            patch("medmcp.settings._run_docker", _fake_docker(None)),
            pytest.raises(FileNotFoundError),
        ):
            settings.read_stack_label("img:tag")

    def test_malformed_label_raises(self) -> None:
        """A label that isn't valid JSON raises ValueError."""
        with (
            patch("medmcp.settings._run_docker", _fake_docker("{not json")),
            pytest.raises(ValueError),
        ):
            settings.read_stack_label("img:tag")

    def test_blank_image_raises(self) -> None:
        """A blank/whitespace image reference is rejected."""
        with pytest.raises(ValueError):
            settings.read_stack_label("  ")


class TestInstall:
    """install_stack_image writes a stacks.d manifest from the image label."""

    def test_writes_manifest_cpu(self, stacks_dir: Path) -> None:
        """A CPU stack manifest has docker run args, timeout, and extracted skills."""
        with (
            patch("medmcp.settings._run_docker", _fake_docker(LABEL)),
            patch("medmcp.settings._extract_image_skills", _fake_extract),
        ):
            name = settings.install_stack_image("ghcr.io/x/foo:dev")

        assert name == "medmcp-foo"
        m = _read_manifest(stacks_dir, "medmcp-foo")
        assert m["command"] == "docker"
        assert m["args"][-1] == "ghcr.io/x/foo:dev"
        assert "--device" not in m["args"]
        assert m["tool_timeout_sec"] == 1800.0
        assert m["skills_path"].endswith("/medmcp-foo/skills")
        assert Path(m["skills_path"]).is_dir()

    def test_gpu_adds_cdi_device(self, stacks_dir: Path) -> None:
        """A gpu=true label adds the CDI device to the docker run args."""
        gpu_label = (
            '{"name": "medmcp-foo", "gpu": true, "skills_path": "/app/src/medmcp_foo/skills"}'
        )
        with (
            patch("medmcp.settings._run_docker", _fake_docker(gpu_label)),
            patch("medmcp.settings._extract_image_skills", _fake_extract),
        ):
            settings.install_stack_image("img:dev")
        m = _read_manifest(stacks_dir, "medmcp-foo")
        assert "nvidia.com/gpu=${MEDMCP_GPU}" in m["args"]

    def test_invalid_name_in_label_raises(self, stacks_dir: Path) -> None:
        """A label name that isn't a safe identifier is rejected."""
        with (
            patch("medmcp.settings._run_docker", _fake_docker('{"name": "../evil"}')),
            pytest.raises(ValueError),
        ):
            settings.install_stack_image("img:dev")

    def test_installed_stack_is_discovered(self, stacks_dir: Path) -> None:
        """After install, load_mcp_servers discovers the new stack."""
        with (
            patch("medmcp.settings._run_docker", _fake_docker(LABEL)),
            patch("medmcp.settings._extract_image_skills", _fake_extract),
        ):
            settings.install_stack_image("img:dev")
        settings.load_mcp_servers.cache_clear()
        try:
            with (
                patch("medmcp.settings.get_uv_tool_dir", return_value=None),
                patch("medmcp.settings.VIBE_HOME", stacks_dir.parent),  # no config.toml here
            ):
                servers = settings.load_mcp_servers()
            assert any(s["name"] == "medmcp-foo" for s in servers)
        finally:
            settings.load_mcp_servers.cache_clear()


class TestUninstall:
    """uninstall_stack removes the manifest and extracted skills."""

    def test_removes_manifest_and_skills(self, stacks_dir: Path) -> None:
        """Uninstall deletes both the .toml and the skills dir."""
        with (
            patch("medmcp.settings._run_docker", _fake_docker(LABEL)),
            patch("medmcp.settings._extract_image_skills", _fake_extract),
        ):
            settings.install_stack_image("img:dev")
        assert (stacks_dir / "medmcp-foo.toml").exists()
        settings.uninstall_stack("medmcp-foo")
        assert not (stacks_dir / "medmcp-foo.toml").exists()
        assert not (stacks_dir / "medmcp-foo").exists()

    def test_missing_raises(self, stacks_dir: Path) -> None:
        """Uninstalling a stack that isn't installed raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            settings.uninstall_stack("nope")

    def test_bad_name_raises(self, stacks_dir: Path) -> None:
        """A path-traversal-ish name is rejected."""
        with pytest.raises(ValueError):
            settings.uninstall_stack("../evil")


class TestListInstalled:
    """list_installed_stacks reports name/image/gpu from the manifests."""

    def test_lists_installed(self, stacks_dir: Path) -> None:
        """An installed stack appears with its image and gpu flag."""
        with (
            patch("medmcp.settings._run_docker", _fake_docker(LABEL)),
            patch("medmcp.settings._extract_image_skills", _fake_extract),
        ):
            settings.install_stack_image("ghcr.io/x/foo:dev")
        assert settings.list_installed_stacks() == [
            {"name": "medmcp-foo", "image": "ghcr.io/x/foo:dev", "gpu": False}
        ]


class TestCatalog:
    """load_catalog reads the curated install catalog."""

    def test_reads_file(self, tmp_path: Path) -> None:
        """Valid entries are returned normalized; incomplete ones are dropped."""
        cat = tmp_path / "catalog.json"
        cat.write_text(
            '{"stacks": [{"name": "medmcp-foo", "image": "img:dev", '
            '"description": "d", "gpu": true}, {"name": "", "image": "x"}]}'
        )
        with (
            patch("medmcp.settings.CATALOG_PATH", cat),
            patch("medmcp.settings.CATALOG_URL", ""),
        ):
            entries = settings.load_catalog()
        assert entries == [
            {"name": "medmcp-foo", "image": "img:dev", "description": "d", "gpu": True}
        ]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """A missing catalog file yields an empty list (graceful)."""
        with (
            patch("medmcp.settings.CATALOG_PATH", tmp_path / "nope.json"),
            patch("medmcp.settings.CATALOG_URL", ""),
        ):
            assert settings.load_catalog() == []
