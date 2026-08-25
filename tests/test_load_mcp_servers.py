"""Tests for load_mcp_servers() — uv tool env scanning and config.toml fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from medmcp import settings
from medmcp.settings import load_mcp_servers

JsonDict = dict[str, Any]


def _clear_cache() -> None:
    load_mcp_servers.cache_clear()


def _make_tool_env(
    tool_dir: Path,
    name: str,
    ep_value: str = "",
) -> tuple[Path, Path]:
    """Create a minimal fake uv tool env under *tool_dir*/<name>.

    Returns ``(python_path, executable_path)`` — both are created as empty files
    so ``candidate.exists()`` checks pass.  The entry_points.txt is written with
    a ``[medmcp.stacks]`` section pointing to *ep_value* (defaults to
    ``<module>:get_mcp_config`` derived from *name*).
    """
    module = name.replace("-", "_")
    value = ep_value or f"{module}:get_mcp_config"

    dist_info = (
        tool_dir / name / "lib" / "python3.12" / "site-packages" / f"{module}-0.0.0.dist-info"
    )
    dist_info.mkdir(parents=True)
    (dist_info / "entry_points.txt").write_text(f"[medmcp.stacks]\n{name} = {value}\n")

    bin_dir = tool_dir / name / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.touch()
    executable = bin_dir / name
    executable.touch()
    return python, executable


# ── uv tool env discovery ─────────────────────────────────────────────────────


class TestUvToolDiscovery:
    """load_mcp_servers scans uv tool envs for [medmcp.stacks] entry points."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test so patches take effect."""
        _clear_cache()

    def test_single_tool_discovered(self, tmp_path: Path) -> None:
        """A stack installed as a uv tool yields one server entry."""
        _make_tool_env(tmp_path, "medmcp-test")
        config: JsonDict = {"name": "medmcp-test", "command": "medmcp-test", "args": []}

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value=config),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "medmcp-test"

    def test_command_resolved_to_absolute_path(self, tmp_path: Path) -> None:
        """The command is resolved to the absolute executable inside the tool env."""
        _make_tool_env(tmp_path, "medmcp-test")
        config = {"name": "medmcp-test", "command": "medmcp-test"}

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value=config),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        expected = str(tmp_path / "medmcp-test" / "bin" / "medmcp-test")
        assert servers[0]["command"] == expected

    def test_env_defaults_to_empty_dict(self, tmp_path: Path) -> None:
        """Env field defaults to {} when not returned by the entry point."""
        _make_tool_env(tmp_path, "medmcp-x")
        config = {"name": "medmcp-x", "command": "medmcp-x"}

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value=config),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert servers[0]["env"] == {}

    def test_broken_entry_point_is_skipped(self, tmp_path: Path) -> None:
        """An entry point whose callable raises is skipped; others still load."""
        _make_tool_env(tmp_path, "medmcp-bad")
        _make_tool_env(tmp_path, "medmcp-ok")

        def _side_effect(python: Path, module: str, attr: str) -> object:
            if "bad" in str(python):
                raise ImportError("missing heavy dep")
            return {"name": "medmcp-ok", "command": "medmcp-ok"}

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", side_effect=_side_effect),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        names = [s["name"] for s in servers]
        assert "medmcp-ok" in names
        assert "medmcp-bad" not in names

    def test_invalid_dict_is_skipped(self, tmp_path: Path) -> None:
        """An entry point returning a dict without 'name' is ignored."""
        _make_tool_env(tmp_path, "medmcp-bad")

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value={"command": "uvx"}),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert servers == []

    def test_malformed_ep_value_is_skipped(self, tmp_path: Path) -> None:
        """An entry_points.txt value without 'module:attr' syntax is ignored."""
        _make_tool_env(tmp_path, "medmcp-bad", ep_value="not_valid")

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert servers == []

    def test_no_uv_tool_dir_returns_empty(self, tmp_path: Path) -> None:
        """When uv is unavailable, discovery falls back to config.toml only."""
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert servers == []

    def test_empty_command_not_resolved_to_bin_dir(self, tmp_path: Path) -> None:
        """An entry point returning command='' must not have its command set to the bin dir.

        Path / "" collapses to the parent (the bin/ dir itself), which exists,
        so without an explicit guard command would be overwritten with a directory path.
        """
        _make_tool_env(tmp_path, "medmcp-test")
        config = {"name": "medmcp-test", "command": ""}

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value=config),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert len(servers) == 1
        # command must stay empty, not become the bin directory path
        assert servers[0]["command"] == ""


# ── Config-toml fallback ──────────────────────────────────────────────────────


class TestConfigTomlFallback:
    """config.toml [[mcp_servers]] entries are accepted for names not in uv tools."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test so patches take effect."""
        _clear_cache()

    def test_config_only_still_works(self, tmp_path: Path) -> None:
        """Stacks not yet installed as uv tools can still be declared manually."""
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[project]\n[[mcp_servers]]\nname = "medmcp-neuro"\n'
            'command = "medmcp-neuro"\nargs = ["medmcp-neuro"]\n'
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "medmcp-neuro"

    def test_uv_tool_wins_over_config_same_name(self, tmp_path: Path) -> None:
        """When the same name appears in both sources, the uv tool wins."""
        _make_tool_env(tmp_path, "medmcp-neuro")
        ep_config = {"name": "medmcp-neuro", "command": "medmcp-neuro"}
        cfg = tmp_path / "config.toml"
        cfg.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "/stale/path"\n')

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value=ep_config),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert len(servers) == 1
        # command is the resolved absolute path, not the stale config value
        assert servers[0]["command"] != "/stale/path"

    def test_uv_tool_and_config_different_names_both_appear(self, tmp_path: Path) -> None:
        """Uv tool and config entries with distinct names are merged."""
        _make_tool_env(tmp_path, "medmcp-cardiac")
        ep_config = {"name": "medmcp-cardiac", "command": "medmcp-cardiac"}
        cfg = tmp_path / "config.toml"
        cfg.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "medmcp-neuro"\n')

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value=ep_config),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        names = {s["name"] for s in servers}
        assert names == {"medmcp-neuro", "medmcp-cardiac"}

    def test_orphaned_docker_entry_dropped(self, tmp_path: Path) -> None:
        """A leftover container-stack entry (command "docker") with no manifest is dropped.

        sync_servers_to_vibe_config writes active container stacks into config.toml;
        once the stacks.d manifest is uninstalled, that entry must not resurrect the
        stack via the config.toml fallback (the uninstall-doesn't-stick bug).
        """
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            '[[mcp_servers]]\nname = "medmcp-neuro"\n'
            'command = "docker"\nargs = ["run", "--rm", "-i", "medmcp-neuro:dev"]\n'
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert servers == []

    def test_orphaned_proxy_entry_dropped(self, tmp_path: Path) -> None:
        """The same leftover, spelled the way the sync actually writes it.

        When the backend pool is in the picture the sync writes the stack's command
        as an absolute path to the ``medmcp-mcp-proxy`` shim, not ``docker``. That
        path exists, so neither the stale-path check nor a docker-only test drops
        it, and an uninstalled stack came back as a manual entry: gone from the
        stacks list, still present in settings, and re-synced into the config on the
        next write when no explicit active set narrowed it away.
        """
        proxy = tmp_path / "bin" / settings.PROXY_COMMAND
        proxy.parent.mkdir(parents=True)
        proxy.touch()
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f'[[mcp_servers]]\nname = "medmcp-neuro"\n'
            f'command = "{proxy}"\nargs = ["medmcp-neuro"]\n'
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert servers == []

    def test_a_genuine_manual_entry_still_loads(self, tmp_path: Path) -> None:
        """The orphan rule must not swallow a hand-written server."""
        cfg = tmp_path / "config.toml"
        cfg.write_text('[[mcp_servers]]\nname = "my-own-tool"\ncommand = "my-server"\nargs = []\n')
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert [s["name"] for s in servers] == ["my-own-tool"]

    def test_no_tools_no_config_returns_empty(self, tmp_path: Path) -> None:
        """Nothing installed and no config → empty list."""
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
        ):
            servers = load_mcp_servers()

        assert servers == []


# ── stacks.d container manifests ──────────────────────────────────────────────


def _make_manifest(stacks_dir: Path, name: str, body: str) -> Path:
    """Write a ``stacks.d/<name>.toml`` manifest with *body* and return its path."""
    stacks_dir.mkdir(parents=True, exist_ok=True)
    path = stacks_dir / f"{name}.toml"
    path.write_text(body)
    return path


class TestStacksDDiscovery:
    """load_mcp_servers reads stacks.d/*.toml container manifests."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test so patches take effect."""
        _clear_cache()

    def test_manifest_discovered(self, tmp_path: Path) -> None:
        """A stacks.d manifest yields a docker-launched server entry."""
        stacks = tmp_path / "stacks.d"
        _make_manifest(
            stacks,
            "medmcp-dicom",
            'name = "medmcp-dicom"\ncommand = "docker"\n'
            'args = ["run", "--rm", "-i", "ghcr.io/medmcp/dicom:dev"]\n',
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
        ):
            servers = load_mcp_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "medmcp-dicom"
        assert servers[0]["command"] == "docker"
        assert servers[0]["args"][-1] == "ghcr.io/medmcp/dicom:dev"

    def test_env_var_expansion(self, tmp_path: Path) -> None:
        """${VAR} references in args are expanded against the environment."""
        stacks = tmp_path / "stacks.d"
        _make_manifest(
            stacks,
            "medmcp-dicom",
            'name = "medmcp-dicom"\ncommand = "docker"\n'
            'args = ["run", "-v", "${MEDMCP_WORKSPACE}:${MEDMCP_WORKSPACE}", "img"]\n',
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
            patch.dict(os.environ, {"MEDMCP_WORKSPACE": "/srv/data"}),
        ):
            servers = load_mcp_servers()

        assert "/srv/data:/srv/data" in servers[0]["args"]

    def test_skills_path_and_timeout_passthrough(self, tmp_path: Path) -> None:
        """Optional skills_path (expanded) and tool_timeout_sec are carried through."""
        stacks = tmp_path / "stacks.d"
        _make_manifest(
            stacks,
            "medmcp-neuro",
            'name = "medmcp-neuro"\ncommand = "docker"\nargs = ["run", "img"]\n'
            'skills_path = "${MEDMCP_WORKSPACE}/skills"\ntool_timeout_sec = 7200.0\n',
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
            patch.dict(os.environ, {"MEDMCP_WORKSPACE": "/srv/data"}),
        ):
            servers = load_mcp_servers()

        assert servers[0]["skills_path"] == "/srv/data/skills"
        assert servers[0]["tool_timeout_sec"] == 7200.0

    def test_startup_timeout_passthrough(self, tmp_path: Path) -> None:
        """A manifest's startup_timeout_sec reaches the server config.

        Without it the entry falls back to vibe's 10s default, which a container
        stack cold-starts past — dropping the whole stack for the session.
        """
        stacks = tmp_path / "stacks.d"
        _make_manifest(
            stacks,
            "medmcp-neuro",
            'name = "medmcp-neuro"\ncommand = "docker"\nargs = ["run", "img"]\n'
            "startup_timeout_sec = 180.0\n",
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
        ):
            servers = load_mcp_servers()

        assert servers[0]["startup_timeout_sec"] == 180.0

    def test_uv_tool_wins_over_manifest(self, tmp_path: Path) -> None:
        """A uv-tool install overrides a manifest of the same name (local dev)."""
        _make_tool_env(tmp_path, "medmcp-neuro")
        ep_config = {"name": "medmcp-neuro", "command": "medmcp-neuro"}
        stacks = tmp_path / "stacks.d"
        _make_manifest(
            stacks,
            "medmcp-neuro",
            'name = "medmcp-neuro"\ncommand = "docker"\nargs = ["run", "img"]\n',
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.settings.call_entry_point", return_value=ep_config),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
        ):
            servers = load_mcp_servers()

        assert len(servers) == 1
        # the uv-tool absolute path wins, not the docker command
        assert servers[0]["command"] != "docker"

    def test_manifest_wins_over_config_toml(self, tmp_path: Path) -> None:
        """A manifest claims a name before the config.toml fallback can."""
        stacks = tmp_path / "stacks.d"
        _make_manifest(
            stacks,
            "medmcp-neuro",
            'name = "medmcp-neuro"\ncommand = "docker"\nargs = ["run", "img"]\n',
        )
        cfg = tmp_path / "config.toml"
        cfg.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "medmcp-neuro"\n')
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
        ):
            servers = load_mcp_servers()

        assert len(servers) == 1
        assert servers[0]["command"] == "docker"

    def test_malformed_manifest_skipped(self, tmp_path: Path) -> None:
        """A manifest that fails to parse is skipped; a valid one still loads."""
        stacks = tmp_path / "stacks.d"
        _make_manifest(stacks, "broken", "this is = not valid = toml = [[[")
        _make_manifest(
            stacks,
            "medmcp-dicom",
            'name = "medmcp-dicom"\ncommand = "docker"\nargs = ["run", "img"]\n',
        )
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
        ):
            servers = load_mcp_servers()

        names = {s["name"] for s in servers}
        assert names == {"medmcp-dicom"}

    def test_manifest_missing_required_fields_skipped(self, tmp_path: Path) -> None:
        """A manifest without name or command is ignored."""
        stacks = tmp_path / "stacks.d"
        _make_manifest(stacks, "noname", 'command = "docker"\nargs = ["run"]\n')
        _make_manifest(stacks, "nocommand", 'name = "x"\nargs = ["run"]\n')
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", stacks),
        ):
            servers = load_mcp_servers()

        assert servers == []

    def test_no_stacks_dir_is_noop(self, tmp_path: Path) -> None:
        """A missing stacks.d directory simply contributes nothing."""
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.STACKS_D_PATH", tmp_path / "does-not-exist"),
        ):
            servers = load_mcp_servers()

        assert servers == []
