"""Tests for _load_mcp_servers() — uv tool env scanning and config.toml fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from medmcp.app import _load_mcp_servers  # pyright: ignore[reportPrivateUsage]

JsonDict = dict[str, Any]


def _clear_cache() -> None:
    _load_mcp_servers.cache_clear()


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
    """_load_mcp_servers scans uv tool envs for [medmcp.stacks] entry points."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test so patches take effect."""
        _clear_cache()

    def test_single_tool_discovered(self, tmp_path: Path) -> None:
        """A stack installed as a uv tool yields one server entry."""
        _make_tool_env(tmp_path, "medmcp-test")
        config: JsonDict = {"name": "medmcp-test", "command": "medmcp-test", "args": []}

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", return_value=config),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "medmcp-test"

    def test_command_resolved_to_absolute_path(self, tmp_path: Path) -> None:
        """The command is resolved to the absolute executable inside the tool env."""
        _make_tool_env(tmp_path, "medmcp-test")
        config = {"name": "medmcp-test", "command": "medmcp-test"}

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", return_value=config),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        expected = str(tmp_path / "medmcp-test" / "bin" / "medmcp-test")
        assert servers[0]["command"] == expected

    def test_env_defaults_to_empty_list(self, tmp_path: Path) -> None:
        """Env field defaults to [] when not returned by the entry point."""
        _make_tool_env(tmp_path, "medmcp-x")
        config = {"name": "medmcp-x", "command": "medmcp-x"}

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", return_value=config),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        assert servers[0]["env"] == []

    def test_broken_entry_point_is_skipped(self, tmp_path: Path) -> None:
        """An entry point whose callable raises is skipped; others still load."""
        _make_tool_env(tmp_path, "medmcp-bad")
        _make_tool_env(tmp_path, "medmcp-ok")

        def _side_effect(python: Path, module: str, attr: str) -> object:
            if "bad" in str(python):
                raise ImportError("missing heavy dep")
            return {"name": "medmcp-ok", "command": "medmcp-ok"}

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", side_effect=_side_effect),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        names = [s["name"] for s in servers]
        assert "medmcp-ok" in names
        assert "medmcp-bad" not in names

    def test_invalid_dict_is_skipped(self, tmp_path: Path) -> None:
        """An entry point returning a dict without 'name' is ignored."""
        _make_tool_env(tmp_path, "medmcp-bad")

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", return_value={"command": "uvx"}),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        assert servers == []

    def test_malformed_ep_value_is_skipped(self, tmp_path: Path) -> None:
        """An entry_points.txt value without 'module:attr' syntax is ignored."""
        _make_tool_env(tmp_path, "medmcp-bad", ep_value="not_valid")

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        assert servers == []

    def test_no_uv_tool_dir_returns_empty(self, tmp_path: Path) -> None:
        """When uv is unavailable, discovery falls back to config.toml only."""
        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=None),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        assert servers == []

    def test_empty_command_not_resolved_to_bin_dir(self, tmp_path: Path) -> None:
        """An entry point returning command='' must not have its command set to the bin dir.

        Path / "" collapses to the parent (the bin/ dir itself), which exists,
        so without an explicit guard command would be overwritten with a directory path.
        """
        _make_tool_env(tmp_path, "medmcp-test")
        config = {"name": "medmcp-test", "command": ""}

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", return_value=config),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

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
            patch("medmcp.app._get_uv_tool_dir", return_value=None),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "medmcp-neuro"

    def test_uv_tool_wins_over_config_same_name(self, tmp_path: Path) -> None:
        """When the same name appears in both sources, the uv tool wins."""
        _make_tool_env(tmp_path, "medmcp-neuro")
        ep_config = {"name": "medmcp-neuro", "command": "medmcp-neuro"}
        cfg = tmp_path / "config.toml"
        cfg.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "/stale/path"\n')

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", return_value=ep_config),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

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
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app._call_entry_point", return_value=ep_config),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        names = {s["name"] for s in servers}
        assert names == {"medmcp-neuro", "medmcp-cardiac"}

    def test_no_tools_no_config_returns_empty(self, tmp_path: Path) -> None:
        """Nothing installed and no config → empty list."""
        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=tmp_path),
            patch("medmcp.app.VIBE_HOME", tmp_path),
        ):
            servers = _load_mcp_servers()

        assert servers == []
