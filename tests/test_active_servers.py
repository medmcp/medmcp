"""Tests for active-server tracking and config.toml sync (issue 2)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

from medmcp.app import (
    _active_servers,  # pyright: ignore[reportPrivateUsage]
    _load_active_server_names,  # pyright: ignore[reportPrivateUsage]
    _load_mcp_servers,  # pyright: ignore[reportPrivateUsage]
    _save_active_server_names,  # pyright: ignore[reportPrivateUsage]
    _sync_servers_to_vibe_config,  # pyright: ignore[reportPrivateUsage]
)

JsonDict = dict[str, Any]

_TWO_SERVERS: list[JsonDict] = [
    {"name": "medmcp-neuro", "command": "uvx", "args": ["medmcp-neuro"], "env": []},
    {"name": "medmcp-cardiac", "command": "uvx", "args": ["medmcp-cardiac"], "env": []},
]


def _clear_cache() -> None:
    _load_mcp_servers.cache_clear()


# ── _load_active_server_names ─────────────────────────────────────────────────


class TestLoadActiveServerNames:
    """_load_active_server_names falls back to all-active when file is absent."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        _clear_cache()

    def test_no_file_returns_all_discovered(self, tmp_path: Path) -> None:
        """Without active_stacks.json every discovered server is active."""
        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=None),
            patch("medmcp.app.VIBE_HOME", tmp_path),
            patch("medmcp.app._ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json"),
        ):
            # Two servers in config.toml, no active_stacks.json → both active.
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            names = _load_active_server_names()

        assert names == {"medmcp-neuro", "medmcp-cardiac"}

    def test_file_with_subset_returns_subset(self, tmp_path: Path) -> None:
        """Only the names listed in active_stacks.json are returned."""
        stacks_file = tmp_path / "active_stacks.json"
        stacks_file.write_text(json.dumps({"active": ["medmcp-neuro"]}))

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=None),
            patch("medmcp.app.VIBE_HOME", tmp_path),
            patch("medmcp.app._ACTIVE_STACKS_PATH", stacks_file),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            names = _load_active_server_names()

        assert names == {"medmcp-neuro"}

    def test_corrupt_file_falls_back_to_all(self, tmp_path: Path) -> None:
        """A JSON-corrupt active_stacks.json is silently ignored."""
        stacks_file = tmp_path / "active_stacks.json"
        stacks_file.write_text("not valid json{{{")

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=None),
            patch("medmcp.app.VIBE_HOME", tmp_path),
            patch("medmcp.app._ACTIVE_STACKS_PATH", stacks_file),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n')
            names = _load_active_server_names()

        assert "medmcp-neuro" in names


# ── _save_active_server_names ─────────────────────────────────────────────────


class TestSaveActiveServerNames:
    """_save_active_server_names writes a round-trippable JSON file."""

    def test_writes_sorted_json(self, tmp_path: Path) -> None:
        """Names are stored as a sorted list under the 'active' key."""
        path = tmp_path / "active_stacks.json"
        with patch("medmcp.app._ACTIVE_STACKS_PATH", path):
            _save_active_server_names({"medmcp-cardiac", "medmcp-neuro"})

        data = json.loads(path.read_text())
        assert data == {"active": ["medmcp-cardiac", "medmcp-neuro"]}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        path = tmp_path / "nested" / "dir" / "active_stacks.json"
        with patch("medmcp.app._ACTIVE_STACKS_PATH", path):
            _save_active_server_names({"medmcp-neuro"})

        assert path.exists()


# ── _active_servers ───────────────────────────────────────────────────────────


class TestActiveServers:
    """_active_servers returns only the enabled subset."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        _clear_cache()

    def test_returns_active_subset(self, tmp_path: Path) -> None:
        """Only servers listed in active_stacks.json are returned."""
        stacks_file = tmp_path / "active_stacks.json"
        stacks_file.write_text(json.dumps({"active": ["medmcp-neuro"]}))

        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=None),
            patch("medmcp.app.VIBE_HOME", tmp_path),
            patch("medmcp.app._ACTIVE_STACKS_PATH", stacks_file),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            servers = _active_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "medmcp-neuro"

    def test_all_active_when_no_file(self, tmp_path: Path) -> None:
        """Without active_stacks.json all discovered servers are returned."""
        with (
            patch("medmcp.app._get_uv_tool_dir", return_value=None),
            patch("medmcp.app.VIBE_HOME", tmp_path),
            patch("medmcp.app._ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json"),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            servers = _active_servers()

        assert {s["name"] for s in servers} == {"medmcp-neuro", "medmcp-cardiac"}


# ── _sync_servers_to_vibe_config ──────────────────────────────────────────────


class TestSyncServersToVibeConfig:
    """_sync_servers_to_vibe_config writes mcp_servers into config.toml."""

    def test_writes_active_servers(self, tmp_path: Path) -> None:
        """Active servers appear in the written config.toml."""
        cfg_path = tmp_path / "config.toml"
        with patch("medmcp.app.VIBE_HOME", tmp_path):
            _sync_servers_to_vibe_config(_TWO_SERVERS)

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        names = {s["name"] for s in result["mcp_servers"]}
        assert names == {"medmcp-neuro", "medmcp-cardiac"}

    def test_empty_list_clears_section(self, tmp_path: Path) -> None:
        """Passing an empty list removes all mcp_servers entries."""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n')
        with patch("medmcp.app.VIBE_HOME", tmp_path):
            _sync_servers_to_vibe_config([])

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        assert result.get("mcp_servers", []) == []

    def test_preserves_existing_timeout_fields(self, tmp_path: Path) -> None:
        """startup_timeout_sec and tool_timeout_sec survive a rewrite."""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(
            "[[mcp_servers]]\n"
            'name = "medmcp-neuro"\n'
            'command = "medmcp-neuro"\n'
            "startup_timeout_sec = 30.0\n"
            "tool_timeout_sec = 3600.0\n"
        )
        new_srv: JsonDict = {
            "name": "medmcp-neuro",
            "command": "uvx",
            "args": ["medmcp-neuro"],
            "env": [],
        }
        with patch("medmcp.app.VIBE_HOME", tmp_path):
            _sync_servers_to_vibe_config([new_srv])

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        entry = result["mcp_servers"][0]
        assert entry["command"] == "uvx"
        assert entry["startup_timeout_sec"] == 30.0
        assert entry["tool_timeout_sec"] == 3600.0

    def test_preserves_other_config_keys(self, tmp_path: Path) -> None:
        """Non-mcp_servers settings in config.toml are not disturbed."""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('active_model = "local"\nauto_approve = false\n')
        with patch("medmcp.app.VIBE_HOME", tmp_path):
            _sync_servers_to_vibe_config(_TWO_SERVERS)

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        assert result["active_model"] == "local"
        assert result["auto_approve"] is False

    def test_new_server_from_entry_point_gets_minimal_fields(self, tmp_path: Path) -> None:
        """A server not previously in config.toml is written with transport=stdio."""
        with patch("medmcp.app.VIBE_HOME", tmp_path):
            _sync_servers_to_vibe_config(
                [{"name": "medmcp-new", "command": "uvx", "args": ["medmcp-new"], "env": []}]
            )

        cfg_path = tmp_path / "config.toml"
        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        entry = result["mcp_servers"][0]
        assert entry["transport"] == "stdio"
        assert entry["name"] == "medmcp-new"
