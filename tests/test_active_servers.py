"""Tests for active-server tracking and config.toml sync (issue 2)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

from medmcp.app import _workflow_commands  # pyright: ignore[reportPrivateUsage]
from medmcp.settings import (
    active_servers,
    discover_workflows,
    load_active_server_names,
    load_mcp_servers,
    load_provenance_enabled,
    load_workflows_enabled,
    save_active_server_names,
    save_provenance_enabled,
    save_workflows_enabled,
    sync_servers_to_vibe_config,
)

JsonDict = dict[str, Any]

_TWO_SERVERS: list[JsonDict] = [
    {"name": "medmcp-neuro", "command": "uvx", "args": ["medmcp-neuro"], "env": []},
    {"name": "medmcp-cardiac", "command": "uvx", "args": ["medmcp-cardiac"], "env": []},
]


def _clear_cache() -> None:
    load_mcp_servers.cache_clear()


# ── load_active_server_names ─────────────────────────────────────────────────


class TestLoadActiveServerNames:
    """load_active_server_names falls back to all-active when file is absent."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        _clear_cache()

    def test_no_file_returns_all_discovered(self, tmp_path: Path) -> None:
        """Without active_stacks.json every discovered server is active."""
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json"),
        ):
            # Two servers in config.toml, no active_stacks.json → both active.
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            names = load_active_server_names()

        assert names == {"medmcp-neuro", "medmcp-cardiac"}

    def test_file_with_subset_returns_subset(self, tmp_path: Path) -> None:
        """Only the names listed in active_stacks.json are returned."""
        stacks_file = tmp_path / "active_stacks.json"
        stacks_file.write_text(json.dumps({"active": ["medmcp-neuro"]}))

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_STACKS_PATH", stacks_file),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            names = load_active_server_names()

        assert names == {"medmcp-neuro"}

    def test_corrupt_file_falls_back_to_all(self, tmp_path: Path) -> None:
        """A JSON-corrupt active_stacks.json is silently ignored."""
        stacks_file = tmp_path / "active_stacks.json"
        stacks_file.write_text("not valid json{{{")

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_STACKS_PATH", stacks_file),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n')
            names = load_active_server_names()

        assert "medmcp-neuro" in names


# ── save_active_server_names ─────────────────────────────────────────────────


class TestSaveActiveServerNames:
    """save_active_server_names writes a round-trippable JSON file."""

    def test_writes_sorted_json(self, tmp_path: Path) -> None:
        """Names are stored as a sorted list under the 'active' key."""
        path = tmp_path / "active_stacks.json"
        with patch("medmcp.settings.ACTIVE_STACKS_PATH", path):
            save_active_server_names({"medmcp-cardiac", "medmcp-neuro"})

        data = json.loads(path.read_text())
        assert data == {"active": ["medmcp-cardiac", "medmcp-neuro"]}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        path = tmp_path / "nested" / "dir" / "active_stacks.json"
        with patch("medmcp.settings.ACTIVE_STACKS_PATH", path):
            save_active_server_names({"medmcp-neuro"})

        assert path.exists()


# ── provenance enabled preference ─────────────────────────────────────────────


class TestProvenanceEnabled:
    """_load/save_provenance_enabled round-trip and default to on."""

    def test_defaults_to_true_when_absent(self, tmp_path: Path) -> None:
        """Without the file, provenance is enabled by default."""
        with patch("medmcp.settings.PROVENANCE_ENABLED_PATH", tmp_path / "provenance_enabled.json"):
            assert load_provenance_enabled() is True

    def test_round_trip_false(self, tmp_path: Path) -> None:
        """A saved disabled preference reads back as False."""
        path = tmp_path / "provenance_enabled.json"
        with patch("medmcp.settings.PROVENANCE_ENABLED_PATH", path):
            save_provenance_enabled(False)
            assert load_provenance_enabled() is False
        assert json.loads(path.read_text()) == {"enabled": False}

    def test_corrupt_file_defaults_to_true(self, tmp_path: Path) -> None:
        """A corrupt preference file falls back to enabled."""
        path = tmp_path / "provenance_enabled.json"
        path.write_text("not json{{{")
        with patch("medmcp.settings.PROVENANCE_ENABLED_PATH", path):
            assert load_provenance_enabled() is True


# ── personal-workflows master toggle ──────────────────────────────────────────


class TestWorkflowsEnabled:
    """_load/save_workflows_enabled round-trip and default to on."""

    def test_defaults_to_true_when_absent(self, tmp_path: Path) -> None:
        """Without the file, the workflows feature is enabled by default."""
        with patch("medmcp.settings.WORKFLOWS_ENABLED_PATH", tmp_path / "workflows_enabled.json"):
            assert load_workflows_enabled() is True

    def test_round_trip_false(self, tmp_path: Path) -> None:
        """A saved disabled preference reads back as False."""
        path = tmp_path / "workflows_enabled.json"
        with patch("medmcp.settings.WORKFLOWS_ENABLED_PATH", path):
            save_workflows_enabled(False)
            assert load_workflows_enabled() is False
        assert json.loads(path.read_text()) == {"enabled": False}

    def test_corrupt_file_defaults_to_true(self, tmp_path: Path) -> None:
        """A corrupt preference file falls back to enabled."""
        path = tmp_path / "workflows_enabled.json"
        path.write_text("not json{{{")
        with patch("medmcp.settings.WORKFLOWS_ENABLED_PATH", path):
            assert load_workflows_enabled() is True


class TestWorkflowCommands:
    """_workflow_commands gates the composer buttons on the master toggle."""

    def test_returns_buttons_when_enabled(self, tmp_path: Path) -> None:
        """With the feature on, both Save and Manage buttons are offered."""
        with patch("medmcp.settings.WORKFLOWS_ENABLED_PATH", tmp_path / "workflows_enabled.json"):
            ids = {cmd["id"] for cmd in _workflow_commands()}
        assert ids == {"save-workflow", "manage-workflows"}

    def test_empty_when_disabled(self, tmp_path: Path) -> None:
        """With the feature off, no composer buttons are rendered."""
        path = tmp_path / "workflows_enabled.json"
        with patch("medmcp.settings.WORKFLOWS_ENABLED_PATH", path):
            save_workflows_enabled(False)
            assert _workflow_commands() == []


class TestWorkflowSkillPathsGating:
    """sync_servers_to_vibe_config honors the workflows master toggle."""

    def test_enabled_adds_skill_paths(self, tmp_path: Path) -> None:
        """With the feature on, workflow dirs are added to skill_paths."""
        _make_workflow(tmp_path, "active", "wf-promoted")
        _make_workflow(tmp_path, "draft", "wf-draft")
        cfg_path = tmp_path / "config.toml"
        with (
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.WORKFLOWS_ENABLED_PATH", tmp_path / "workflows_enabled.json"),
            patch("medmcp.settings.ACTIVE_WORKFLOWS_PATH", tmp_path / "active_workflows.json"),
        ):
            sync_servers_to_vibe_config([])
        cfg = tomllib.loads(cfg_path.read_text())
        skill_paths = cfg["skill_paths"]
        assert str(tmp_path / "workflows" / "active") in skill_paths
        assert str(tmp_path / "workflows" / "draft") in skill_paths
        # All discovered workflows are active by default, so none disabled.
        assert cfg["disabled_skills"] == []

    def test_disabled_drops_skill_paths_and_disables_all(self, tmp_path: Path) -> None:
        """With the feature off, no workflow dir loads and all are disabled."""
        _make_workflow(tmp_path, "active", "wf-promoted")
        _make_workflow(tmp_path, "draft", "wf-draft")
        cfg_path = tmp_path / "config.toml"
        with (
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.WORKFLOWS_ENABLED_PATH", tmp_path / "workflows_enabled.json"),
            patch("medmcp.settings.ACTIVE_WORKFLOWS_PATH", tmp_path / "active_workflows.json"),
        ):
            save_workflows_enabled(False)
            sync_servers_to_vibe_config([])
        cfg = tomllib.loads(cfg_path.read_text())
        skill_paths = cfg["skill_paths"]
        assert str(tmp_path / "workflows" / "active") not in skill_paths
        assert str(tmp_path / "workflows" / "draft") not in skill_paths
        assert cfg["disabled_skills"] == ["wf-draft", "wf-promoted"]


# ── active_servers ───────────────────────────────────────────────────────────


class TestActiveServers:
    """active_servers returns only the enabled subset."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        _clear_cache()

    def test_returns_active_subset(self, tmp_path: Path) -> None:
        """Only servers listed in active_stacks.json are returned."""
        stacks_file = tmp_path / "active_stacks.json"
        stacks_file.write_text(json.dumps({"active": ["medmcp-neuro"]}))

        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_STACKS_PATH", stacks_file),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            servers = active_servers()

        assert len(servers) == 1
        assert servers[0]["name"] == "medmcp-neuro"

    def test_all_active_when_no_file(self, tmp_path: Path) -> None:
        """Without active_stacks.json all discovered servers are returned."""
        with (
            patch("medmcp.settings.get_uv_tool_dir", return_value=None),
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_STACKS_PATH", tmp_path / "active_stacks.json"),
        ):
            cfg = tmp_path / "config.toml"
            cfg.write_text(
                '[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n'
                '[[mcp_servers]]\nname = "medmcp-cardiac"\ncommand = "uvx"\n'
            )
            servers = active_servers()

        assert {s["name"] for s in servers} == {"medmcp-neuro", "medmcp-cardiac"}


# ── sync_servers_to_vibe_config ──────────────────────────────────────────────


class TestSyncServersToVibeConfig:
    """sync_servers_to_vibe_config writes mcp_servers into config.toml."""

    def test_writes_active_servers(self, tmp_path: Path) -> None:
        """Active servers appear in the written config.toml."""
        cfg_path = tmp_path / "config.toml"
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(_TWO_SERVERS)

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        names = {s["name"] for s in result["mcp_servers"]}
        assert names == {"medmcp-neuro", "medmcp-cardiac"}

    def test_empty_list_clears_section(self, tmp_path: Path) -> None:
        """Passing an empty list removes all mcp_servers entries."""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('[[mcp_servers]]\nname = "medmcp-neuro"\ncommand = "uvx"\n')
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config([])

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
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config([new_srv])

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
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(_TWO_SERVERS)

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        assert result["active_model"] == "local"
        assert result["auto_approve"] is False

    def test_new_server_from_entry_point_gets_minimal_fields(self, tmp_path: Path) -> None:
        """A server not previously in config.toml is written with transport=stdio."""
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(
                [{"name": "medmcp-new", "command": "uvx", "args": ["medmcp-new"], "env": []}]
            )

        cfg_path = tmp_path / "config.toml"
        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        entry = result["mcp_servers"][0]
        assert entry["transport"] == "stdio"
        assert entry["name"] == "medmcp-new"

    def test_creates_config_when_absent(self, tmp_path: Path) -> None:
        """Sync creates config.toml from scratch when the file does not exist yet."""
        cfg_path = tmp_path / "config.toml"
        assert not cfg_path.exists()

        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(_TWO_SERVERS)

        assert cfg_path.exists()
        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        names = {s["name"] for s in result["mcp_servers"]}
        assert names == {"medmcp-neuro", "medmcp-cardiac"}

    def test_skill_paths_written_when_present(self, tmp_path: Path) -> None:
        """skill_paths in server dicts are collected into config.toml skill_paths."""
        servers: list[JsonDict] = [
            {
                "name": "medmcp-neuro",
                "command": "uvx",
                "args": ["medmcp-neuro"],
                "env": [],
                "skills_path": "/opt/neuro/skills",
            },
        ]
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(servers)

        with (tmp_path / "config.toml").open("rb") as f:
            result = tomllib.load(f)

        assert result["skill_paths"] == ["/opt/neuro/skills"]

    def test_skill_paths_cleared_when_stack_deactivated(self, tmp_path: Path) -> None:
        """Deactivating all stacks removes stale skill_paths from config.toml."""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('skill_paths = ["/opt/neuro/skills"]\n')

        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config([])

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        assert result.get("skill_paths") == []

    def test_active_workflows_dir_added_to_skill_paths(self, tmp_path: Path) -> None:
        """A promoted-workflows active/ dir is included in skill_paths."""
        (tmp_path / "workflows" / "active").mkdir(parents=True)
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(_TWO_SERVERS)

        with (tmp_path / "config.toml").open("rb") as f:
            result = tomllib.load(f)

        assert str(tmp_path / "workflows" / "active") in result["skill_paths"]

    def test_draft_workflows_dir_added_to_skill_paths(self, tmp_path: Path) -> None:
        """The draft/ dir is included in skill_paths so drafts can be tested."""
        (tmp_path / "workflows" / "draft").mkdir(parents=True)
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(_TWO_SERVERS)

        with (tmp_path / "config.toml").open("rb") as f:
            result = tomllib.load(f)

        assert str(tmp_path / "workflows" / "draft") in result["skill_paths"]

    def test_skill_paths_updated_when_partial_deactivation(self, tmp_path: Path) -> None:
        """skill_paths reflects only the active servers after partial deactivation."""
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text('skill_paths = ["/opt/neuro/skills", "/opt/cardiac/skills"]\n')
        active: list[JsonDict] = [
            {
                "name": "medmcp-neuro",
                "command": "uvx",
                "args": ["medmcp-neuro"],
                "env": [],
                "skills_path": "/opt/neuro/skills",
            },
        ]

        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            sync_servers_to_vibe_config(active)

        with cfg_path.open("rb") as f:
            result = tomllib.load(f)

        assert result["skill_paths"] == ["/opt/neuro/skills"]


# ── Personal workflows: discovery & disabled_skills sync ──────────────────────


def _make_workflow(root: Path, kind: str, name: str, description: str = "") -> None:
    """Create a minimal <root>/workflows/<kind>/<name>/SKILL.md."""
    d = root / "workflows" / kind / name
    d.mkdir(parents=True)
    front = f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n"
    (d / "SKILL.md").write_text(front)


class TestDiscoverWorkflows:
    """discover_workflows scans draft/ and active/ for SKILL.md entries."""

    def test_finds_active_and_draft(self, tmp_path: Path) -> None:
        """Both promoted and draft workflows are discovered with their kind."""
        _make_workflow(tmp_path, "active", "brain-mri", "skull strip")
        _make_workflow(tmp_path, "draft", "spine-seg", "segment spine")
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            found = {w["name"]: w for w in discover_workflows()}
        assert found["brain-mri"]["kind"] == "active"
        assert found["brain-mri"]["description"] == "skull strip"
        assert found["spine-seg"]["kind"] == "draft"

    def test_active_shadows_draft_of_same_name(self, tmp_path: Path) -> None:
        """A name present in both dirs resolves to the active entry."""
        _make_workflow(tmp_path, "active", "dup")
        _make_workflow(tmp_path, "draft", "dup")
        with patch("medmcp.settings.VIBE_HOME", tmp_path):
            found = [w for w in discover_workflows() if w["name"] == "dup"]
        assert len(found) == 1
        assert found[0]["kind"] == "active"


class TestWorkflowDisabledSkills:
    """Deactivated workflows are written to disabled_skills on config sync."""

    def test_deactivated_workflow_disabled(self, tmp_path: Path) -> None:
        """A workflow not in the active set is written to disabled_skills."""
        _make_workflow(tmp_path, "active", "keep-me")
        _make_workflow(tmp_path, "active", "turn-off")
        active_path = tmp_path / "active_workflows.json"
        active_path.write_text(json.dumps({"active": ["keep-me"]}))

        with (
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_WORKFLOWS_PATH", active_path),
        ):
            sync_servers_to_vibe_config([])

        with (tmp_path / "config.toml").open("rb") as f:
            result = tomllib.load(f)
        assert result["disabled_skills"] == ["turn-off"]

    def test_all_active_means_none_disabled(self, tmp_path: Path) -> None:
        """With no active_workflows.json, every workflow is active (none disabled)."""
        _make_workflow(tmp_path, "active", "a")
        _make_workflow(tmp_path, "draft", "b")
        with (
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_WORKFLOWS_PATH", tmp_path / "active_workflows.json"),
        ):
            sync_servers_to_vibe_config([])
        with (tmp_path / "config.toml").open("rb") as f:
            result = tomllib.load(f)
        assert result["disabled_skills"] == []

    def test_preserves_non_workflow_disabled_skills(self, tmp_path: Path) -> None:
        """A manually-disabled non-workflow skill survives the sync."""
        _make_workflow(tmp_path, "active", "wf-a")
        cfg = tmp_path / "config.toml"
        cfg.write_text('disabled_skills = ["some-builtin"]\n')
        active_path = tmp_path / "active_workflows.json"
        active_path.write_text(json.dumps({"active": []}))  # wf-a deactivated

        with (
            patch("medmcp.settings.VIBE_HOME", tmp_path),
            patch("medmcp.settings.ACTIVE_WORKFLOWS_PATH", active_path),
        ):
            sync_servers_to_vibe_config([])

        with cfg.open("rb") as f:
            result = tomllib.load(f)
        assert set(result["disabled_skills"]) == {"some-builtin", "wf-a"}
