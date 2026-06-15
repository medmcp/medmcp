"""Tests for the workspace server's path guard and settings-merge logic.

These two pieces carry the workspace's security-relevant invariants — the
filesystem API must never resolve outside ``WORKSPACE_ROOT``, and a settings
save must not silently deactivate entries the drawer never saw — and both are
exercisable without a live vibe-acp subprocess.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# pyright: reportPrivateUsage=false
from medmcp import server, settings

# ── _safe_path: the workspace traversal guard ────────────────────────────────


class TestSafePath:
    """``_safe_path`` resolves inside the workspace and rejects everything else."""

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the server's workspace root at a temp dir (resolved, no symlinks)."""
        resolved = tmp_path.resolve()
        monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
        return resolved

    def test_allows_nested_relative_path(self, root: Path) -> None:
        """A normal relative path resolves to its location under the root."""
        assert server._safe_path("sub/dir/scan.nii.gz") == root / "sub" / "dir" / "scan.nii.gz"

    def test_allows_root_itself(self, root: Path) -> None:
        """An empty path (and ``.``) resolve to the workspace root, not an escape."""
        assert server._safe_path("") == root
        assert server._safe_path(".") == root

    def test_inner_dotdot_that_stays_inside_is_allowed(self, root: Path) -> None:
        """``..`` segments are fine as long as the result stays in the workspace."""
        assert server._safe_path("a/../b.txt") == root / "b.txt"

    def test_rejects_parent_traversal(self, root: Path) -> None:
        """A leading ``..`` that escapes the root is refused with 400."""
        with pytest.raises(HTTPException) as exc:
            server._safe_path("../secret.txt")
        assert exc.value.status_code == 400
        assert "escapes" in str(exc.value.detail)

    def test_rejects_deep_traversal(self, root: Path) -> None:
        """A path that climbs out via many ``..`` is refused."""
        with pytest.raises(HTTPException) as exc:
            server._safe_path("a/b/../../../../etc/passwd")
        assert exc.value.status_code == 400

    def test_rejects_absolute_path(self, root: Path) -> None:
        """An absolute path is rejected before any resolution."""
        with pytest.raises(HTTPException) as exc:
            server._safe_path("/etc/passwd")
        assert exc.value.status_code == 400
        assert "absolute" in str(exc.value.detail)

    def test_rejects_sibling_with_shared_prefix(self, root: Path) -> None:
        """A sibling dir whose name merely starts with the root's name is outside.

        Guards against a string-prefix check; ``is_relative_to`` compares path
        components, so ``<root>_evil`` must not be treated as inside ``<root>``.
        """
        sibling = root.parent / (root.name + "_evil")
        sibling.mkdir()
        with pytest.raises(HTTPException) as exc:
            server._safe_path(f"../{sibling.name}/loot.txt")
        assert exc.value.status_code == 400

    def test_rejects_symlink_escape(self, root: Path) -> None:
        """A symlink inside the workspace pointing out is followed and rejected."""
        outside = root.parent / "outside"
        outside.mkdir()
        (root / "link").symlink_to(outside)
        with pytest.raises(HTTPException) as exc:
            server._safe_path("link/secret.txt")
        assert exc.value.status_code == 400
        assert "escapes" in str(exc.value.detail)


# ── PUT /api/settings: the merge-not-overwrite logic ─────────────────────────


class TestSettingsMerge:
    """The settings PUT merges the drawer's known lists with the current state.

    The drawer submits every entry it knew about (name + active). Entries it
    never saw (e.g. a workflow distilled while the drawer was open) must keep
    their current active state rather than being dropped.
    """

    @pytest.fixture
    def harness(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, set[str]]:
        """Stub settings persistence and the vibe restart; capture what's saved.

        ``old`` seeds the current active sets; ``saved`` records the merged sets
        the endpoint persists. The vibe-acp restart and config sync are stubbed
        so the endpoint can run without a subprocess.
        """
        old_stacks = {"alpha", "beta"}
        old_workflows = {"wf-keep"}
        saved: dict[str, set[str]] = {}

        # Strict pyright rejects untyped lambdas, so the stubs are typed defs.
        def _save_stacks(names: Iterable[str]) -> None:
            saved["stacks"] = set(names)

        def _save_workflows(names: Iterable[str]) -> None:
            saved["workflows"] = set(names)

        def _noop_bool(_value: bool) -> None:
            return None

        def _noop_sync(_servers: object) -> None:
            return None

        def _active_servers() -> list[object]:
            return []

        async def _no_restart() -> None:
            return None

        monkeypatch.setattr(settings, "load_active_server_names", lambda: set(old_stacks))
        monkeypatch.setattr(settings, "load_active_workflow_names", lambda: set(old_workflows))
        monkeypatch.setattr(settings, "load_workflows_enabled", lambda: True)
        monkeypatch.setattr(settings, "save_explain_enabled", _noop_bool)
        monkeypatch.setattr(settings, "save_provenance_enabled", _noop_bool)
        monkeypatch.setattr(settings, "save_workflows_enabled", _noop_bool)
        monkeypatch.setattr(settings, "save_active_server_names", _save_stacks)
        monkeypatch.setattr(settings, "save_active_workflow_names", _save_workflows)
        monkeypatch.setattr(settings, "sync_servers_to_vibe_config", _noop_sync)
        monkeypatch.setattr(settings, "active_servers", _active_servers)
        monkeypatch.setattr(server, "_restart_vibe", _no_restart)
        return saved

    @staticmethod
    def _put(client: TestClient, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "explain_tools": True,
            "record_provenance": False,
            "workflows_enabled": True,
            "stacks": [],
            "workflows": [],
        }
        body.update(overrides)
        resp = client.put("/api/settings", json=body)
        assert resp.status_code == 200
        return resp.json()

    def test_preserves_entry_unknown_to_the_drawer(self, harness: dict[str, set[str]]) -> None:
        """A stack the drawer never listed stays active after a save."""
        client = TestClient(server.app)
        result = self._put(client, stacks=[{"name": "alpha", "active": True}])
        # "beta" was unknown to this drawer payload, so it is kept; nothing changed.
        assert harness["stacks"] == {"alpha", "beta"}
        assert result["restarted"] is False

    def test_deactivates_a_known_entry(self, harness: dict[str, set[str]]) -> None:
        """Turning a listed stack off removes it and triggers a restart."""
        client = TestClient(server.app)
        result = self._put(
            client,
            stacks=[{"name": "alpha", "active": False}, {"name": "beta", "active": True}],
        )
        assert harness["stacks"] == {"beta"}
        assert result["restarted"] is True

    def test_workflows_master_toggle_triggers_restart(self, harness: dict[str, set[str]]) -> None:
        """Flipping the workflows master switch restarts vibe even if sets match."""
        client = TestClient(server.app)
        result = self._put(
            client,
            workflows_enabled=False,
            stacks=[{"name": "alpha", "active": True}, {"name": "beta", "active": True}],
            workflows=[{"name": "wf-keep", "active": True}],
        )
        assert result["restarted"] is True

    def test_no_restart_when_nothing_changes(self, harness: dict[str, set[str]]) -> None:
        """Re-submitting the current state is a no-op restart-wise."""
        client = TestClient(server.app)
        result = self._put(
            client,
            stacks=[{"name": "alpha", "active": True}, {"name": "beta", "active": True}],
            workflows=[{"name": "wf-keep", "active": True}],
        )
        assert harness["stacks"] == {"alpha", "beta"}
        assert harness["workflows"] == {"wf-keep"}
        assert result["restarted"] is False
