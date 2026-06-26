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
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

# pyright: reportPrivateUsage=false
from medmcp import provenance, server, sessions, settings

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


# ── _resolve_input_path: workspace-relative replay inputs → absolute ──────────


class TestResolveInputPath:
    """Replay inputs arrive workspace-relative; the stack tools need absolute paths.

    The Run form (drags carry explorer tree ids) hands the server paths relative
    to ``WORKSPACE_ROOT``, but replay calls the stack tools directly and they
    resolve paths on disk. Existing relative paths must be lifted to absolute;
    everything else (already-absolute, non-path args, missing files) is left as-is
    so non-path inputs pass through and a missing scan still yields a clear error.
    """

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the server's workspace root at a temp dir (resolved, no symlinks)."""
        resolved = tmp_path.resolve()
        monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
        return resolved

    def test_relative_existing_file_becomes_absolute(self, root: Path) -> None:
        """A relative path naming an existing file resolves to its absolute path."""
        scan = root / "patient_01" / "visit_02" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        assert server._resolve_input_path("patient_01/visit_02/t1n_3d.nii.gz") == str(scan)

    def test_relative_existing_dir_becomes_absolute(self, root: Path) -> None:
        """Directory inputs resolve too (some tools take a folder)."""
        visit = root / "patient_01" / "visit_02"
        visit.mkdir(parents=True)
        assert server._resolve_input_path("patient_01/visit_02") == str(visit)

    def test_absolute_path_is_left_unchanged(self, root: Path) -> None:
        """An already-absolute path (the recorded example) passes through verbatim."""
        scan = root / "patient_01" / "visit_01" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        assert server._resolve_input_path(str(scan)) == str(scan)

    def test_non_path_argument_is_left_unchanged(self, root: Path) -> None:
        """A non-path argument like a device string is not mistaken for a file."""
        assert server._resolve_input_path("cuda") == "cuda"

    def test_missing_relative_path_is_left_unchanged(self, root: Path) -> None:
        """A relative path that doesn't exist is left as-is for a clear tool error."""
        assert (
            server._resolve_input_path("patient_01/visit_09/missing.nii.gz")
            == "patient_01/visit_09/missing.nii.gz"
        )

    def test_empty_value_is_left_unchanged(self, root: Path) -> None:
        """An empty value is returned untouched (not joined to the root)."""
        assert server._resolve_input_path("") == ""

    def test_resolve_input_paths_maps_every_value(self, root: Path) -> None:
        """The dict helper resolves each binding value independently."""
        scan = root / "patient_01" / "visit_02" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        resolved = server._resolve_input_paths(
            {"in_1": "patient_01/visit_02/t1n_3d.nii.gz", "in_2": "cuda"}
        )
        assert resolved == {"in_1": str(scan), "in_2": "cuda"}


# ── _workspace_note: the viewer-context note handed to the agent ─────────────


class TestWorkspaceNote:
    """The viewer-context note carries the absolute path so the agent calls tools right.

    The viewer reports a workspace-relative path, but the stack tools run in
    sibling containers and resolve paths on disk. The note must hand the agent the
    absolute path so its *first* tool call hits — without it the agent passes the
    relative path, the tool reports "not found", and it only recovers after a
    filesystem search. The note must also still be strippable from transcripts.
    """

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the server's workspace root at a temp dir (resolved, no symlinks)."""
        resolved = tmp_path.resolve()
        monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
        return resolved

    def test_note_carries_absolute_path(self, root: Path) -> None:
        """A relative viewer path is rendered as its absolute location in the note."""
        scan = root / "patient_01" / "visit_01" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        note = server._workspace_note("patient_01/visit_01/t1n_3d.nii.gz")
        assert str(scan) in note
        assert "patient_01/visit_01/t1n_3d.nii.gz" not in note.replace(str(scan), "")

    def test_note_is_strippable(self, root: Path) -> None:
        """The note appends cleanly and `_strip_workspace_note` removes it again."""
        scan = root / "patient_01" / "visit_01" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        prompt = "skull strip this image" + server._workspace_note(
            "patient_01/visit_01/t1n_3d.nii.gz"
        )
        assert server._strip_workspace_note(prompt) == "skull strip this image"


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


# ── Workflow export / import endpoints ───────────────────────────────────────


class TestWorkflowShareEndpoints:
    """Export/import workflow endpoints: wiring + error mapping."""

    @pytest.fixture
    def vibe_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point both the server and provenance/share VIBE_HOME at a temp dir."""
        monkeypatch.setattr(server, "VIBE_HOME", tmp_path)
        monkeypatch.setattr(provenance, "VIBE_HOME", tmp_path)
        return tmp_path

    def _envelope(self) -> str:
        return yaml.safe_dump(
            {
                "medmcp_workflow": 1,
                "name": "demo-flow",
                "description": "demo",
                "inputs": [{"name": "in_1", "example": "/d/a.nii.gz"}],
                "steps": [
                    {
                        "server": "medmcp-neuro",
                        "tool": "skull_strip",
                        "arguments": {"input_path": "{{in_1}}"},
                    }
                ],
            }
        )

    def test_import_then_export_round_trip(self, vibe_home: Path) -> None:
        """A valid envelope imports as a draft and exports back as a YAML download."""
        client = TestClient(server.app)
        resp = client.post("/api/workflows/import", json={"content": self._envelope()})
        assert resp.status_code == 200
        assert resp.json()["name"] == "demo-flow"

        exp = client.get("/api/workflows/demo-flow/export")
        assert exp.status_code == 200
        assert "medmcp_workflow" in exp.text
        assert exp.headers["content-disposition"].endswith('demo-flow.workflow.yaml"')

    def test_import_malformed_returns_400(self, vibe_home: Path) -> None:
        """A payload that isn't a workflow envelope maps to HTTP 400."""
        client = TestClient(server.app)
        resp = client.post("/api/workflows/import", json={"content": "- not a mapping"})
        assert resp.status_code == 400

    def test_export_missing_returns_404(self, vibe_home: Path) -> None:
        """Exporting an unknown workflow maps to HTTP 404."""
        client = TestClient(server.app)
        assert client.get("/api/workflows/nope/export").status_code == 404


# ── Session resume helpers ───────────────────────────────────────────────────


class TestStripWorkspaceNote:
    """The viewer-context note is removed from replayed user messages."""

    def test_strips_appended_note(self) -> None:
        """The trailing ``[workspace context: …]`` block is removed."""
        text = (
            "segment this scan\n\n"
            '[workspace context: the file "a/b.nii.gz" is currently open in the '
            'viewer; references like "this image" or "the current image" mean that file]'
        )
        assert server._strip_workspace_note(text) == "segment this scan"

    def test_leaves_plain_message_untouched(self) -> None:
        """A message without the note is returned verbatim."""
        assert server._strip_workspace_note("just a question") == "just a question"

    def test_only_strips_the_trailing_note(self) -> None:
        """A bracketed mention that isn't the appended note is left alone."""
        text = "see [workspace context: x] mid-sentence"
        assert server._strip_workspace_note(text) == text

    def test_strips_truncated_note(self) -> None:
        """A note cut off mid-way (no closing ``]``, as in a title) is removed."""
        text = 'Hi\n\n[workspace context: the file "test_data/2014…'
        assert server._strip_workspace_note(text) == "Hi"


class TestSessionsApi:
    """GET /api/sessions maps vibe-acp's session/list to the UI shape."""

    @staticmethod
    def _stub_client(monkeypatch: pytest.MonkeyPatch, response: dict[str, object]) -> None:
        async def _ensure() -> None:
            return None

        async def _request(method: str, params: dict[str, object]) -> dict[str, object]:
            assert method == "session/list"
            return response

        monkeypatch.setattr(server._client, "ensure_started", _ensure)
        monkeypatch.setattr(server._client, "request", _request)

    def test_maps_and_drops_idless_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each session is reshaped to {id,title,updatedAt}; idless rows are dropped."""
        self._stub_client(
            monkeypatch,
            {
                "result": {
                    "sessions": [
                        {
                            "sessionId": "s1",
                            "title": "Skull strip",
                            "updatedAt": "2026-06-15T10:00:00Z",
                        },
                        {"sessionId": "s2", "title": None, "updatedAt": None},
                        {
                            "sessionId": "s3",
                            "title": 'Hi\n\n[workspace context: the file "scan…',
                            "updatedAt": None,
                        },
                        {"title": "no id — dropped"},
                    ]
                }
            },
        )
        client = TestClient(server.app)
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert [s["id"] for s in sessions] == ["s1", "s2", "s3"]
        assert sessions[0]["title"] == "Skull strip"
        assert sessions[0]["updatedAt"] == "2026-06-15T10:00:00Z"
        # The leaked viewer-context note is stripped from the title, even truncated.
        assert sessions[2]["title"] == "Hi"

    def test_vibe_error_maps_to_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A vibe-acp error surfaces as a 502, not a 500."""
        self._stub_client(monkeypatch, {"error": {"message": "boom"}})
        client = TestClient(server.app)
        resp = client.get("/api/sessions")
        assert resp.status_code == 502

    def test_overlays_registry_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Title overrides, the archived flag, and provenance presence are merged in."""
        self._stub_client(
            monkeypatch,
            {
                "result": {
                    "sessions": [
                        {"sessionId": "s1", "title": "raw vibe title", "updatedAt": "t1"},
                        {"sessionId": "s2", "title": "kept", "updatedAt": "t2"},
                    ]
                }
            },
        )

        def _registry() -> dict[str, dict[str, object]]:
            return {"s1": {"title": "My name"}, "s2": {"archived": True}}

        (tmp_path / "s1").mkdir()  # s1 has a provenance record; s2 does not

        def _prov_dir(session_id: str) -> Path:
            return tmp_path / session_id

        monkeypatch.setattr(sessions, "load_registry", _registry)
        monkeypatch.setattr(provenance, "provenance_dir", _prov_dir)

        client = TestClient(server.app)
        rows = {s["id"]: s for s in client.get("/api/sessions").json()["sessions"]}
        assert rows["s1"]["title"] == "My name"  # override wins over vibe's title
        assert rows["s1"]["archived"] is False
        assert rows["s1"]["hasProvenance"] is True
        assert rows["s2"]["title"] == "kept"
        assert rows["s2"]["archived"] is True
        assert rows["s2"]["hasProvenance"] is False

    def test_rename_sets_title(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """POST …/rename forwards to the registry's set_title."""
        captured: dict[str, tuple[str, str]] = {}

        def _set_title(session_id: str, title: str) -> None:
            captured["rename"] = (session_id, title)

        monkeypatch.setattr(sessions, "set_title", _set_title)
        client = TestClient(server.app)
        resp = client.post("/api/sessions/abc/rename", json={"title": "New name"})
        assert resp.status_code == 200
        assert captured["rename"] == ("abc", "New name")

    def test_archive_sets_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """POST …/archive forwards to the registry's set_archived."""
        captured: dict[str, tuple[str, bool]] = {}

        def _set_archived(session_id: str, archived: bool) -> None:
            captured["archive"] = (session_id, archived)

        monkeypatch.setattr(sessions, "set_archived", _set_archived)
        client = TestClient(server.app)
        resp = client.post("/api/sessions/abc/archive", json={"archived": True})
        assert resp.status_code == 200
        assert captured["archive"] == ("abc", True)

    def test_delete_purges_then_forgets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DELETE …/{id} purges the transcript/provenance and drops the registry entry."""
        calls: list[tuple[str, str]] = []

        def _purge(session_id: str) -> None:
            calls.append(("purge", session_id))

        def _remove(session_id: str) -> None:
            calls.append(("remove", session_id))

        monkeypatch.setattr(provenance, "purge_session", _purge)
        monkeypatch.setattr(sessions, "remove", _remove)
        client = TestClient(server.app)
        resp = client.delete("/api/sessions/abc")
        assert resp.status_code == 200
        assert calls == [("purge", "abc"), ("remove", "abc")]
