"""Tests for the workspace server's path guard and settings-merge logic.

These two pieces carry the workspace's security-relevant invariants — the
filesystem API must never resolve outside ``WORKSPACE_ROOT``, and a settings
save must not silently deactivate entries the drawer never saw — and both are
exercisable without a live vibe-acp subprocess.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

from medmcp import provenance, server, sessions, settings

from .conftest import WS_HEADERS


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the server's workspace root at a temp dir (resolved, no symlinks)."""
    resolved = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
    return resolved


# ── _safe_path: the workspace traversal guard ────────────────────────────────


class TestSafePath:
    """``_safe_path`` resolves inside the workspace and rejects everything else."""

    def test_allows_nested_relative_path(self, workspace_root: Path) -> None:
        """A normal relative path resolves to its location under the root."""
        assert (
            server._safe_path("sub/dir/scan.nii.gz")
            == workspace_root / "sub" / "dir" / "scan.nii.gz"
        )

    def test_allows_root_itself(self, workspace_root: Path) -> None:
        """An empty path (and ``.``) resolve to the workspace root, not an escape."""
        assert server._safe_path("") == workspace_root
        assert server._safe_path(".") == workspace_root

    def test_inner_dotdot_that_stays_inside_is_allowed(self, workspace_root: Path) -> None:
        """``..`` segments are fine as long as the result stays in the workspace."""
        assert server._safe_path("a/../b.txt") == workspace_root / "b.txt"

    def test_rejects_parent_traversal(self, workspace_root: Path) -> None:
        """A leading ``..`` that escapes the root is refused with 400."""
        with pytest.raises(HTTPException) as exc:
            server._safe_path("../secret.txt")
        assert exc.value.status_code == 400
        assert "escapes" in str(exc.value.detail)

    def test_rejects_deep_traversal(self, workspace_root: Path) -> None:
        """A path that climbs out via many ``..`` is refused."""
        with pytest.raises(HTTPException) as exc:
            server._safe_path("a/b/../../../../etc/passwd")
        assert exc.value.status_code == 400

    def test_rejects_absolute_path(self, workspace_root: Path) -> None:
        """An absolute path is rejected before any resolution."""
        with pytest.raises(HTTPException) as exc:
            server._safe_path("/etc/passwd")
        assert exc.value.status_code == 400
        assert "absolute" in str(exc.value.detail)

    def test_rejects_sibling_with_shared_prefix(self, workspace_root: Path) -> None:
        """A sibling dir whose name merely starts with the root's name is outside.

        Guards against a string-prefix check; ``is_relative_to`` compares path
        components, so ``<root>_evil`` must not be treated as inside ``<root>``.
        """
        sibling = workspace_root.parent / (workspace_root.name + "_evil")
        sibling.mkdir()
        with pytest.raises(HTTPException) as exc:
            server._safe_path(f"../{sibling.name}/loot.txt")
        assert exc.value.status_code == 400

    def test_rejects_symlink_escape(self, workspace_root: Path) -> None:
        """A symlink inside the workspace pointing out is followed and rejected."""
        outside = workspace_root.parent / "outside"
        outside.mkdir()
        (workspace_root / "link").symlink_to(outside)
        with pytest.raises(HTTPException) as exc:
            server._safe_path("link/secret.txt")
        assert exc.value.status_code == 400
        assert "escapes" in str(exc.value.detail)


class TestMountPointGuard:
    """A mounted drive inside the workspace, or a folder holding one, cannot be removed."""

    @pytest.fixture
    def drive(self, workspace_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A workspace whose ``usb`` folder counts as a bind mount (invisible to ismount)."""
        drive = workspace_root / "study" / "usb"
        (drive / "series0").mkdir(parents=True)
        (drive / "series0" / "img0.dcm").write_bytes(b"x")
        monkeypatch.setattr(server, "_mount_points", lambda: {Path("/"), drive})
        return drive

    def test_mount_point_cannot_be_deleted_or_moved(self, drive: Path, client: TestClient) -> None:
        """Deleting the mount point would empty the drive behind it."""
        assert client.delete("/api/files", params={"path": "study/usb"}).status_code == 400
        resp = client.post("/api/files/rename", json={"path": "study/usb", "new_path": "x"})
        assert resp.status_code == 400
        assert (drive / "series0" / "img0.dcm").exists()

    def test_folder_holding_a_mount_cannot_be_deleted(
        self, drive: Path, client: TestClient
    ) -> None:
        """Deleting a parent would empty the drive below it before failing."""
        assert client.delete("/api/files", params={"path": "study"}).status_code == 400
        assert (drive / "series0" / "img0.dcm").exists()

    def test_files_on_the_drive_can_still_be_deleted(self, drive: Path, client: TestClient) -> None:
        """Only the mount point itself is protected."""
        assert client.delete("/api/files", params={"path": "study/usb/series0"}).status_code == 200
        assert not (drive / "series0").exists()

    def test_mount_table_is_read(self) -> None:
        """The real mount table parses and includes the root filesystem."""
        assert Path("/") in server._mount_points()

    def test_mount_paths_are_decoded_as_bytes(self) -> None:
        """Octal escapes become the bytes they stand for; non-ASCII bytes stay as written."""
        assert server._unescape_mount_path(rb"/media/u/My\040Disk") == Path("/media/u/My Disk")
        assert server._unescape_mount_path(b"/media/u/K\xc3\xbcche") == Path("/media/u/Küche")
        assert server._unescape_mount_path(rb"/media/u/a\134b") == Path("/media/u/a\\b")


# ── _resolve_input_path: workspace-relative replay inputs → absolute ──────────


class TestResolveInputPath:
    """Replay inputs arrive workspace-relative; the stack tools need absolute paths.

    The Run form (drags carry explorer tree ids) hands the server paths relative
    to ``WORKSPACE_ROOT``, but replay calls the stack tools directly and they
    resolve paths on disk. Existing relative paths must be lifted to absolute;
    everything else (already-absolute, non-path args, missing files) is left as-is
    so non-path inputs pass through and a missing scan still yields a clear error.
    """

    def test_relative_existing_file_becomes_absolute(self, workspace_root: Path) -> None:
        """A relative path naming an existing file resolves to its absolute path."""
        scan = workspace_root / "patient_01" / "visit_02" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        assert server._resolve_input_path("patient_01/visit_02/t1n_3d.nii.gz") == str(scan)

    def test_relative_existing_dir_becomes_absolute(self, workspace_root: Path) -> None:
        """Directory inputs resolve too (some tools take a folder)."""
        visit = workspace_root / "patient_01" / "visit_02"
        visit.mkdir(parents=True)
        assert server._resolve_input_path("patient_01/visit_02") == str(visit)

    def test_absolute_path_is_left_unchanged(self, workspace_root: Path) -> None:
        """An already-absolute path (the recorded example) passes through verbatim."""
        scan = workspace_root / "patient_01" / "visit_01" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        assert server._resolve_input_path(str(scan)) == str(scan)

    def test_non_path_argument_is_left_unchanged(self, workspace_root: Path) -> None:
        """A non-path argument like a device string is not mistaken for a file."""
        assert server._resolve_input_path("cuda") == "cuda"

    def test_missing_relative_path_is_left_unchanged(self, workspace_root: Path) -> None:
        """A relative path that doesn't exist is left as-is for a clear tool error."""
        assert (
            server._resolve_input_path("patient_01/visit_09/missing.nii.gz")
            == "patient_01/visit_09/missing.nii.gz"
        )

    def test_empty_value_is_left_unchanged(self, workspace_root: Path) -> None:
        """An empty value is returned untouched (not joined to the root)."""
        assert server._resolve_input_path("") == ""

    def test_confined_input_paths_maps_every_value(self, workspace_root: Path) -> None:
        """The dict helper resolves each binding value independently."""
        scan = workspace_root / "patient_01" / "visit_02" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        resolved = server._confined_input_paths(
            {"in_1": "patient_01/visit_02/t1n_3d.nii.gz", "in_2": "cuda"}
        )
        assert resolved == {"in_1": str(scan), "in_2": "cuda"}


class TestConfinedInputPath:
    """Replay inputs may not name anything outside the workspace.

    Replay calls stack tools directly, with no permission prompt between the
    binding and the call, so a path leaving the workspace is either a mistake or
    an attempt to have a tool read something it should not. In the containerized
    deployment only the workspace is mounted into a stack, but a host-native
    stack runs as the operator — so the confinement belongs in the server.
    """

    def test_absolute_path_inside_is_kept(self, workspace_root: Path) -> None:
        """Path parity means an absolute workspace path is the normal case."""
        scan = workspace_root / "patient_01" / "scan.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        assert server._confined_input_path(str(scan)) == str(scan)

    def test_absolute_path_outside_is_refused(self, workspace_root: Path) -> None:
        """The case that mattered: a tool pointed at the operator's own files."""
        with pytest.raises(ValueError, match="outside the workspace"):
            server._confined_input_path("/etc/passwd")

    def test_relative_escape_is_refused(self, workspace_root: Path) -> None:
        """`..` is not a way around the same rule."""
        with pytest.raises(ValueError, match="escapes the workspace"):
            server._confined_input_path("../../etc/passwd")

    def test_symlink_out_of_the_workspace_is_refused(self, workspace_root: Path) -> None:
        """Resolution happens before the check, so a link inside is not a loophole."""
        link = workspace_root / "escape"
        link.symlink_to("/etc")
        with pytest.raises(ValueError, match="workspace"):
            server._confined_input_path("escape/passwd")

    def test_non_path_and_missing_values_pass_through(self, workspace_root: Path) -> None:
        """Device strings and not-yet-created outputs are not paths to refuse."""
        assert server._confined_input_path("cuda") == "cuda"
        assert server._confined_input_path("out/derivatives") == "out/derivatives"


# ── _workspace_note: the viewer-context note handed to the agent ─────────────


class TestWorkspaceNote:
    """The viewer-context note carries the absolute path so the agent calls tools right.

    The viewer reports a workspace-relative path, but the stack tools run in
    sibling containers and resolve paths on disk. The note must hand the agent the
    absolute path so its *first* tool call hits — without it the agent passes the
    relative path, the tool reports "not found", and it only recovers after a
    filesystem search. The note must also still be strippable from transcripts.
    """

    def test_note_carries_absolute_path(self, workspace_root: Path) -> None:
        """A relative viewer path is rendered as its absolute location in the note."""
        scan = workspace_root / "patient_01" / "visit_01" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        note = server._workspace_note("patient_01/visit_01/t1n_3d.nii.gz")
        assert str(scan) in note
        assert "patient_01/visit_01/t1n_3d.nii.gz" not in note.replace(str(scan), "")

    def test_note_is_strippable(self, workspace_root: Path) -> None:
        """The note appends cleanly and `_strip_workspace_note` removes it again."""
        scan = workspace_root / "patient_01" / "visit_01" / "t1n_3d.nii.gz"
        scan.parent.mkdir(parents=True)
        scan.touch()
        prompt = "skull strip this image" + server._workspace_note(
            "patient_01/visit_01/t1n_3d.nii.gz"
        )
        assert server._strip_workspace_note(prompt) == "skull strip this image"


# ── PUT /api/settings: the merge-not-overwrite logic ─────────────────────────


class TestSettingsMerge:
    """The settings PUT merges the drawer's known stack list with the current state.

    The drawer submits every stack it knew about (name + active). Stacks it never
    saw (e.g. one installed while the drawer was open) must keep their current
    active state rather than being dropped.
    """

    @pytest.fixture
    def harness(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, set[str]]:
        """Stub settings persistence and the vibe restart; capture what's saved.

        ``old_stacks`` seeds the current active set; ``saved`` records the merged
        set the endpoint persists. The vibe-acp restart and config sync are
        stubbed so the endpoint can run without a subprocess.
        """
        old_stacks = {"alpha", "beta"}
        saved: dict[str, set[str]] = {}

        # Strict pyright rejects untyped lambdas, so the stubs are typed defs.
        def _save_stacks(names: Iterable[str]) -> None:
            saved["stacks"] = set(names)

        def _noop_bool(_value: bool) -> None:
            return None

        def _noop_sync(_servers: object) -> None:
            return None

        def _active_servers() -> list[object]:
            return []

        async def _no_restart() -> None:
            return None

        monkeypatch.setattr(settings, "load_active_server_names", lambda: set(old_stacks))
        monkeypatch.setattr(settings, "save_explain_enabled", _noop_bool)
        monkeypatch.setattr(settings, "save_provenance_enabled", _noop_bool)
        monkeypatch.setattr(settings, "save_active_server_names", _save_stacks)
        monkeypatch.setattr(settings, "sync_servers_to_vibe_config", _noop_sync)
        monkeypatch.setattr(settings, "active_servers", _active_servers)
        monkeypatch.setattr(server, "_restart_vibe", _no_restart)
        return saved

    @staticmethod
    def _put(client: TestClient, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "explain_tools": True,
            "record_provenance": False,
            "stacks": [],
        }
        body.update(overrides)
        resp = client.put("/api/settings", json=body)
        assert resp.status_code == 200
        return resp.json()

    def test_preserves_entry_unknown_to_the_drawer(
        self, harness: dict[str, set[str]], client: TestClient
    ) -> None:
        """A stack the drawer never listed stays active after a save."""
        result = self._put(client, stacks=[{"name": "alpha", "active": True}])
        # "beta" was unknown to this drawer payload, so it is kept; nothing changed.
        assert harness["stacks"] == {"alpha", "beta"}
        assert result["restarted"] is False

    def test_deactivates_a_known_entry(
        self, harness: dict[str, set[str]], client: TestClient
    ) -> None:
        """Turning a listed stack off removes it and triggers a restart."""
        result = self._put(
            client,
            stacks=[{"name": "alpha", "active": False}, {"name": "beta", "active": True}],
        )
        assert harness["stacks"] == {"beta"}
        assert result["restarted"] is True

    def test_no_restart_when_nothing_changes(
        self, harness: dict[str, set[str]], client: TestClient
    ) -> None:
        """Re-submitting the current state is a no-op restart-wise."""
        result = self._put(
            client,
            stacks=[{"name": "alpha", "active": True}, {"name": "beta", "active": True}],
        )
        assert harness["stacks"] == {"alpha", "beta"}
        assert result["restarted"] is False


# ── Workflow export / import endpoints ───────────────────────────────────────


class TestWorkflowShareEndpoints:
    """Export/import workflow endpoints: wiring + error mapping."""

    @pytest.fixture
    def vibe_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the workflow root (resolved via provenance.VIBE_HOME) at a temp dir."""
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

    def test_import_then_export_round_trip(self, vibe_home: Path, client: TestClient) -> None:
        """A valid envelope imports as a draft and exports back as a YAML download."""
        resp = client.post("/api/workflows/import", json={"content": self._envelope()})
        assert resp.status_code == 200
        assert resp.json()["name"] == "demo-flow"

        exp = client.get("/api/workflows/demo-flow/export")
        assert exp.status_code == 200
        assert "medmcp_workflow" in exp.text
        assert exp.headers["content-disposition"].endswith('demo-flow.workflow.yaml"')

    def test_import_malformed_returns_400(self, vibe_home: Path, client: TestClient) -> None:
        """A payload that isn't a workflow envelope maps to HTTP 400."""
        resp = client.post("/api/workflows/import", json={"content": "- not a mapping"})
        assert resp.status_code == 400

    def test_export_missing_returns_404(self, vibe_home: Path, client: TestClient) -> None:
        """Exporting an unknown workflow maps to HTTP 404."""
        assert client.get("/api/workflows/nope/export").status_code == 404

    def test_rename_onto_a_taken_name_is_409(self, vibe_home: Path, client: TestClient) -> None:
        """A rename never replaces another workflow."""
        first = client.post("/api/workflows/import", json={"content": self._envelope()})
        assert first.status_code == 200
        other = yaml.safe_load(self._envelope())
        other["name"] = "other-flow"
        second = client.post("/api/workflows/import", json={"content": yaml.safe_dump(other)})
        assert second.status_code == 200

        resp = client.post("/api/workflows/other-flow/rename", json={"new_name": "Demo Flow"})

        assert resp.status_code == 409
        assert client.get("/api/workflows/demo-flow").json()["description"] == "demo"
        assert client.get("/api/workflows/other-flow").status_code == 200


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


class TestReplayedUserText:
    """Replayed user turns prefer the note-free display content vibe echoes back."""

    def test_prefers_user_display_content(self) -> None:
        """The ``user_display_content`` meta wins over the stored text."""
        update = {
            "sessionUpdate": "user_message_chunk",
            "content": {"type": "text", "text": "segment this\n\n[workspace context: …]"},
            "_meta": {
                "user_display_content": {
                    "version": "1",
                    "host": "medmcp",
                    "content": [{"type": "text", "text": "segment this"}],
                }
            },
        }
        assert server._replayed_user_text(update) == "segment this"

    def test_falls_back_to_stripping_the_note(self) -> None:
        """Transcripts from before the display-content meta still strip cleanly."""
        update = {
            "sessionUpdate": "user_message_chunk",
            "content": {
                "type": "text",
                "text": 'segment this\n\n[workspace context: the file "a/b.nii.gz" is currently '
                "open in the viewer]",
            },
        }
        assert server._replayed_user_text(update) == "segment this"

    def test_empty_display_content_falls_back(self) -> None:
        """A present-but-empty display meta does not blank the message."""
        empty: list[object] = []
        update = {
            "sessionUpdate": "user_message_chunk",
            "content": {"type": "text", "text": "plain question"},
            "_meta": {"user_display_content": {"version": "1", "host": "x", "content": empty}},
        }
        assert server._replayed_user_text(update) == "plain question"

    def test_non_text_content_is_ignored(self) -> None:
        """Image/resource chunks yield no text (the UI only renders text turns)."""
        update = {
            "sessionUpdate": "user_message_chunk",
            "content": {"type": "image", "data": "…"},
        }
        assert server._replayed_user_text(update) == ""


class _StubWs:
    """Collects frames a _ChatConnection would send to the browser."""

    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, msg: dict[str, object]) -> None:
        self.sent.append(msg)


def _connection() -> tuple[server._ChatConnection, asyncio.Queue[dict[str, object]]]:
    """A chat connection over a stub socket, with the vibe frame queue it drains."""
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    conn = server._ChatConnection(cast("Any", _StubWs()), "sid", cast("Any", queue), [])
    return conn, queue


class TestIdlePump:
    """Unsolicited vibe frames (e.g. MCP failure notices) reach the browser."""

    async def test_pump_relays_queued_notice(self) -> None:
        """A warm-up agent message queued before any prompt is forwarded."""
        conn, queue = _connection()
        await queue.put(
            {
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {
                            "type": "text",
                            "text": "The following MCP servers failed to connect:\n- x: boom",
                        },
                    }
                },
            }
        )
        conn.start_idle_pump()
        await asyncio.sleep(0.05)
        await conn._stop_idle_pump()
        ws = cast("_StubWs", conn.ws)
        assert any(
            m.get("type") == "chunk" and "failed to connect" in str(m.get("text")) for m in ws.sent
        )

    async def test_stop_is_idempotent_and_halts_relay(self) -> None:
        """After stopping, queued frames stay queued (drained by the turn loop)."""
        conn, queue = _connection()
        conn.start_idle_pump()
        await conn._stop_idle_pump()
        await conn._stop_idle_pump()  # no error
        await queue.put({"method": "session/update", "params": {}})
        await asyncio.sleep(0.05)
        assert cast("_StubWs", conn.ws).sent == []
        assert queue.qsize() == 1


class TestToolCallRawInput:
    """Arguments streamed in after the call is announced still reach the UI.

    vibe announces a tool call before the model has finished streaming its
    arguments, so the opening ``tool_call`` frame can carry an empty rawInput and
    the completed one arrives on a later ``tool_call_update``. Missing that
    second copy leaves the approval box with no arguments to show and the
    provenance event with no ``arguments`` — see ``_effect_progress_updates`` in
    vibe's session_updates (rawInput is re-sent only when the detail changed).
    """

    @staticmethod
    def _update(payload: dict[str, object]) -> dict[str, object]:
        return {"method": "session/update", "params": {"update": payload}}

    async def test_late_arguments_overwrite_the_partial_placeholder(self) -> None:
        """The completed rawInput replaces the empty one from the opening frame."""
        conn, queue = _connection()
        await queue.put(
            self._update(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_1",
                    "title": "medmcp-neuro-core_skull_strip",
                    "status": "pending",
                    "rawInput": {},
                }
            )
        )
        await queue.put(
            self._update(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call_1",
                    "status": "in_progress",
                    "rawInput": {"image": "/data/T1.nii.gz"},
                }
            )
        )
        conn.start_idle_pump()
        await asyncio.sleep(0.05)
        await conn._stop_idle_pump()

        assert conn._tool_calls["call_1"]["rawInput"] == {"image": "/data/T1.nii.gz"}
        sent = cast("_StubWs", conn.ws).sent
        update_frame = next(m for m in sent if m.get("type") == "tool_call_update")
        assert update_frame["rawInput"] == {"image": "/data/T1.nii.gz"}

    async def test_replayed_settled_call_carries_its_output(self) -> None:
        """A finished call replayed on resume keeps its output.

        vibe replays it as one ``tool_call`` frame with the output in ``content``
        and no ``tool_call_update`` after it; the browser needs that output to show
        the result and its file chips.
        """
        conn, queue = _connection()
        await queue.put(
            self._update(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_1",
                    "title": "medmcp-neuro-core_segment_brain",
                    "status": "completed",
                    "rawInput": {"input_path": "/data/T1.nii.gz"},
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "text", "text": "Seg labels: /data/T1_dseg.mgz"},
                        }
                    ],
                }
            )
        )
        conn.start_idle_pump()
        await asyncio.sleep(0.05)
        await conn._stop_idle_pump()

        frame = next(m for m in cast("_StubWs", conn.ws).sent if m.get("type") == "tool_call")
        assert frame["output"] == "Seg labels: /data/T1_dseg.mgz"
        assert frame["pathGuardRetry"] is False

    async def test_update_without_arguments_keeps_the_known_ones(self) -> None:
        """A status-only update (vibe sends rawInput only when detail changed)."""
        conn, queue = _connection()
        await queue.put(
            self._update(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_1",
                    "title": "bash",
                    "status": "pending",
                    "rawInput": {"command": "ls"},
                }
            )
        )
        await queue.put(
            self._update(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call_1",
                    "status": "in_progress",
                }
            )
        )
        conn.start_idle_pump()
        await asyncio.sleep(0.05)
        await conn._stop_idle_pump()

        assert conn._tool_calls["call_1"]["rawInput"] == {"command": "ls"}


class TestUsageWindow:
    """The context meter denominator: Ollama's num_ctx wins over vibe's registry figure."""

    def test_fetched_ollama_value_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fetched num_ctx beats the frame's size (vibe reports its registry figure)."""
        monkeypatch.setattr(settings, "_context_window_tokens", 131072)
        assert server._usage_window({"used": 1, "size": 200000}) == 131072

    def test_frame_size_stands_in_before_first_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Before the cache is warmed, vibe's size beats the static default."""
        monkeypatch.setattr(settings, "_context_window_tokens", None)
        assert server._usage_window({"size": 200000}) == 200000

    def test_static_default_last(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no fetch and no frame size, the static default applies."""
        monkeypatch.setattr(settings, "_context_window_tokens", None)
        assert server._usage_window({}) == settings.DEFAULT_CONTEXT_WINDOW


class TestVisiblePermissionOptions:
    """No auto-approval option is ever offered to the browser."""

    @staticmethod
    def _vibe_options() -> list[dict[str, object]]:
        """The four options vibe-acp offers on every permission request."""
        return [
            {"optionId": "allow_once", "name": "Allow once"},
            {"optionId": "allow_always", "name": "Allow for remainder of this session"},
            {"optionId": "allow_always_permanent", "name": "Always allow"},
            {"optionId": "reject_once", "name": "Deny"},
        ]

    def test_drops_both_auto_approve_options(self) -> None:
        """Session-scoped and permanent "always" both go; allow/deny remain."""
        visible = server._visible_permission_options(self._vibe_options())
        assert [o["optionId"] for o in visible] == ["allow_once", "reject_once"]

    def test_relabels_allow_once(self) -> None:
        """With no "always" variant to contrast against, the scope drops from the label."""
        visible = server._visible_permission_options(self._vibe_options())
        assert [o["name"] for o in visible] == ["Allow", "Deny"]

    def test_does_not_mutate_input(self) -> None:
        """The relabel copies — the caller's option dicts are left untouched."""
        options = self._vibe_options()
        server._visible_permission_options(options)
        assert options[0]["name"] == "Allow once"

    def test_passes_unknown_options_through(self) -> None:
        """Only the auto-approve ids are dropped — future option ids are relayed."""
        options = [{"optionId": "reject_once"}, {"optionId": "something_new"}]
        assert server._visible_permission_options(options) == options


class TestReadOnlySessionApproval:
    """Read-only shell commands can be approved for the rest of the chat."""

    @staticmethod
    def _bash_options(*rules: tuple[str, str]) -> list[dict[str, object]]:
        """Options vibe sends for a bash call whose session approval adds *rules*."""
        meta = {
            "required_permissions": [
                {
                    "scope": scope,
                    "invocation_pattern": pattern,
                    "session_pattern": pattern,
                    "label": pattern,
                }
                for scope, pattern in rules
            ]
        }
        return [
            {"optionId": "allow_once", "name": "Allow once"},
            {
                "optionId": "allow_always",
                "name": "Allow for remainder of this session",
                "_meta": meta,
            },
            {"optionId": "allow_always_permanent", "name": "Always allow", "_meta": meta},
            {"optionId": "reject_once", "name": "Deny"},
        ]

    @pytest.fixture(autouse=True)
    def _default_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(server.READ_ONLY_APPROVAL_ENV_VAR, raising=False)

    def test_offers_session_approval_for_read_only_command(self) -> None:
        """``ls -la`` gets a third button naming the command; "Always allow" stays hidden."""
        visible = server._visible_permission_options(
            self._bash_options(("command_pattern", "ls *"))
        )
        assert [o["optionId"] for o in visible] == ["allow_once", "allow_always", "reject_once"]
        assert visible[1]["name"] == "Allow ls for this chat"

    def test_names_every_command_of_a_compound(self) -> None:
        """``ls; wc -l x`` needs two rules; both are read-only, so both are named."""
        options = self._bash_options(("command_pattern", "ls *"), ("command_pattern", "wc *"))
        visible = server._visible_permission_options(options)
        assert visible[1]["name"] == "Allow ls, wc for this chat"

    @pytest.mark.parametrize(
        "rules",
        [
            (("command_pattern", "rm *"),),
            (("command_pattern", "ls *"), ("command_pattern", "rm *")),
            (("command_pattern", "ls *"), ("command_pattern", "<redirect> *")),
            (("command_pattern", "find . -delete"),),
            (("command_pattern", "python *"),),
            (("outside_directory", "/etc/*"),),
            (("command_pattern", "ls *"), ("outside_directory", "/etc/*")),
        ],
    )
    def test_withholds_session_approval_unless_every_rule_is_read_only(
        self, rules: tuple[tuple[str, str], ...]
    ) -> None:
        """A writing command, redirect, exact pattern or outside path disqualifies."""
        visible = server._visible_permission_options(self._bash_options(*rules))
        assert [o["optionId"] for o in visible] == ["allow_once", "reject_once"]

    def test_withholds_session_approval_for_a_redirect(self) -> None:
        """``ls > out.txt`` reaches us as ``ls <redirect>`` with session pattern ``ls *``."""
        options = self._bash_options(("command_pattern", "ls *"))
        meta = cast("dict[str, list[dict[str, str]]]", options[1]["_meta"])
        meta["required_permissions"][0]["invocation_pattern"] = "ls -la <redirect>"
        visible = server._visible_permission_options(options)
        assert [o["optionId"] for o in visible] == ["allow_once", "reject_once"]

    def test_withholds_tool_wide_session_approval(self) -> None:
        """No rules means vibe would approve the whole tool for the session — never offered."""
        options: list[dict[str, object]] = [
            {"optionId": "allow_always", "name": "x", "_meta": {"required_permissions": []}}
        ]
        assert server._visible_permission_options(options) == []

    @pytest.mark.parametrize("value", ["0", "false", "off"])
    def test_kill_switch_restores_per_call_approval(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """``MEDMCP_READ_ONLY_APPROVAL=0`` drops the session option again."""
        monkeypatch.setenv(server.READ_ONLY_APPROVAL_ENV_VAR, value)
        visible = server._visible_permission_options(
            self._bash_options(("command_pattern", "ls *"))
        )
        assert [o["optionId"] for o in visible] == ["allow_once", "reject_once"]


def _prov_dir_in(tmp_path: Path) -> Callable[[str], Path]:
    """Return a provenance_dir substitute rooted at *tmp_path*."""

    def _prov_dir(session_id: str) -> Path:
        return tmp_path / session_id

    return _prov_dir


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

    def test_maps_and_drops_idless_entries(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
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
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert [s["id"] for s in sessions] == ["s1", "s2", "s3"]
        assert sessions[0]["title"] == "Skull strip"
        assert sessions[0]["updatedAt"] == "2026-06-15T10:00:00Z"
        # The leaked viewer-context note is stripped from the title, even truncated.
        assert sessions[2]["title"] == "Hi"

    def test_vibe_error_maps_to_502(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        """A vibe-acp error surfaces as a 502, not a 500."""
        self._stub_client(monkeypatch, {"error": {"message": "boom"}})
        resp = client.get("/api/sessions")
        assert resp.status_code == 502

    def test_overlays_registry_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
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

        rows = {s["id"]: s for s in client.get("/api/sessions").json()["sessions"]}
        assert rows["s1"]["title"] == "My name"  # override wins over vibe's title
        assert rows["s1"]["archived"] is False
        assert rows["s1"]["hasProvenance"] is True
        assert rows["s2"]["title"] == "kept"
        assert rows["s2"]["archived"] is True
        assert rows["s2"]["hasProvenance"] is False

    def test_folds_compaction_continuations(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
    ) -> None:
        """A continuation is hidden; its root carries the chain's newest updatedAt."""
        self._stub_client(
            monkeypatch,
            {
                "result": {
                    "sessions": [
                        {"sessionId": "root", "title": "Chat", "updatedAt": "t1"},
                        {"sessionId": "cont", "title": None, "updatedAt": "t9"},
                        {"sessionId": "other", "title": "Other", "updatedAt": "t2"},
                    ]
                }
            },
        )
        monkeypatch.setattr(sessions, "load_registry", dict)
        monkeypatch.setattr(provenance, "vibe_session_parents", lambda: {"cont": "root"})
        monkeypatch.setattr(provenance, "provenance_dir", _prov_dir_in(tmp_path))
        rows = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in rows] == ["root", "other"]
        assert rows[0]["updatedAt"] == "t9"
        assert rows[1]["updatedAt"] == "t2"

    def test_multi_hop_chain_folds_to_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
    ) -> None:
        """Several compactions still collapse into the one root entry."""
        self._stub_client(
            monkeypatch,
            {
                "result": {
                    "sessions": [
                        {"sessionId": "root", "title": "Chat", "updatedAt": "t1"},
                        {"sessionId": "c1", "title": None, "updatedAt": "t5"},
                        {"sessionId": "c2", "title": None, "updatedAt": "t9"},
                    ]
                }
            },
        )
        monkeypatch.setattr(sessions, "load_registry", dict)
        monkeypatch.setattr(provenance, "vibe_session_parents", lambda: {"c1": "root", "c2": "c1"})
        monkeypatch.setattr(provenance, "provenance_dir", _prov_dir_in(tmp_path))
        rows = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in rows] == ["root"]
        assert rows[0]["updatedAt"] == "t9"

    def test_registry_entry_is_never_folded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
    ) -> None:
        """A child the user can see (registry record — e.g. a fork) stays listed."""
        self._stub_client(
            monkeypatch,
            {
                "result": {
                    "sessions": [
                        {"sessionId": "root", "title": "Chat", "updatedAt": "t1"},
                        {"sessionId": "fork", "title": None, "updatedAt": "t9"},
                    ]
                }
            },
        )
        monkeypatch.setattr(sessions, "load_registry", lambda: {"fork": {"title": "My branch"}})
        monkeypatch.setattr(provenance, "vibe_session_parents", lambda: {"fork": "root"})
        monkeypatch.setattr(provenance, "provenance_dir", _prov_dir_in(tmp_path))
        rows = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in rows] == ["root", "fork"]
        assert rows[0]["updatedAt"] == "t1"  # nothing folded into the root
        assert rows[1]["title"] == "My branch"

    def test_child_of_unlisted_parent_stays(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: TestClient
    ) -> None:
        """A backlink to a session outside this workspace's list does not hide it."""
        self._stub_client(
            monkeypatch,
            {"result": {"sessions": [{"sessionId": "c1", "title": "Solo", "updatedAt": "t1"}]}},
        )
        monkeypatch.setattr(sessions, "load_registry", dict)
        monkeypatch.setattr(provenance, "vibe_session_parents", lambda: {"c1": "elsewhere"})
        monkeypatch.setattr(provenance, "provenance_dir", _prov_dir_in(tmp_path))
        rows = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in rows] == ["c1"]

    def test_rename_sets_title(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """POST …/rename forwards to the registry's set_title."""
        captured: dict[str, tuple[str, str]] = {}

        def _set_title(session_id: str, title: str) -> None:
            captured["rename"] = (session_id, title)

        monkeypatch.setattr(sessions, "set_title", _set_title)
        resp = client.post("/api/sessions/abc/rename", json={"title": "New name"})
        assert resp.status_code == 200
        assert captured["rename"] == ("abc", "New name")

    def test_archive_sets_flag(self, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
        """POST …/archive forwards to the registry's set_archived."""
        captured: dict[str, tuple[str, bool]] = {}

        def _set_archived(session_id: str, archived: bool) -> None:
            captured["archive"] = (session_id, archived)

        monkeypatch.setattr(sessions, "set_archived", _set_archived)
        resp = client.post("/api/sessions/abc/archive", json={"archived": True})
        assert resp.status_code == 200
        assert captured["archive"] == ("abc", True)

    def test_delete_purges_then_forgets(
        self, monkeypatch: pytest.MonkeyPatch, client: TestClient
    ) -> None:
        """DELETE …/{id} purges the transcript/provenance and drops the registry entry."""
        calls: list[tuple[str, str]] = []

        def _purge(session_id: str, *, stop_ids: Iterable[str] = ()) -> None:
            del stop_ids
            calls.append(("purge", session_id))

        def _remove(session_id: str) -> None:
            calls.append(("remove", session_id))

        monkeypatch.setattr(provenance, "purge_session", _purge)
        monkeypatch.setattr(sessions, "remove", _remove)
        resp = client.delete("/api/sessions/abc")
        assert resp.status_code == 200
        assert calls == [("purge", "abc"), ("remove", "abc")]


def test_requirement_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-requirement status flags missing stacks and image-digest mismatches."""
    from medmcp.workflow import Recipe, RecipeStep, StackRequirement

    recipe = Recipe(
        name="wf",
        description="d",
        inputs=[],
        steps=[RecipeStep(server="medmcp-neuro", tool="t", arguments={})],
        requires=[
            StackRequirement(
                stack="medmcp-neuro", image="ghcr.io/medmcp/neuro:main", digest="sha256:aaa"
            ),
            StackRequirement(stack="medmcp-cardiac", image="ghcr.io/medmcp/cardiac:main"),
        ],
    )
    servers = [
        {
            "name": "medmcp-neuro",
            "command": "docker",
            "args": ["run", "-i", "ghcr.io/medmcp/neuro:main"],
        }
    ]

    def _installed_digest(_image: str) -> str | None:
        return "sha256:zzz"  # differs from the pinned sha256:aaa

    monkeypatch.setattr(settings, "resolve_image_digest", _installed_digest)
    by_stack = {s["stack"]: s for s in server._requirement_statuses(recipe, servers)}
    assert by_stack["medmcp-neuro"]["status"] == "mismatch"
    assert by_stack["medmcp-neuro"]["installed_digest"] == "sha256:zzz"
    assert by_stack["medmcp-cardiac"]["status"] == "missing"  # not in servers

    def _matching_digest(_image: str) -> str | None:
        return "sha256:aaa"  # matches the pin

    monkeypatch.setattr(settings, "resolve_image_digest", _matching_digest)
    ok = {s["stack"]: s for s in server._requirement_statuses(recipe, servers)}
    assert ok["medmcp-neuro"]["status"] == "ok"


class TestPathGuardDenialTagging:
    """A guard denial is tagged so the browser need not parse vibe's wording."""

    def test_recognises_a_pathguard_denial(self) -> None:
        """The hook's own name is the marker, not the reason text."""
        out = (
            "Tool 'medmcp-neuro-core_skull_strip' was denied by hook "
            "'medmcp-pathguard': RETRY THIS CALL WITH A CORRECTED PATH."
        )
        assert server._is_pathguard_denial(out)

    def test_a_real_tool_failure_is_not_tagged(self) -> None:
        """An actual failure must keep rendering as one."""
        assert not server._is_pathguard_denial("Error executing tool: exit 1")
        assert not server._is_pathguard_denial("")

    def test_another_hook_is_not_tagged(self) -> None:
        """Only this guard's denials are softened; someone else's stay visible."""
        assert not server._is_pathguard_denial("Tool 'x' was denied by hook 'other': no")


# ── Replay runs: pre-flight preview, detached runs over the socket ───────────


class TestReplayRuns:
    """The run screen's preview pre-flights every item; runs outlive the socket."""

    @pytest.fixture
    def env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient) -> Path:
        """A temp VIBE_HOME with one imported workflow, a temp workspace, temp runs."""
        from medmcp import replay, runs

        monkeypatch.setattr(provenance, "VIBE_HOME", tmp_path)
        monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
        workspace = (tmp_path / "ws").resolve()
        workspace.mkdir()
        monkeypatch.setattr(server, "WORKSPACE_ROOT", workspace)
        neuro: list[dict[str, Any]] = [{"name": "medmcp-neuro", "command": "x", "args": []}]
        monkeypatch.setattr(settings, "active_servers", lambda: neuro)
        (workspace / "t1.nii.gz").write_bytes(b"x")
        envelope = yaml.safe_dump(
            {
                "medmcp_workflow": 1,
                "name": "strip",
                "description": "demo",
                "inputs": [{"name": "in_1", "example": "/d/a.nii.gz"}],
                "steps": [
                    {
                        "server": "medmcp-neuro",
                        "tool": "skull_strip",
                        "arguments": {"input_path": "{{in_1}}"},
                        "produces": {"brain_path": "step1.brain_path"},
                    }
                ],
            }
        )
        assert client.post("/api/workflows/import", json={"content": envelope}).status_code == 200

        # The engine never spawns a real stack here.
        import contextlib
        from collections.abc import AsyncGenerator

        @contextlib.asynccontextmanager
        async def fake_caller(*_a: object, **_k: object) -> AsyncGenerator[replay.ToolCaller]:
            async def _call(
                srv: str, tool: str, args: dict[str, Any]
            ) -> tuple[bool, dict[str, Any], str | None]:
                return True, {"brain_path": str(workspace / "brain.nii.gz")}, None

            yield _call

        monkeypatch.setattr(replay, "mcp_caller", fake_caller)
        return workspace

    def test_preview_flags_a_missing_input_per_item(self, env: Path, client: TestClient) -> None:
        """A row pointing at a file that is not there is reported before anything runs."""
        resp = client.post(
            "/api/workflows/strip/replay-preview",
            json={"runs": [{"in_1": "t1.nii.gz"}, {"in_1": "typo.nii.gz"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        items = body["items"]
        assert [i["ok"] for i in items] == [True, False]
        assert items[1]["findings"][0]["status"] == "missing"
        assert items[1]["findings"][0]["role"] == "input"  # named after the tool's argument
        assert body["steps"][0]["arguments"]["input_path"] == str(env / "t1.nii.gz")

    def test_run_survives_disconnect_and_can_be_reattached(
        self, env: Path, client: TestClient
    ) -> None:
        """Closing the socket detaches; the run finishes and a later attach replays it."""
        with client.websocket_connect("/ws/replay", headers=WS_HEADERS) as ws:
            ws.send_json({"name": "strip", "runs": [{"in_1": "t1.nii.gz"}]})
            started = ws.receive_json()
            assert started["type"] == "started"
            run_id = started["run_id"]
        # Detached — the run keeps going. Attach again and read the whole story.
        with client.websocket_connect("/ws/replay", headers=WS_HEADERS) as ws:
            ws.send_json({"attach": run_id})
            frames: list[dict[str, Any]] = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "result":
                    break
        assert frames[0]["type"] == "started" and frames[0]["run_id"] == run_id
        assert frames[-1]["ok"] is True
        assert any(f["type"] == "step" and f["ok"] for f in frames)

        listing = client.get("/api/runs", params={"workflow": "strip"}).json()
        assert [r["id"] for r in listing["runs"]] == [run_id]
        assert listing["runs"][0]["status"] == "done"
        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["items"][0]["steps"][0]["produced"]["step1.brain_path"].endswith(
            "brain.nii.gz"
        )
        assert client.delete(f"/api/runs/{run_id}").status_code == 200
        assert client.get(f"/api/runs/{run_id}").status_code == 404

    def test_bad_first_message_is_refused(self, env: Path, client: TestClient) -> None:
        """Neither a start nor an attach: the socket answers with an error and closes."""
        with client.websocket_connect("/ws/replay", headers=WS_HEADERS) as ws:
            ws.send_json({"attach": "nope"})
            assert ws.receive_json()["ok"] is False


def test_clip_output_cuts_on_whitespace() -> None:
    """A long output is shortened at a word boundary, never through a path."""
    path = "/data/sub-01/T1_dseg.nii.gz"
    text = ("x" * 1990) + " " + path
    clipped = server._clip_output(text)
    assert clipped == ("x" * 1990) + " …"
    assert server._clip_output("short") == "short"
    # No whitespace anywhere near the limit: fall back to a hard cut.
    assert len(server._clip_output("y" * 5000)) == 2000 + 2


class TestFolderTransfer:
    """Whole folders can be uploaded file by file and downloaded as a zip."""

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point the server's workspace root at a temp dir."""
        resolved = (tmp_path / "ws").resolve()
        resolved.mkdir()
        monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
        return resolved

    @pytest.fixture
    def client(self, root: Path) -> TestClient:
        """A same-origin client for the real routes."""
        del root
        return TestClient(server.app, base_url="http://127.0.0.1:8100")

    def test_upload_recreates_nested_folders(self, root: Path, client: TestClient) -> None:
        """A folder upload sends each file with its relative folder as ``dir``."""
        resp = client.post(
            "/api/files/upload",
            params={"dir": "case1/series2"},
            files={"file": ("IM0001.dcm", b"dicom-bytes")},
        )
        assert resp.status_code == 200
        assert (root / "case1" / "series2" / "IM0001.dcm").read_bytes() == b"dicom-bytes"

    def test_upload_rejects_dot_names(self, client: TestClient) -> None:
        """A file named ``..`` cannot land beside the target folder."""
        resp = client.post("/api/files/upload", params={"dir": "a"}, files={"file": ("..", b"x")})
        assert resp.status_code == 400

    def test_upload_rejects_escaping_dir(self, client: TestClient) -> None:
        """The target folder goes through the workspace path guard."""
        resp = client.post(
            "/api/files/upload", params={"dir": "../out"}, files={"file": ("a.txt", b"x")}
        )
        assert resp.status_code == 400

    def test_download_file_is_an_attachment(self, root: Path, client: TestClient) -> None:
        """A single file comes back as itself, named for saving."""
        (root / "t1.nii.gz").write_bytes(b"nifti")
        resp = client.get("/api/files/download", params={"path": "t1.nii.gz"})
        assert resp.status_code == 200
        assert resp.content == b"nifti"
        assert "attachment" in resp.headers["content-disposition"]

    def test_download_folder_is_a_zip_of_its_tree(self, root: Path, client: TestClient) -> None:
        """Nested files keep their paths under the folder's own name."""
        (root / "sub-01" / "anat").mkdir(parents=True)
        (root / "sub-01" / "anat" / "T1w.nii.gz").write_bytes(b"t1")
        (root / "sub-01" / "notes.txt").write_text("hello")
        resp = client.get("/api/files/download", params={"path": "sub-01"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "sub-01.zip" in resp.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert sorted(zf.namelist()) == ["sub-01/anat/T1w.nii.gz", "sub-01/notes.txt"]
            assert zf.read("sub-01/anat/T1w.nii.gz") == b"t1"
            assert zf.testzip() is None

    def test_download_folder_skips_symlinks_and_skip_dirs(
        self, root: Path, tmp_path: Path, client: TestClient
    ) -> None:
        """A link pointing outside the workspace, or ``.git``, never enters the archive."""
        outside = tmp_path / "secret.txt"
        outside.write_text("host file")
        (root / "d" / ".git").mkdir(parents=True)
        (root / "d" / ".git" / "HEAD").write_text("ref")
        (root / "d" / "kept.txt").write_text("ok")
        (root / "d" / "link.txt").symlink_to(outside)
        (root / "d" / "linkdir").symlink_to(tmp_path, target_is_directory=True)
        resp = client.get("/api/files/download", params={"path": "d"})
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert zf.namelist() == ["d/kept.txt"]

    def test_download_rejects_escaping_path(self, client: TestClient) -> None:
        """The download goes through the workspace path guard."""
        assert client.get("/api/files/download", params={"path": "../"}).status_code == 400

    def test_download_missing_path_is_404(self, client: TestClient) -> None:
        """Nothing there, nothing to send."""
        assert client.get("/api/files/download", params={"path": "nope"}).status_code == 404
