"""Tests for the workspace server's path guard and settings-merge logic.

These two pieces carry the workspace's security-relevant invariants — the
filesystem API must never resolve outside ``WORKSPACE_ROOT``, and a settings
save must not silently deactivate entries the drawer never saw — and both are
exercisable without a live vibe-acp subprocess.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import io
import logging
import os
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import yaml
from fastapi import HTTPException, WebSocketDisconnect
from fastapi.testclient import TestClient

from medmcp import provenance, server, sessions, settings

from .conftest import AUTH_HEADERS, WS_HEADERS


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the server's workspace root at a temp dir (resolved, no symlinks)."""
    resolved = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
    return resolved


# ── /api/raw: a file's bytes, never a stale copy ─────────────────────────────


class TestRawFiles:
    """The viewer must see a file as it is now, not as the browser cached it."""

    def test_raw_is_marked_no_cache_and_answers_head(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A rewritten file must reach the viewer: no heuristic caching, and HEAD revalidates."""
        (workspace_root / "seg.nii.gz").write_bytes(b"v1")
        got = client.get("/api/raw/seg.nii.gz")
        assert got.status_code == 200
        assert got.content == b"v1"
        assert got.headers["cache-control"] == "no-cache"
        head = client.head("/api/raw/seg.nii.gz")
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["etag"] == got.headers["etag"]
        (workspace_root / "seg.nii.gz").write_bytes(b"v2-longer")
        assert client.head("/api/raw/seg.nii.gz").headers["etag"] != got.headers["etag"]

    def test_optional_raw_answers_a_missing_file_with_204(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """An optional sidecar that is absent is not an error; the traversal guard still holds."""
        assert client.get("/api/raw/seg_labels.csv").status_code == 404
        missing = client.get("/api/raw/seg_labels.csv?optional=1")
        assert missing.status_code == 204
        assert missing.content == b""
        (workspace_root / "seg_labels.csv").write_text("label,structure\n1,liver\n")
        there = client.get("/api/raw/seg_labels.csv?optional=1")
        assert there.status_code == 200
        assert there.text == "label,structure\n1,liver\n"
        outside = client.get("/api/raw/%2e%2e/%2e%2e/etc/passwd?optional=1")
        assert outside.status_code not in (200, 204)

    def test_raw_answers_a_revalidation_with_304(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A client that already holds the bytes gets the validators back, not the file."""
        (workspace_root / "seg.nii.gz").write_bytes(b"v1")
        first = client.get("/api/raw/seg.nii.gz")
        etag = first.headers["etag"]
        again = client.get("/api/raw/seg.nii.gz", headers={"If-None-Match": etag})
        assert again.status_code == 304
        assert again.content == b""
        assert again.headers["etag"] == etag
        assert again.headers["cache-control"] == "no-cache"
        assert again.headers["last-modified"] == first.headers["last-modified"]
        head = client.head("/api/raw/seg.nii.gz", headers={"If-None-Match": etag})
        assert head.status_code == 304

    def test_raw_resends_the_file_once_the_etag_stops_matching(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """The point of revalidating: a file a tool rewrote is sent again, not withheld."""
        (workspace_root / "seg.nii.gz").write_bytes(b"v1")
        stale = client.get("/api/raw/seg.nii.gz").headers["etag"]
        (workspace_root / "seg.nii.gz").write_bytes(b"v2-longer")
        fresh = client.get("/api/raw/seg.nii.gz", headers={"If-None-Match": stale})
        assert fresh.status_code == 200
        assert fresh.content == b"v2-longer"
        assert fresh.headers["etag"] != stale

    def test_raw_reads_every_spelling_of_if_none_match(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Clients send a list, a weak tag or ``*``; each is a valid revalidation."""
        (workspace_root / "seg.nii.gz").write_bytes(b"v1")
        etag = client.get("/api/raw/seg.nii.gz").headers["etag"]
        for header in (f'"other", {etag}', f"W/{etag}", "*"):
            assert (
                client.get("/api/raw/seg.nii.gz", headers={"If-None-Match": header}).status_code
                == 304
            )
        assert (
            client.get("/api/raw/seg.nii.gz", headers={"If-None-Match": '"nope"'}).status_code
            == 200
        )

    def test_raw_streams_a_file_larger_than_one_chunk(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A volume spans several reads; no byte may be dropped or doubled at a seam."""
        blob = bytes(range(256)) * 12_000  # ~3 MB: three reads at the 1 MiB chunk size
        (workspace_root / "big.nii.gz").write_bytes(blob)
        got = client.get("/api/raw/big.nii.gz")
        assert got.status_code == 200
        assert got.content == blob
        assert got.headers["content-length"] == str(len(blob))

    def test_raw_still_serves_ranges_and_head(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """The response subclass keeps what ``FileResponse`` gave: ranges, HEAD, ``no-cache``."""
        blob = bytes(range(256)) * 8_200  # ~2 MB, so a range can straddle a chunk seam
        (workspace_root / "big.nii.gz").write_bytes(blob)
        seam = 1024 * 1024
        part = client.get("/api/raw/big.nii.gz", headers={"Range": f"bytes={seam - 10}-{seam + 9}"})
        assert part.status_code == 206
        assert part.content == blob[seam - 10 : seam + 10]
        assert part.headers["content-range"] == f"bytes {seam - 10}-{seam + 9}/{len(blob)}"
        assert part.headers["cache-control"] == "no-cache"
        tail = client.get("/api/raw/big.nii.gz", headers={"Range": "bytes=-5"})
        assert tail.status_code == 206
        assert tail.content == blob[-5:]
        head = client.head("/api/raw/big.nii.gz")
        assert head.status_code == 200
        assert head.content == b""
        assert head.headers["content-length"] == str(len(blob))
        assert head.headers["cache-control"] == "no-cache"
        assert head.headers["accept-ranges"] == "bytes"
        full = client.get("/api/raw/big.nii.gz")
        assert full.headers["cache-control"] == "no-cache"
        stale = client.head("/api/raw/big.nii.gz", headers={"If-None-Match": full.headers["etag"]})
        assert stale.status_code == 304
        assert stale.headers["cache-control"] == "no-cache"

    def test_raw_html_and_svg_are_sandboxed(self, workspace_root: Path, client: TestClient) -> None:
        """Opened directly, a workspace page must not run with the app's origin."""
        (workspace_root / "qc.html").write_text("<script>1</script>")
        (workspace_root / "seg.nii.gz").write_bytes(b"x")
        page = client.get("/api/raw/qc.html")
        assert page.headers["content-security-policy"] == "sandbox allow-scripts"
        assert page.headers["x-content-type-options"] == "nosniff"
        volume = client.get("/api/raw/seg.nii.gz")
        assert "content-security-policy" not in volume.headers
        assert volume.headers["x-content-type-options"] == "nosniff"

    def test_raw_missing_file_is_404_for_head_too(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """HEAD on an absent file is a 404 like GET, not a 405 that reads as a server fault."""
        assert client.head("/api/raw/nope.nii.gz").status_code == 404


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


class TestSymlinksInTheWorkspace:
    """A symlink is acted on as itself, and never a way to see or write outside."""

    @pytest.fixture
    def outside(self, workspace_root: Path) -> Path:
        """A folder next to the workspace holding a file."""
        outside = workspace_root.parent / f"{workspace_root.name}-outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("s")
        return outside

    def test_upload_does_not_write_through_a_link(
        self, workspace_root: Path, outside: Path, client: TestClient
    ) -> None:
        """An upload named like an existing link must not land where the link points."""
        (workspace_root / "notes.txt").symlink_to(outside / "secret.txt")
        resp = client.post("/api/files/upload", files={"file": ("notes.txt", io.BytesIO(b"pwned"))})
        assert resp.status_code == 400
        assert (outside / "secret.txt").read_text() == "s"

    def test_upload_does_not_follow_a_dangling_link(
        self, workspace_root: Path, outside: Path, client: TestClient
    ) -> None:
        """A dangling link would otherwise create its target outside the workspace."""
        (workspace_root / "new.txt").symlink_to(outside / "planted.txt")
        resp = client.post("/api/files/upload", files={"file": ("new.txt", io.BytesIO(b"x"))})
        assert resp.status_code == 400
        assert not (outside / "planted.txt").exists()

    def test_tree_does_not_list_through_an_outside_link(
        self, workspace_root: Path, outside: Path, client: TestClient
    ) -> None:
        """The explorer shows the link, but not the host folder behind it."""
        (workspace_root / "host").symlink_to(outside)
        (workspace_root / "secret").symlink_to(outside / "secret.txt")
        tree = {n["name"]: n for n in client.get("/api/tree").json()["tree"]}
        assert tree["host"]["children"] == []
        assert "size" not in tree["secret"]

    def test_tree_lists_through_an_inside_link_but_not_a_cycle(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A link to another workspace folder is browsable; one to an ancestor is not walked."""
        (workspace_root / "data").mkdir()
        (workspace_root / "data" / "a.nii").write_bytes(b"x")
        (workspace_root / "alias").symlink_to(workspace_root / "data")
        (workspace_root / "data" / "loop").symlink_to(workspace_root)
        tree = {n["name"]: n for n in client.get("/api/tree").json()["tree"]}
        assert [c["name"] for c in tree["alias"]["children"]] == ["loop", "a.nii"]
        loop = next(c for c in tree["data"]["children"] if c["name"] == "loop")
        assert loop["children"] == []

    def test_deleting_a_link_keeps_its_target(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Deleting a link to a folder removes the link, not the folder's contents."""
        (workspace_root / "data").mkdir()
        (workspace_root / "data" / "a.nii").write_bytes(b"x")
        (workspace_root / "alias").symlink_to(workspace_root / "data")
        assert client.delete("/api/files", params={"path": "alias"}).status_code == 200
        assert not (workspace_root / "alias").is_symlink()
        assert (workspace_root / "data" / "a.nii").exists()

    def test_renaming_a_link_moves_the_link(
        self, workspace_root: Path, outside: Path, client: TestClient
    ) -> None:
        """A link pointing out can be renamed or removed, but its target is never touched."""
        (workspace_root / "host").symlink_to(outside)
        resp = client.post("/api/files/rename", json={"path": "host", "new_path": "renamed"})
        assert resp.status_code == 200
        assert (workspace_root / "renamed").is_symlink()
        assert (outside / "secret.txt").exists()
        assert client.delete("/api/files", params={"path": "renamed"}).status_code == 200
        assert (outside / "secret.txt").exists()


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

        async def _no_restart(*_changed: object) -> None:
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
        assert visible[1]["name"] == "Allow reading files in this chat"

    def test_names_every_command_of_a_compound(self) -> None:
        """``ls; wc -l x`` needs two rules; both are read-only, so both are named."""
        options = self._bash_options(("command_pattern", "ls *"), ("command_pattern", "wc *"))
        visible = server._visible_permission_options(options)
        assert visible[1]["name"] == "Allow reading files in this chat"

    @pytest.mark.parametrize(
        "rules",
        [
            (("command_pattern", "rm *"),),
            (("command_pattern", "ls *"), ("command_pattern", "rm *")),
            (("command_pattern", "ls *"), ("command_pattern", "<redirect> *")),
            (("command_pattern", "find *"),),
            (("command_pattern", "tree *"),),
            (("command_pattern", "file *"),),
            (("command_pattern", "python *"),),
            (("outside_directory", "/etc/*"),),
            (("command_pattern", "ls *"), ("outside_directory", "/etc/*")),
        ],
    )
    def test_withholds_session_approval_unless_every_rule_is_read_only(
        self, rules: tuple[tuple[str, str], ...]
    ) -> None:
        """A command that can write (by any flag), a redirect or an outside path disqualifies."""
        visible = server._visible_permission_options(self._bash_options(*rules))
        assert [o["optionId"] for o in visible] == ["allow_once", "reject_once"]

    def test_find_delete_would_ride_a_find_approval(self) -> None:
        """Why ``find`` is excluded: vibe derives ``find *`` from ``find . -delete``.

        Only the ``-exec`` family gets an exact pattern, so an approved ``find *``
        rule would let ``find . -delete`` run without asking.
        """
        arity: Any = importlib.import_module("vibe.core.tools.arity")
        permissions: Any = importlib.import_module("vibe.core.tools.permissions")

        assert arity.build_session_pattern(["find", ".", "-delete"]) == "find *"
        assert permissions.wildcard_match("find . -delete", "find *")
        assert "find *" not in server._READ_ONLY_SESSION_PATTERNS

    def test_withholds_session_approval_for_a_redirect(self) -> None:
        """``ls > out.txt`` reaches us as ``ls <redirect>`` with session pattern ``ls *``."""
        options = self._bash_options(("command_pattern", "ls *"))
        meta = cast("dict[str, list[dict[str, str]]]", options[1]["_meta"])
        meta["required_permissions"][0]["invocation_pattern"] = "ls -la <redirect>"
        visible = server._visible_permission_options(options)
        assert [o["optionId"] for o in visible] == ["allow_once", "reject_once"]

    @pytest.mark.parametrize(
        "command",
        [
            "ls -R /ws/demo_data 2>&1 | head -n 200",
            "ls /ws >&2",
            "cat /ws/x.csv 2>/dev/null",
            "grep -r pattern /ws 2>>/dev/null",
        ],
    )
    def test_offers_session_approval_when_redirects_touch_no_file(self, command: str) -> None:
        """``2>&1`` or ``2>/dev/null`` writes nothing; the raw command tells it from ``> x``."""
        options = self._bash_options(("command_pattern", "ls *"))
        meta = cast("dict[str, list[dict[str, str]]]", options[1]["_meta"])
        meta["required_permissions"][0]["invocation_pattern"] = "ls -R <redirect>"
        visible = server._visible_permission_options(options, command)
        assert [o["optionId"] for o in visible] == ["allow_once", "allow_always", "reject_once"]

    @pytest.mark.parametrize(
        "command",
        [
            "ls /ws > out.txt",
            "ls /ws >> out.txt",
            "ls /ws &> out.txt",
            "ls /ws 2>&1 > out.txt",
            "ls /ws 2> err.txt",
            "cat /ws/x > /dev/null/../out",
            "cat <<EOF\nhello\nEOF",
            "ls /ws 2>&1 | head -n 200",  # a harmless redirect, but the command is unknown
        ],
    )
    def test_withholds_session_approval_when_a_redirect_may_write(self, command: str) -> None:
        """A file target, ``&>``, a heredoc, or no command text at all keeps the per-call box."""
        options = self._bash_options(("command_pattern", "ls *"))
        meta = cast("dict[str, list[dict[str, str]]]", options[1]["_meta"])
        meta["required_permissions"][0]["invocation_pattern"] = "ls <redirect>"
        unknown = command.endswith("head -n 200")
        visible = server._visible_permission_options(options, None if unknown else command)
        assert [o["optionId"] for o in visible] == ["allow_once", "reject_once"]

    def test_drops_session_scoped_options_by_kind(self) -> None:
        """The 2.26 ``allow_session_exact`` is an "always" by kind; ids are not a list we own."""
        meta = {
            "required_permissions": [
                {
                    "scope": "outside_directory",
                    "invocation_pattern": "/nonexistent/x",
                    "session_pattern": "/nonexistent/*",
                    "label": "/nonexistent/*",
                }
            ]
        }
        options: list[dict[str, object]] = [
            {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
            {
                "optionId": "allow_session_exact",
                "name": "Allow this file only for this session",
                "kind": "allow_always",
                "_meta": meta,
            },
            {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
        ]
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


class TestReadOnlyGrantForTheChat:
    """One "allow looking at files" covers every later read-only command in the chat."""

    @pytest.fixture
    def responses(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, object]]:
        """Capture what is answered to vibe; provenance and explanations off."""
        sent: list[tuple[int, object]] = []

        async def _respond(req_id: int, result: object) -> None:
            sent.append((req_id, result))

        monkeypatch.setattr(server._client, "respond", _respond)
        monkeypatch.setattr(settings, "load_provenance_enabled", lambda: False)
        monkeypatch.setattr(settings, "load_explain_enabled", lambda: False)
        monkeypatch.delenv(server.READ_ONLY_APPROVAL_ENV_VAR, raising=False)
        return sent

    @staticmethod
    def _request(req_id: int, command: str, *rules: tuple[str, str]) -> dict[str, object]:
        return {
            "id": req_id,
            "method": "session/request_permission",
            "params": {
                "toolCall": {
                    "toolCallId": f"c{req_id}",
                    "title": "bash",
                    "rawInput": {"command": command},
                },
                "options": TestReadOnlySessionApproval._bash_options(*rules),
            },
        }

    async def _answer(
        self, conn: server._ChatConnection, req: dict[str, object], option: str
    ) -> None:
        """Run the handler until its dialog is shown, then choose *option*."""
        task = asyncio.create_task(conn._handle_permission(req))
        await asyncio.sleep(0.05)
        conn._pending_perms[cast("int", req["id"])].set_result(option)
        await task

    async def test_later_read_only_commands_are_allowed_without_a_dialog(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """After approving ``ls`` for the chat, a ``grep`` goes through unasked."""
        conn, _ = _connection()
        await self._answer(
            conn, self._request(1, "ls", ("command_pattern", "ls *")), "allow_always"
        )
        await conn._handle_permission(
            self._request(2, "grep -c x a.csv", ("command_pattern", "grep *"))
        )
        assert responses[-1] == (
            2,
            {"outcome": {"outcome": "selected", "optionId": "allow_always"}},
        )
        dialogs = [
            m for m in cast("_StubWs", conn.ws).sent if m.get("type") == "permission_request"
        ]
        assert len(dialogs) == 1

    async def test_a_path_scoped_read_is_allowed_once(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """``wc -l <file>`` comes with path-scoped options only; it is allowed once."""
        conn, _ = _connection()
        await self._answer(
            conn, self._request(1, "ls", ("command_pattern", "ls *")), "allow_always"
        )
        req = self._request(2, f"wc -l {server.WORKSPACE_ROOT}/a.csv")
        cast("Any", req["params"])["options"] = [
            {"optionId": "allow_once", "name": "Allow once"},
            {"optionId": "allow_session_exact", "name": "Allow a.csv", "kind": "allow_always"},
            {"optionId": "reject_once", "name": "Deny"},
        ]
        await conn._handle_permission(req)
        assert responses[-1] == (2, {"outcome": {"outcome": "selected", "optionId": "allow_once"}})

    @pytest.mark.parametrize(
        ("command", "read_only"),
        [
            ("ls -1 {ws}/x | wc -l", True),
            ("ls -la {ws} 2>&1 | head -n 5", True),
            ("head -n 5 a.csv; grep -c x b.csv", True),
            ('cat "{ws}/a b.csv"', True),
            ("cat /etc/passwd", False),
            ('cat /etc/"passwd"', False),
            ("cat ../../etc/passwd", False),
            ("ls {ws}/../secret", False),
            ("ls ~", False),
            ("ls $HOME", False),
            ("ls > out.txt", False),
            ("ls $(rm -rf x)", False),
            ("ls `rm x`", False),
            ("cat <(rm x)", False),
            ("ls | xargs rm", False),
            ("ls && rm x", False),
            ("echo hi", False),
        ],
    )
    def test_read_only_command(self, command: str, read_only: bool, tmp_path: Path) -> None:
        """Only read-only commands on workspace paths, with nothing executed inside."""
        assert server._is_read_only_command(command.format(ws=tmp_path), tmp_path) is read_only

    async def test_a_write_still_asks(self, responses: list[tuple[int, object]]) -> None:
        """The grant never covers a command outside the read-only set."""
        conn, _ = _connection()
        await self._answer(
            conn, self._request(1, "ls", ("command_pattern", "ls *")), "allow_always"
        )
        task = asyncio.create_task(
            conn._handle_permission(self._request(2, "rm a.csv", ("command_pattern", "rm *")))
        )
        await asyncio.sleep(0.05)
        dialogs = [
            m for m in cast("_StubWs", conn.ws).sent if m.get("type") == "permission_request"
        ]
        assert len(dialogs) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_allow_once_grants_nothing_more(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """Plain "Allow" on a read-only command leaves the next one asking."""
        conn, _ = _connection()
        await self._answer(conn, self._request(1, "ls", ("command_pattern", "ls *")), "allow_once")
        task = asyncio.create_task(
            conn._handle_permission(self._request(2, "ls", ("command_pattern", "ls *")))
        )
        await asyncio.sleep(0.05)
        dialogs = [
            m for m in cast("_StubWs", conn.ws).sent if m.get("type") == "permission_request"
        ]
        assert len(dialogs) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


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
        (tmp_path / "s1" / "run.jsonl").write_text("{}\n")

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

        # The test client's loop ends with the socket; keep record-keeping on it.
        async def inline(fn: Callable[[], object]) -> object:
            return fn()

        def no_image(image: str) -> dict[str, Any]:
            return {"digest": "sha256:" + "ab" * 32, "id": None}

        monkeypatch.setattr(runs, "_to_thread", inline)
        monkeypatch.setattr(settings, "image_identity", no_image)
        # A fresh grant store per test: no token outlives the test that got it.
        from medmcp import previewgrant

        monkeypatch.setattr(server, "_preview_grants", previewgrant.PreviewGrants())
        return workspace

    @staticmethod
    def _token(client: TestClient, name: str, bindings: list[dict[str, str]]) -> str:
        """Preview *bindings* and return the token that may run them."""
        body = client.post(f"/api/workflows/{name}/replay-preview", json={"runs": bindings}).json()
        assert body["ok"] is True and isinstance(body["token"], str)
        return body["token"]

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
        token = self._token(client, "strip", [{"in_1": "t1.nii.gz"}])
        with client.websocket_connect("/ws/replay", headers=WS_HEADERS) as ws:
            ws.send_json({"name": "strip", "runs": [{"in_1": "t1.nii.gz"}], "preview_token": token})
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

    def _import_reorient(self, client: TestClient) -> None:
        """A workflow that reads ``in_1`` and writes the file ``in_2``."""
        envelope = yaml.safe_dump(
            {
                "medmcp_workflow": 1,
                "name": "reorient",
                "description": "demo",
                "inputs": [
                    {"name": "in_1", "example": "/d/p1/consensus_gt.nii.gz"},
                    {"name": "in_2", "example": "/d/out/consensus_gt_RAS.nii.gz"},
                ],
                "steps": [
                    {
                        "server": "medmcp-neuro",
                        "tool": "reorient",
                        "arguments": {"input_path": "{{in_1}}", "output_path": "{{in_2}}"},
                    }
                ],
            }
        )
        assert client.post("/api/workflows/import", json={"content": envelope}).status_code == 200

    def test_preview_refuses_an_output_on_its_input(self, env: Path, client: TestClient) -> None:
        """Output bound to the input file: the item is refused with the reason."""
        self._import_reorient(client)
        (env / "b.nii.gz").write_bytes(b"x")
        resp = client.post(
            "/api/workflows/reorient/replay-preview",
            json={
                "runs": [
                    {"in_1": "t1.nii.gz", "in_2": "t1.nii.gz"},
                    {"in_1": "b.nii.gz", "in_2": "new/deep/b_RAS.nii.gz"},
                ]
            },
        )
        items = resp.json()["items"]
        assert [i["ok"] for i in items] == [False, True]
        assert "overwrite its own input" in items[0]["error"]
        # The output folder does not exist yet: that is the run's job, not an error.
        out = next(f for f in items[1]["findings"] if f["role"] == "output")
        assert out["severity"] == "info" and "created" in out["note"]

    def test_preview_refuses_two_items_with_one_output(self, env: Path, client: TestClient) -> None:
        """Both rows that would write the same file are refused, each naming the other."""
        self._import_reorient(client)
        (env / "b.nii.gz").write_bytes(b"x")
        resp = client.post(
            "/api/workflows/reorient/replay-preview",
            json={
                "runs": [
                    {"in_1": "t1.nii.gz", "in_2": "out/r.nii.gz"},
                    {"in_1": "b.nii.gz", "in_2": "out/r.nii.gz"},
                ]
            },
        )
        body = resp.json()
        assert body["ok"] is False
        assert "item 2" in body["items"][0]["error"] and "item 1" in body["items"][1]["error"]

    def test_batch_from_plan_places_outputs_under_a_root(
        self, env: Path, client: TestClient
    ) -> None:
        """The plan's file column feeds the input; the output gets a folder per subject."""
        self._import_reorient(client)
        for subject in ("p1", "p2"):
            (env / subject).mkdir()
            (env / subject / "consensus_gt.nii.gz").write_bytes(b"x")
        plan = env / "batch_plan.csv"
        plan.write_text(
            "subject,consensus_gt,status,reason\n"
            f"1,{env}/p1/consensus_gt.nii.gz,ok,\n"
            f"2,{env}/p2/consensus_gt.nii.gz,ok,\n"
        )
        body = client.post(
            "/api/workflows/reorient/batch-from-plan", json={"plan_csv": "batch_plan.csv"}
        ).json()
        assert body["ok"] is True
        assert body["column_map"] == {"in_1": "consensus_gt"}
        assert body["labels"] == ["1", "2"]
        assert body["output_root"] == str(env / "results" / "reorient")
        assert [r["in_2"] for r in body["runs"]] == [
            str(env / "results" / "reorient" / s / "consensus_gt_RAS.nii.gz") for s in ("1", "2")
        ]
        chosen = client.post(
            "/api/workflows/reorient/batch-from-plan",
            json={"plan_csv": "batch_plan.csv", "output_root": "mine"},
        ).json()
        assert chosen["runs"][0]["in_2"] == str(env / "mine" / "1" / "consensus_gt_RAS.nii.gz")
        outside = client.post(
            "/api/workflows/reorient/batch-from-plan",
            json={"plan_csv": "batch_plan.csv", "output_root": "/etc"},
        )
        assert outside.status_code == 400

    def test_batch_from_plan_names_the_columns_when_it_cannot_map(
        self, env: Path, client: TestClient
    ) -> None:
        """An unmatched input comes back with the columns to choose from, and a map fixes it."""
        self._import_reorient(client)
        plan = env / "batch_plan.csv"
        plan.write_text(
            f"subject,mask,lesions,status,reason\n1,{env}/t1.nii.gz,{env}/x.nii.gz,ok,\n"
        )
        body = client.post(
            "/api/workflows/reorient/batch-from-plan", json={"plan_csv": "batch_plan.csv"}
        ).json()
        assert body["ok"] is False
        assert body["columns"] == ["mask", "lesions"] and body["unmapped"] == ["in_1"]
        fixed = client.post(
            "/api/workflows/reorient/batch-from-plan",
            json={"plan_csv": "batch_plan.csv", "column_map": {"in_1": "mask"}},
        ).json()
        assert fixed["ok"] is True and fixed["runs"][0]["in_1"] == str(env / "t1.nii.gz")

    def test_run_record_is_self_contained_and_exportable(
        self, env: Path, client: TestClient
    ) -> None:
        """Labels, manifest copy, environment, results and confirmation reach the record."""
        (env / "sel.csv").write_text("subject_id,path\nP1,t1.nii.gz\n")
        (env / "brain.nii.gz").write_bytes(b"brain")
        bindings = [{"in_1": "t1.nii.gz"}]
        token = self._token(client, "strip", bindings)
        with client.websocket_connect("/ws/replay", headers=WS_HEADERS) as ws:
            ws.send_json(
                {
                    "name": "strip",
                    "runs": bindings,
                    "labels": ["P1/ses-1"],
                    "manifest": "sel.csv",
                    "preview_token": token,
                }
            )
            frames: list[dict[str, Any]] = []
            while not frames or frames[-1]["type"] != "result":
                frames.append(ws.receive_json())
        assert frames[0]["labels"] == ["P1/ses-1"]
        run_id = frames[0]["run_id"]

        record = client.get(f"/api/runs/{run_id}").json()
        assert record["items"][0]["label"] == "P1/ses-1"
        step = record["items"][0]["steps"][0]
        assert step["result"] == {"brain_path": str(env / "brain.nii.gz")}
        assert record["recipe"]["steps"][0]["tool"] == "skull_strip"
        assert record["confirmation"]["previewed"] is True
        assert record["confirmation"]["previewed_at"]
        assert record["manifest"]["copy"] == "manifest.csv"
        assert record["environment"]["medmcp"]["version"]

        exported = client.get(f"/api/runs/{run_id}/export")
        assert exported.headers["content-disposition"].endswith(f'run-{run_id}.json"')
        assert exported.json()["manifest"]["text"].startswith("subject_id,path")
        report = client.get(f"/api/runs/{run_id}/export", params={"format": "md"})
        assert "# MedMCP run report" in report.text and "P1/ses-1" in report.text
        assert client.get("/api/runs/nope/export").status_code == 404

    def _count_calls(self, env: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Re-patch the engine's tool caller to record every call it is asked to make."""
        import contextlib
        from collections.abc import AsyncGenerator

        from medmcp import replay

        calls: list[str] = []

        @contextlib.asynccontextmanager
        async def counting(*_a: object, **_k: object) -> AsyncGenerator[replay.ToolCaller]:
            async def _call(
                srv: str, tool: str, args: dict[str, Any]
            ) -> tuple[bool, dict[str, Any], str | None]:
                calls.append(tool)
                return True, {"brain_path": str(env / "brain.nii.gz")}, None

            yield _call

        monkeypatch.setattr(replay, "mcp_caller", counting)
        return calls

    def _refused(
        self, client: TestClient, message: dict[str, Any], calls: list[str], reason: str
    ) -> None:
        """One refusal naming *reason* answers the start; nothing ran, nothing was recorded."""
        frames = self._run(client, message)
        assert frames == [{"type": "result", "ok": False, "error": frames[0]["error"]}]
        assert reason in frames[0]["error"]
        assert calls == []
        assert client.get("/api/runs").json()["runs"] == []

    def test_a_start_without_a_preview_is_refused(
        self, env: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No token: no run, no record, and the refusal is in the audit trail."""
        calls = self._count_calls(env, monkeypatch)
        audit: list[logging.LogRecord] = []

        class Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                audit.append(record)

        handler = Collect(logging.WARNING)
        server._audit.addHandler(handler)
        try:
            message: dict[str, Any] = {"name": "strip", "runs": [{"in_1": "t1.nii.gz"}]}
            self._refused(client, message, calls, "not previewed")
            self._refused(client, {**message, "preview_token": "made-up"}, calls, "unknown")
        finally:
            server._audit.removeHandler(handler)
        lines = [r.getMessage() for r in audit if "replay refused" in r.getMessage()]
        assert len(lines) == 2
        assert "strip" in lines[0] and "not previewed" in lines[0] and "unknown" in lines[1]

    def test_a_token_runs_only_what_its_preview_showed(
        self, env: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Other rows, another workflow, or a recipe changed on disk: refused."""
        calls = self._count_calls(env, monkeypatch)
        self._import_reorient(client)
        (env / "b.nii.gz").write_bytes(b"x")
        shown = [{"in_1": "t1.nii.gz"}]

        other_rows = [{"in_1": "b.nii.gz"}]
        token = self._token(client, "strip", shown)
        message = {"name": "strip", "runs": other_rows, "preview_token": token}
        self._refused(client, message, calls, "differ from the ones the preview showed")
        # One more row than was shown is other rows too.
        token = self._token(client, "strip", shown)
        message = {"name": "strip", "runs": shown + other_rows, "preview_token": token}
        self._refused(client, message, calls, "differ from the ones the preview showed")

        token = self._token(client, "strip", shown)
        rows = [{"in_1": "t1.nii.gz", "in_2": "out/t1_RAS.nii.gz"}]
        message = {"name": "reorient", "runs": rows, "preview_token": token}
        self._refused(client, message, calls, "was for workflow 'strip'")

        token = self._token(client, "strip", shown)
        recipe_file = provenance.VIBE_HOME / "workflows" / "strip" / "recipe.yaml"
        assert recipe_file.is_file()
        recipe_file.write_text(recipe_file.read_text().replace("skull_strip", "bet"))
        message = {"name": "strip", "runs": shown, "preview_token": token}
        self._refused(client, message, calls, "changed after the preview")

    def test_a_token_works_once(
        self, env: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The previewed run starts, with a record saying so; the same token again does not."""
        calls = self._count_calls(env, monkeypatch)
        shown = [{"in_1": "t1.nii.gz"}]
        token = self._token(client, "strip", shown)
        message = {"name": "strip", "runs": shown, "preview_token": token}
        frames = self._run(client, message)
        assert frames[0]["type"] == "started" and frames[-1]["ok"] is True
        assert calls == ["skull_strip"]
        record = client.get(f"/api/runs/{frames[0]['run_id']}").json()
        assert record["confirmation"]["previewed"] is True

        again = self._run(client, message)
        assert again == [{"type": "result", "ok": False, "error": again[0]["error"]}]
        assert "already used" in again[0]["error"]
        assert calls == ["skull_strip"]

    def test_an_expired_token_is_refused(
        self, env: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A preview older than its lifetime no longer starts anything."""
        from medmcp import previewgrant

        calls = self._count_calls(env, monkeypatch)
        clock = [1000.0]
        grants = previewgrant.PreviewGrants(ttl_s=60, clock=lambda: clock[0])
        monkeypatch.setattr(server, "_preview_grants", grants)
        shown = [{"in_1": "t1.nii.gz"}]
        body = client.post("/api/workflows/strip/replay-preview", json={"runs": shown}).json()
        assert body["expires_in"] == 60
        clock[0] += 61
        message = {"name": "strip", "runs": shown, "preview_token": body["token"]}
        self._refused(client, message, calls, "expired")

    def test_a_batch_runs_the_rows_that_passed_its_preview(
        self, env: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The token admits the rows the preview found runnable: those run, all of them."""
        calls = self._count_calls(env, monkeypatch)
        (env / "b.nii.gz").write_bytes(b"x")
        rows = [{"in_1": "t1.nii.gz"}, {"in_1": "typo.nii.gz"}, {"in_1": "b.nii.gz"}]
        body = client.post("/api/workflows/strip/replay-preview", json={"runs": rows}).json()
        assert [i["ok"] for i in body["items"]] == [True, False, True]
        message = {"name": "strip", "runs": rows, "preview_token": body["token"]}
        self._refused(client, message, calls, "differ from the ones the preview showed")

        body = client.post("/api/workflows/strip/replay-preview", json={"runs": rows}).json()
        message = {"name": "strip", "runs": [rows[0], rows[2]], "preview_token": body["token"]}
        frames = self._run(client, message)
        assert frames[-1]["ok"] is True
        assert [f["ok"] for f in frames if f["type"] == "item_result"] == [True, True]
        assert calls == ["skull_strip", "skull_strip"]

    def test_a_preview_with_nothing_runnable_gives_no_token(
        self, env: Path, client: TestClient
    ) -> None:
        """Every row failed its checks: there is nothing a token could start."""
        body = client.post(
            "/api/workflows/strip/replay-preview", json={"runs": [{"in_1": "typo.nii.gz"}]}
        ).json()
        assert body["ok"] is False and body["token"] is None

    def test_a_selection_manifest_feeds_the_roll_out(self, env: Path, client: TestClient) -> None:
        """One row per selected series: the single input takes ``path``, no status needed."""
        for subject in ("P1", "P2"):
            (env / subject).mkdir()
            (env / subject / "FLAIR.nii.gz").write_bytes(b"x")
        (env / ".medmcp").mkdir()
        (env / ".medmcp" / "sel_series.csv").write_text(
            "subject_id,session_id,sequence,contrast,plane,path\n"
            f"P1,ses-1,FLAIR,no,axial,{env}/P1/FLAIR.nii.gz\n"
            "P2,ses-1,FLAIR,no,axial,P2/FLAIR.nii.gz\n"
        )
        body = client.post(
            "/api/workflows/strip/batch-from-plan", json={"plan_csv": ".medmcp/sel_series.csv"}
        ).json()
        assert body["ok"] is True and body["kind"] == "selection"
        assert body["labels"] == ["P1/ses-1", "P2/ses-1"]
        assert [r["in_1"] for r in body["runs"]] == [
            str(env / "P1" / "FLAIR.nii.gz"),
            str(env / "P2" / "FLAIR.nii.gz"),
        ]
        assert body["roles"] == [{"role": "FLAIR-contrast", "series": 2, "units": 2}]

    def _import_lesions(self, client: TestClient) -> None:
        """A workflow reading a T1, a T1 with contrast and a FLAIR, writing one file."""
        envelope = yaml.safe_dump(
            {
                "medmcp_workflow": 1,
                "name": "lesions",
                "description": "demo",
                "inputs": [
                    {"name": "in_1", "example": "/d/T1.nii.gz"},
                    {"name": "in_2", "example": "/d/T1KM.nii.gz"},
                    {"name": "in_3", "example": "/d/FLAIR.nii.gz"},
                    {"name": "in_4", "example": "/d/out/lesions.nii.gz"},
                ],
                "steps": [
                    {
                        "server": "medmcp-neuro",
                        "tool": "segment_lesions",
                        "arguments": {
                            "t1_path": "{{in_1}}",
                            "t1ce_path": "{{in_2}}",
                            "flair_path": "{{in_3}}",
                            "output_path": "{{in_4}}",
                        },
                    }
                ],
            }
        )
        assert client.post("/api/workflows/import", json={"content": envelope}).status_code == 200

    def _study(self, env: Path) -> Path:
        """The cohort tools' confirmed selection (tests/fixtures/confirmed) in a study folder.

        Every series it names exists, so the preview finds the files.
        """
        folder = env / "study" / ".medmcp"
        folder.mkdir(parents=True)
        fixtures = Path(__file__).parent / "fixtures" / "confirmed"
        for name in ("lesions_series.csv", "lesions_confirmed.csv", "lesions_confirmed.json"):
            (folder / name).write_bytes((fixtures / name).read_bytes())
        for line in (folder / "lesions_series.csv").read_text().splitlines()[1:]:
            scan = env / "study" / line.split(",")[11]
            scan.parent.mkdir(parents=True, exist_ok=True)
            scan.write_bytes(b"x")
        return folder

    ROLES: ClassVar[dict[str, str]] = {
        "in_1": "T1-contrast",
        "in_2": "T1+contrast",
        "in_3": "FLAIR",
    }

    def test_a_plain_selection_is_refused_for_several_inputs(
        self, env: Path, client: TestClient
    ) -> None:
        """No roles are asked for: the answer says to review and confirm the selection."""
        self._import_lesions(client)
        self._study(env)
        for role_map in ({}, self.ROLES):
            body = client.post(
                "/api/workflows/lesions/batch-from-plan",
                json={"plan_csv": "study/.medmcp/lesions_series.csv", "role_map": role_map},
            ).json()
            assert body["ok"] is False and body["refused"] is True
            assert "reviewed and confirmed first" in body["error"]
            assert "unmapped" not in body
        # One input is never ambiguous: the same selection still feeds it, a run per series.
        one = client.post(
            "/api/workflows/strip/batch-from-plan",
            json={"plan_csv": "study/.medmcp/lesions_series.csv"},
        ).json()
        assert one["ok"] is True and len(one["runs"]) == 10 and one["confirmation"] is None

    def test_a_confirmed_selection_asks_for_its_roles_then_binds(
        self, env: Path, client: TestClient
    ) -> None:
        """The roles offered are the confirmed labels; the answer carries the confirmation."""
        self._import_lesions(client)
        folder = self._study(env)
        plan = {"plan_csv": "study/.medmcp/lesions_confirmed.csv"}
        asked = client.post("/api/workflows/lesions/batch-from-plan", json=plan).json()
        assert asked["ok"] is False and asked["unmapped"] == ["in_1", "in_2", "in_3"]
        assert [r["role"] for r in asked["roles"]] == ["T1-contrast", "T1+contrast", "FLAIR"]
        assert asked["roles"][0] == {"role": "T1-contrast", "series": 2, "units": 2}

        bound = client.post(
            "/api/workflows/lesions/batch-from-plan", json={**plan, "role_map": self.ROLES}
        ).json()
        assert bound["ok"] is True and bound["labels"] == ["P01/ses-01", "P02/ses-01"]
        assert bound["runs"][1]["in_1"] == str(env / "study" / "P02" / "ses-01" / "s06.nii.gz")
        assert bound["runs"][1]["in_4"] == str(
            env / "results" / "lesions" / "P02" / "ses-01" / "lesions.nii.gz"
        )
        info = bound["confirmation"]
        assert info["rule_version"] == "2" and info["overrides"] == 1 and info["excluded"] == 1
        assert info["confirmed_at"] and info["path"] == str(folder / "lesions_confirmed.json")

        csv_path = folder / "lesions_confirmed.csv"
        csv_path.write_text(csv_path.read_text().replace("s06.nii.gz", "s05.nii.gz"))
        refused = client.post(
            "/api/workflows/lesions/batch-from-plan", json={**plan, "role_map": self.ROLES}
        ).json()
        assert refused["ok"] is False and refused["refused"] is True
        assert "the confirmed selection was changed after confirmation" in refused["error"]

    def _run(self, client: TestClient, message: dict[str, Any]) -> list[dict[str, Any]]:
        with client.websocket_connect("/ws/replay", headers=WS_HEADERS) as ws:
            ws.send_json(message)
            frames: list[dict[str, Any]] = []
            while not frames or frames[-1]["type"] != "result":
                frames.append(ws.receive_json())
        return frames

    def test_a_confirmed_roll_out_records_its_confirmation(
        self, env: Path, client: TestClient
    ) -> None:
        """The run cites the record by path and hash, with the source manifest and index."""
        from medmcp import runs

        self._import_lesions(client)
        folder = self._study(env)
        manifest = "study/.medmcp/lesions_confirmed.csv"
        bound = client.post(
            "/api/workflows/lesions/batch-from-plan",
            json={"plan_csv": manifest, "role_map": self.ROLES},
        ).json()
        frames = self._run(
            client,
            {
                "name": "lesions",
                "runs": bound["runs"],
                "labels": bound["labels"],
                "manifest": manifest,
                "preview_token": self._token(client, "lesions", bound["runs"]),
            },
        )
        assert frames[-1]["ok"] is True
        run_id = frames[0]["run_id"]

        record = client.get(f"/api/runs/{run_id}").json()
        selection = record["manifest"]["confirmed_selection"]
        record_file = folder / "lesions_confirmed.json"
        csv_bytes = (folder / "lesions_confirmed.csv").read_bytes()
        assert selection["path"] == str(record_file)
        assert selection["sha256"] == hashlib.sha256(record_file.read_bytes()).hexdigest()
        assert selection["confirmed_csv_sha256"] == hashlib.sha256(csv_bytes).hexdigest()
        assert record["manifest"]["sha256"] == selection["confirmed_csv_sha256"]
        assert (
            selection["source_manifest"]["sha256"]
            == hashlib.sha256((folder / "lesions_series.csv").read_bytes()).hexdigest()
        )
        assert selection["index"]["sha256"]
        assert selection["copy"] == "confirmation.json" and "error" not in selection
        copied = runs.run_folder(run_id) / "confirmation.json"
        assert copied.read_bytes() == record_file.read_bytes()

        exported = client.get(f"/api/runs/{run_id}/export").json()
        assert (
            exported["manifest"]["confirmed_selection"]["record"]["counts"]["overrides_applied"]
            == 1
        )
        report = client.get(f"/api/runs/{run_id}/export", params={"format": "md"}).text
        assert "**Confirmed selection:**" in report and selection["sha256"] in report

    def test_a_run_rechecks_the_manifest_at_its_start(self, env: Path, client: TestClient) -> None:
        """Edited after binding, or a plain selection under several inputs: no run starts."""
        self._import_lesions(client)
        folder = self._study(env)
        manifest = "study/.medmcp/lesions_confirmed.csv"
        bound = client.post(
            "/api/workflows/lesions/batch-from-plan",
            json={"plan_csv": manifest, "role_map": self.ROLES},
        ).json()
        csv_path = folder / "lesions_confirmed.csv"
        csv_path.write_bytes(csv_path.read_bytes() + b"\n")
        message = {"name": "lesions", "runs": bound["runs"], "manifest": manifest}
        frames = self._run(
            client, {**message, "preview_token": self._token(client, "lesions", bound["runs"])}
        )
        assert frames == [
            {"type": "result", "ok": False, "error": frames[0]["error"]},
        ]
        assert "the confirmed selection was changed after confirmation" in frames[0]["error"]

        csv_path.write_bytes(csv_path.read_bytes()[:-1])  # as confirmed again
        edited = [dict(r) for r in bound["runs"]]
        edited[1]["in_1"] = str(env / "study" / "P02" / "ses-01" / "s05.nii.gz")
        frames = self._run(
            client,
            {
                **message,
                "runs": edited,
                "preview_token": self._token(client, "lesions", edited),
            },
        )
        assert frames[0]["ok"] is False
        assert "item 2 reads" in frames[0]["error"] and "not a series of" in frames[0]["error"]

        plain = self._run(
            client,
            {
                **message,
                "manifest": "study/.medmcp/lesions_series.csv",
                "preview_token": self._token(client, "lesions", bound["runs"]),
            },
        )
        assert plain[0]["ok"] is False and "reviewed and confirmed first" in plain[0]["error"]
        assert client.get("/api/runs", params={"workflow": "lesions"}).json()["runs"] == []

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
        return TestClient(server.app, base_url="http://127.0.0.1:8100", headers=AUTH_HEADERS)

    def test_upload_recreates_nested_folders(self, root: Path, client: TestClient) -> None:
        """A folder upload sends each file with its relative folder as ``dir``."""
        resp = client.post(
            "/api/files/upload",
            params={"dir": "case1/series2"},
            files={"file": ("IM0001.dcm", b"dicom-bytes")},
        )
        assert resp.status_code == 200
        assert (root / "case1" / "series2" / "IM0001.dcm").read_bytes() == b"dicom-bytes"

    def test_upload_over_a_folder_is_a_conflict(self, root: Path, client: TestClient) -> None:
        """A file named like an existing folder is refused with a reason, not a 500."""
        (root / "scans").mkdir()
        resp = client.post("/api/files/upload", files={"file": ("scans", b"x")})
        assert resp.status_code == 409
        assert "folder" in resp.json()["detail"]
        assert (root / "scans").is_dir()

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


class _FakeAcpClient:
    """Records the JSON-RPC traffic a _ChatConnection sends to vibe-acp.

    ``session/prompt`` stays pending until ``session/cancel`` arrives (after
    ``cancel_delay``), like vibe-acp ending a turn asynchronously; ``hang`` keeps
    it pending for good.
    """

    def __init__(self, *, cancel_delay: float = 0.05, hang: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.cancel_delay = cancel_delay
        self.hang = hang
        self._turn: asyncio.Future[dict[str, object]] | None = None
        # Set to a connection's queue to act like vibe-acp taking an approval's
        # answer in: ``answer_delay`` later the turn's next frame arrives there.
        self.frames: asyncio.Queue[dict[str, object]] | None = None
        self.answer_delay = 0.05

    async def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        del params
        self.calls.append(("request", method))
        if method == "session/prompt":
            self._turn = asyncio.get_running_loop().create_future()
            result = await self._turn
            self.calls.append(("turn-ended", method))
            return result
        return {"result": {}}

    async def notify(self, method: str, params: dict[str, object]) -> None:
        del params
        self.calls.append(("notify", method))
        if method == "session/cancel" and not self.hang:
            turn = self._turn
            loop = asyncio.get_running_loop()
            loop.call_later(
                self.cancel_delay,
                lambda: (
                    turn is not None
                    and not turn.done()
                    and turn.set_result({"result": {"stopReason": "cancelled"}})
                ),
            )

    async def respond(self, req_id: int, result: dict[str, object]) -> None:
        del req_id
        outcome = cast("dict[str, object]", result.get("outcome") or {})
        self.calls.append(("respond", str(outcome.get("outcome"))))
        frames = self.frames
        if frames is not None:
            asyncio.get_running_loop().call_later(self.answer_delay, self._take_answer, frames)

    def _take_answer(self, frames: asyncio.Queue[dict[str, object]]) -> None:
        self.calls.append(("answer-taken", ""))
        frames.put_nowait(
            {
                "method": "session/update",
                "params": {"update": {"sessionUpdate": "tool_call_update", "status": "failed"}},
            }
        )

    def unregister_session(self, session_id: str) -> None:
        del session_id
        self.calls.append(("unregister", ""))


class TestChatSessionLifecycle:
    """Cancelled turns really end, and a closed chat can be reopened."""

    @pytest.fixture
    def fake(self, monkeypatch: pytest.MonkeyPatch) -> _FakeAcpClient:
        """Swap vibe-acp for the recorder; keep provenance writes out of the way."""
        client = _FakeAcpClient()
        monkeypatch.setattr(server, "_client", client)
        monkeypatch.setattr(server.settings, "load_provenance_enabled", lambda: False)
        return client

    @staticmethod
    def _conn() -> server._ChatConnection:
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        return server._ChatConnection(cast("Any", _StubWs()), "sid", cast("Any", queue), [])

    @pytest.mark.asyncio
    async def test_cancel_waits_until_vibe_ends_the_turn(self, fake: _FakeAcpClient) -> None:
        """The next prompt may only go out after the old session/prompt returned."""
        conn = self._conn()
        conn._prompt_task = asyncio.create_task(conn._run_prompt("first"))
        await asyncio.sleep(0.01)
        await conn._cancel_prompt()
        assert fake.calls.index(("notify", "session/cancel")) < fake.calls.index(
            ("turn-ended", "session/prompt")
        )
        assert fake.calls[-1] == ("turn-ended", "session/prompt")

    @pytest.mark.asyncio
    async def test_cancel_gives_up_on_a_turn_that_never_ends(
        self, fake: _FakeAcpClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wedged vibe-acp delays the next prompt by the timeout, not forever."""
        fake.hang = True
        monkeypatch.setattr(server, "TURN_CANCEL_TIMEOUT", 0.05)
        conn = self._conn()
        conn._prompt_task = asyncio.create_task(conn._run_prompt("first"))
        await asyncio.sleep(0.01)
        await asyncio.wait_for(conn._cancel_prompt(), timeout=1.0)
        assert conn._prompt_rpc is None

    @pytest.mark.asyncio
    async def test_close_releases_the_session_in_vibe(self, fake: _FakeAcpClient) -> None:
        """Leaving a chat sends session/close, so coming back can load it again."""
        conn = self._conn()
        await conn.close()
        assert ("request", "session/close") in fake.calls
        assert fake.calls.index(("request", "session/close")) < fake.calls.index(("unregister", ""))

    @pytest.mark.asyncio
    async def test_close_mid_turn_ends_the_turn_before_closing(self, fake: _FakeAcpClient) -> None:
        """A disconnect during a reply cancels it, waits for the end, then closes."""
        conn = self._conn()
        conn._prompt_task = asyncio.create_task(conn._run_prompt("first"))
        await asyncio.sleep(0.01)
        await conn.close()
        order = [c for c in fake.calls if c[0] != "unregister"]
        assert order[-3:] == [
            ("notify", "session/cancel"),
            ("turn-ended", "session/prompt"),
            ("request", "session/close"),
        ]


class TestSessionRaceRetry:
    """Opening a chat waits out vibe 2.26.1's post-close race, and only that."""

    RACE: ClassVar[dict[str, object]] = {
        "error": {"message": "The child Runtime access method is not implemented yet."}
    }
    OK: ClassVar[dict[str, object]] = {"result": {"sessionId": "s1"}}

    @pytest.fixture
    def waits(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Record the pauses between attempts instead of sleeping through them."""
        slept: list[float] = []

        async def _sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(server.asyncio, "sleep", _sleep)
        return slept

    def _answers(self, monkeypatch: pytest.MonkeyPatch, races: int) -> list[str]:
        """Let vibe answer *races* times with the race error, then succeed."""
        calls: list[str] = []

        async def _request(method: str, params: object) -> dict[str, object]:
            del params
            calls.append(method)
            return self.RACE if len(calls) <= races else self.OK

        monkeypatch.setattr(server._client, "request", _request)
        return calls

    @pytest.mark.asyncio
    async def test_a_race_that_clears_on_the_sixth_attempt_opens_the_chat(
        self, monkeypatch: pytest.MonkeyPatch, waits: list[float]
    ) -> None:
        """The comparison saw races last to the fourth try; a longer one still opens."""
        calls = self._answers(monkeypatch, races=5)
        resp = await server._session_request("session/new", {})
        assert resp == self.OK
        assert len(calls) == 6
        assert sum(waits) < 3.5

    @pytest.mark.asyncio
    async def test_a_race_that_never_clears_gives_up_with_the_error(
        self, monkeypatch: pytest.MonkeyPatch, waits: list[float]
    ) -> None:
        """A permanent race ends after a bounded wait, returning vibe's error."""
        calls = self._answers(monkeypatch, races=1000)
        resp = await server._session_request("session/load", {})
        assert resp == self.RACE
        assert len(calls) == 8
        assert sum(waits) <= 6.0

    @pytest.mark.asyncio
    async def test_any_other_error_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, waits: list[float]
    ) -> None:
        """Only the race is waited out; a real failure comes back at once."""
        calls: list[str] = []

        async def _request(method: str, params: object) -> dict[str, object]:
            del params
            calls.append(method)
            return {"error": {"message": "Session not found"}}

        monkeypatch.setattr(server._client, "request", _request)
        resp = await server._session_request("session/load", {})
        assert resp == {"error": {"message": "Session not found"}}
        assert calls == ["session/load"]
        assert waits == []


class TestStackChangeForgetsToolLists:
    """A stack change clears vibe's cached tool list for that stack while vibe is down."""

    @pytest.fixture
    def events(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, object]]:
        """Record vibe-acp's stop and every cache clear, in order."""
        seen: list[tuple[str, object]] = []

        async def _stop() -> None:
            seen.append(("stop", None))

        def _clear(names: Iterable[str], root: Path | None = None) -> int:
            del root
            seen.append(("clear", set(names)))
            return 0

        monkeypatch.setattr(server._client, "stop", _stop)
        monkeypatch.setattr(settings, "clear_tool_cache", _clear)
        monkeypatch.setattr(server, "_apply_stack_change", lambda: None)
        return seen

    def test_install_clears_the_installed_stack(
        self, events: list[tuple[str, object]], client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A (re)install drops the stack's cached tools after vibe-acp stopped."""

        def _install(image: str, *_args: object, **_kwargs: object) -> str:
            del image
            return "medmcp-neuro"

        monkeypatch.setattr(settings, "install_stack_image", _install)
        resp = client.post("/api/stacks/install", json={"image": "ghcr.io/medmcp/neuro:main"})
        assert resp.status_code == 200
        assert events == [("stop", None), ("clear", {"medmcp-neuro"})]

    def test_streamed_install_clears_the_installed_stack(
        self, events: list[tuple[str, object]], client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Tool stacks window's install path does the same."""

        def _install(image: str, *_args: object, **_kwargs: object) -> str:
            del image
            return "medmcp-neuro"

        monkeypatch.setattr(settings, "install_stack_image", _install)
        with client.websocket_connect("/ws/stacks/install", headers=WS_HEADERS) as ws:
            ws.send_json({"image": "ghcr.io/medmcp/neuro:main"})
            frame = ws.receive_json()
        assert frame == {"type": "done", "name": "medmcp-neuro"}
        assert events == [("stop", None), ("clear", {"medmcp-neuro"})]

    def test_uninstall_clears_the_removed_stack(
        self, events: list[tuple[str, object]], client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stack installed again later must not come back with the old list."""

        def _uninstall(_name: str) -> None:
            return None

        monkeypatch.setattr(settings, "uninstall_stack", _uninstall)
        resp = client.post("/api/stacks/uninstall", json={"name": "medmcp-neuro"})
        assert resp.status_code == 200
        assert events == [("stop", None), ("clear", {"medmcp-neuro"})]

    def test_toggling_stacks_clears_exactly_those(
        self, events: list[tuple[str, object]], client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Switching stacks on or off clears those, and leaves the others cached."""
        saved: list[set[str]] = []

        def _save(names: Iterable[str]) -> None:
            saved.append(set(names))

        def _noop(*_args: object) -> None:
            return None

        monkeypatch.setattr(settings, "load_active_server_names", lambda: {"alpha", "beta"})
        monkeypatch.setattr(settings, "save_active_server_names", _save)
        monkeypatch.setattr(settings, "save_explain_enabled", _noop)
        monkeypatch.setattr(settings, "save_provenance_enabled", _noop)
        monkeypatch.setattr(settings, "sync_servers_to_vibe_config", _noop)
        monkeypatch.setattr(settings, "active_servers", list)
        stacks = [
            {"name": "alpha", "active": False},
            {"name": "beta", "active": True},
            {"name": "gamma", "active": True},
        ]
        body = {"explain_tools": False, "record_provenance": False, "stacks": stacks}
        resp = client.put("/api/settings", json=body)
        assert resp.status_code == 200
        assert saved == [{"beta", "gamma"}]
        assert events == [("stop", None), ("clear", {"alpha", "gamma"})]

    def test_other_restarts_leave_the_cache_alone(self, events: list[tuple[str, object]]) -> None:
        """A model or external-server change restarts vibe without clearing anything."""
        asyncio.run(server._restart_vibe())
        assert events == [("stop", None)]


def _approval_request(req_id: int) -> dict[str, object]:
    """A ``session/request_permission`` frame for a shell command."""
    return {
        "id": req_id,
        "method": "session/request_permission",
        "params": {
            "toolCall": {
                "toolCallId": f"c{req_id}",
                "title": "bash: echo hi",
                "toolName": "bash",
                "rawInput": {"command": "echo hi"},
            },
            "options": [
                {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
            ],
        },
    }


class TestCancelWithAnApprovalOpen:
    """An approval is answered and taken in before its turn is interrupted.

    vibe 2.26 corrupts the session store when the answer and the interrupt
    arrive together: the chat then refuses every prompt and cannot be reopened.
    """

    @pytest.fixture
    def fake(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeAcpClient:
        """Swap vibe-acp for the recorder; provenance and explanations off."""
        client = _FakeAcpClient()
        monkeypatch.setattr(server, "_client", client)
        monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.setattr(server.settings, "load_provenance_enabled", lambda: False)
        monkeypatch.setattr(server.settings, "load_explain_enabled", lambda: False)
        return client

    @staticmethod
    async def _dialog_shown(conn: server._ChatConnection) -> None:
        ws = cast("_StubWs", conn.ws)
        for _ in range(200):
            if any(m.get("type") == "permission_request" for m in ws.sent):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the approval dialog was never shown")

    @staticmethod
    def _assert_answered_then_interrupted(fake: _FakeAcpClient) -> None:
        answered = fake.calls.index(("respond", "cancelled"))
        taken = fake.calls.index(("answer-taken", ""))
        interrupted = fake.calls.index(("notify", "session/cancel"))
        assert answered < taken < interrupted

    async def test_open_dialog_is_answered_and_taken_before_the_interrupt(
        self, fake: _FakeAcpClient
    ) -> None:
        """Stop, a new prompt or a reload with the dialog open: answer, wait, interrupt."""
        conn, queue = _connection()
        fake.frames = queue
        conn._prompt_task = asyncio.create_task(conn._run_prompt("first"))
        await asyncio.sleep(0.01)
        await queue.put(_approval_request(7))
        await self._dialog_shown(conn)
        await conn._cancel_prompt()
        self._assert_answered_then_interrupted(fake)
        assert fake.calls[-1] == ("turn-ended", "session/prompt")
        assert queue.empty()

    async def test_queued_request_is_answered_and_taken_before_the_interrupt(
        self, fake: _FakeAcpClient
    ) -> None:
        """A request the relay had not reached yet is not left for after the interrupt."""
        conn, queue = _connection()
        fake.frames = queue
        conn._prompt_rpc = asyncio.create_task(fake.request("session/prompt", {}))
        conn._prompt_task = asyncio.create_task(asyncio.sleep(60))
        await asyncio.sleep(0.01)
        await queue.put(_approval_request(8))
        await conn._cancel_prompt()
        self._assert_answered_then_interrupted(fake)

    async def test_interrupt_still_goes_out_when_vibe_never_takes_the_answer(
        self, fake: _FakeAcpClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silent vibe-acp delays the interrupt by the timeout, not forever."""
        monkeypatch.setattr(server, "PERMISSION_SETTLE_TIMEOUT", 0.05)
        conn, queue = _connection()
        conn._prompt_task = asyncio.create_task(conn._run_prompt("first"))
        await asyncio.sleep(0.01)
        await queue.put(_approval_request(9))
        await self._dialog_shown(conn)
        await asyncio.wait_for(conn._cancel_prompt(), timeout=1.0)
        assert ("notify", "session/cancel") in fake.calls

    async def test_close_marks_the_session_until_vibe_released_it(
        self, fake: _FakeAcpClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resume can tell that the chat's last socket is still closing it."""
        monkeypatch.setattr(server, "_closing_sessions", dict[str, asyncio.Event]())
        conn, _ = _connection()
        marked: list[bool] = []
        plain_request = fake.request

        async def _request(method: str, params: dict[str, object]) -> dict[str, object]:
            if method == "session/close":
                event = server._closing_sessions.get("sid")
                marked.append(event is not None and not event.is_set())
            return await plain_request(method, params)

        monkeypatch.setattr(fake, "request", _request)
        await conn.close()
        assert marked == [True]
        assert server._closing_sessions == {}


class _StubChatWs(_StubWs):
    """A browser socket that connects and leaves again at once."""

    async def accept(self) -> None:
        """Nothing to negotiate."""

    async def receive_json(self) -> dict[str, object]:
        raise WebSocketDisconnect

    async def close(self) -> None:
        """Nothing to release."""


class _FakeSessionClient(_FakeAcpClient):
    """vibe-acp as ``ws_chat`` and the fork endpoint see it: sessions open and close."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, dict[str, object]]] = []

    async def ensure_started(self) -> None:
        """Already running."""

    async def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.requests.append((method, params))
        self.calls.append(("request", method))
        if method in ("session/new", "session/fork"):
            return {"result": {"sessionId": "fresh"}}
        return {"result": {}}

    def register_session(self, session_id: str) -> asyncio.Queue[dict[str, object]]:
        self.calls.append(("register", session_id))
        return asyncio.Queue()

    def unregister_session(self, session_id: str) -> None:
        self.calls.append(("unregister", session_id))


class TestOneAttachmentPerSession:
    """vibe 2.26 attaches an open session again instead of refusing; we never ask it to."""

    @pytest.fixture
    def fake(self, monkeypatch: pytest.MonkeyPatch) -> _FakeSessionClient:
        """Swap vibe-acp for the recorder and keep every disk access out of the way."""

        async def _no_window() -> None:
            return None

        def _no_servers() -> list[dict[str, object]]:
            return []

        def _no_sync(servers: list[dict[str, object]]) -> None:
            del servers

        def _no_model() -> None:
            return None

        def _no_stop_ids(session_id: str) -> set[str]:
            del session_id
            return set()

        def _own_tip(sid: str, stop_ids: set[str] | None = None) -> str:
            del stop_ids
            return sid

        def _no_title(sid: str) -> None:
            del sid

        def _no_purge(sid: str, stop_ids: set[str] | None = None) -> None:
            del sid, stop_ids

        def _no_copy(src: str, dst: str) -> None:
            del src, dst

        def _no_entry(sid: str) -> dict[str, object]:
            del sid
            return {}

        def _titled(sid: str, title: str) -> bool:
            del sid, title
            return True

        client = _FakeSessionClient()
        monkeypatch.setattr(server, "_client", client)
        monkeypatch.setattr(server, "_connections", set[server._ChatConnection]())
        monkeypatch.setattr(server, "_closing_sessions", dict[str, asyncio.Event]())
        monkeypatch.setattr(server, "_chain_stop_ids", _no_stop_ids)
        monkeypatch.setattr(server.settings, "active_servers", _no_servers)
        monkeypatch.setattr(server.settings, "sync_servers_to_vibe_config", _no_sync)
        monkeypatch.setattr(server.settings, "active_cloud_model", _no_model)
        monkeypatch.setattr(server.settings, "load_local_model", _no_model)
        monkeypatch.setattr(server.settings, "fetch_context_window", _no_window)
        monkeypatch.setattr(provenance, "vibe_chain_tip", _own_tip)
        monkeypatch.setattr(provenance, "vibe_session_title", _no_title)
        monkeypatch.setattr(provenance, "purge_session", _no_purge)
        monkeypatch.setattr(provenance, "copy_transcript", _no_copy)
        monkeypatch.setattr(sessions, "get_entry", _no_entry)
        monkeypatch.setattr(sessions, "set_auto_title", _titled)
        return client

    async def test_branch_is_released_before_the_browser_opens_it(
        self, fake: _FakeSessionClient
    ) -> None:
        """session/fork leaves the fork attached; loading it on top doubled every frame."""
        assert await server.fork_session("src") == {"id": "fresh"}
        methods = [method for method, _ in fake.requests]
        assert methods == ["session/fork", "session/close"]
        assert fake.requests[-1][1] == {"sessionId": "fresh"}

    async def test_resume_waits_for_the_socket_still_closing_the_chat(
        self, fake: _FakeSessionClient
    ) -> None:
        """A reload's new socket loads the chat only once the old one released it."""
        released = asyncio.Event()
        server._closing_sessions["sid"] = released
        ws = _StubChatWs()
        task = asyncio.create_task(server.ws_chat(cast("Any", ws), resume="sid"))
        await asyncio.sleep(0.05)
        assert ("request", "session/load") not in fake.calls
        released.set()
        await asyncio.wait_for(task, timeout=1.0)
        assert ("request", "session/load") in fake.calls
        assert ws.sent[0]["sessionId"] == "sid"

    async def test_chat_open_in_another_socket_is_not_loaded_again(
        self, fake: _FakeSessionClient
    ) -> None:
        """A second tab gets a fresh chat; the first keeps its session and its frames."""
        first, _ = _connection()
        server._connections.add(first)
        ws = _StubChatWs()
        await server.ws_chat(cast("Any", ws), resume="sid")
        assert ("request", "session/load") not in fake.calls
        assert ("register", "sid") not in fake.calls
        assert ("unregister", "sid") not in fake.calls
        assert ("request", "session/new") in fake.calls
        assert ws.sent[0]["sessionId"] == "fresh"


class TestPathGuardDenial:
    """A call the path guard turned back renders as a note, not a failure."""

    def test_vibes_hook_wording_and_our_own_marker_both_count(self) -> None:
        """Older vibe names the hook; the harness relays only our refusal text."""
        assert server._is_pathguard_denial(
            f"Tool 'x' was denied by hook '{settings.PATHGUARD_HOOK_NAME}': no"
        )
        assert server._is_pathguard_denial(
            "RETRY THIS CALL WITH A CORRECTED PATH.\n\nNobody refused this call"
        )
        assert not server._is_pathguard_denial("Error executing tool: boom")


class TestDisplayTitle:
    """A tool card is titled by vibe's prose, or by the tool's own name."""

    def test_route_title_becomes_the_canonical_name(self) -> None:
        """The harness titles an MCP call with its route; the person sees the stack tool."""
        title = server._display_title(
            "mcp_medmcp_neuro.skull_strip",
            "mcp_medmcp_neuro.skull_strip",
            "medmcp-neuro_skull_strip",
        )
        assert title == "medmcp-neuro_skull_strip"

    def test_prose_title_is_kept(self) -> None:
        """A title that says more than the route stays as written."""
        assert server._display_title("bash: ls -la", "file_system.bash", "bash") == "bash: ls -la"

    def test_empty_title_falls_back(self) -> None:
        """No title and no name still gives the card something to say."""
        assert server._display_title(None, "", "") == "tool"
        assert server._display_title("", "x.y", "x_y") == "x_y"


def test_an_approved_stack_call_is_granted_to_the_broker() -> None:
    """Approving a stack call lets that exact call through the broker; a built-in grants none."""
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    conn = server._ChatConnection(
        cast("Any", _StubWs()), "sid", cast("Any", queue), [{"name": "medmcp-neuro-core"}]
    )
    conn._grant_stack_call(  # pyright: ignore[reportPrivateUsage]
        {"toolName": "medmcp-neuro-core_skull_strip", "rawInput": '{"input_path": "/w/t1.nii"}'}
    )
    conn._grant_stack_call({"toolName": "bash", "rawInput": {"command": "ls"}})  # pyright: ignore[reportPrivateUsage]
    ledger = server._approvals  # pyright: ignore[reportPrivateUsage]
    assert ledger.consume("medmcp-neuro-core", "skull_strip", {"input_path": "/w/t1.nii"})
    assert not ledger.consume("medmcp-neuro-core", "skull_strip", {"input_path": "/w/t1.nii"})
    assert not ledger.consume("builtin", "bash", {"command": "ls"})


class TestStackToolGrantForTheChat:
    """A stack tool approved for the chat runs unasked, unless a call needs a look."""

    @pytest.fixture
    def responses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> list[tuple[int, object]]:
        """Capture what is answered to vibe; provenance and explanations off."""
        sent: list[tuple[int, object]] = []

        async def _respond(req_id: int, result: object) -> None:
            sent.append((req_id, result))

        monkeypatch.setattr(server._client, "respond", _respond)
        monkeypatch.setattr(settings, "load_provenance_enabled", lambda: False)
        monkeypatch.setattr(settings, "load_explain_enabled", lambda: False)
        monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.delenv(server.STACK_TOOL_APPROVAL_ENV_VAR, raising=False)
        (tmp_path / "study").mkdir()
        return sent

    @staticmethod
    def _request(req_id: int, tool: str, **arguments: str) -> dict[str, object]:
        return {
            "id": req_id,
            "method": "session/request_permission",
            "params": {
                "toolCall": {
                    "toolCallId": f"c{req_id}",
                    "title": tool,
                    "toolName": tool,
                    "rawInput": arguments,
                },
                "options": [
                    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "allow_always", "name": "Always", "kind": "allow_always"},
                    {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
                ],
            },
        }

    @staticmethod
    def _dialogs(conn: server._ChatConnection) -> list[dict[str, Any]]:
        return [m for m in cast("_StubWs", conn.ws).sent if m.get("type") == "permission_request"]

    async def _answer(
        self, conn: server._ChatConnection, req: dict[str, object], option: str
    ) -> None:
        """Run the handler until its dialog is shown, then choose *option*."""
        task = asyncio.create_task(conn._handle_permission(req))
        await asyncio.sleep(0.05)
        conn._pending_perms[cast("int", req["id"])].set_result(option)
        await task

    async def test_the_dialog_offers_the_tool_for_the_rest_of_the_chat(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """The option names the tool, and vibe's own "always" stays hidden."""
        conn, _ = _connection()
        req = self._request(1, "medmcp-cohort_index_dataset", dataset_dir=str(tmp_path / "study"))
        await self._answer(conn, req, "allow_once")
        offered = {o["optionId"]: o["name"] for o in self._dialogs(conn)[0]["options"]}
        assert offered[server._STACK_TOOL_CHAT_OPTION] == ("Allow index_dataset in this chat")
        assert "allow_always" not in offered

    async def test_later_calls_of_that_tool_are_allowed_without_a_dialog(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """Vibe is answered ``allow_once`` each time; only the first call shows a dialog."""
        conn, _ = _connection()
        folder = str(tmp_path / "study")
        await self._answer(
            conn,
            self._request(1, "medmcp-cohort_summarize_dataset", index_path=folder),
            server._STACK_TOOL_CHAT_OPTION,
        )
        await conn._handle_permission(
            self._request(2, "medmcp-cohort_summarize_dataset", index_path=folder)
        )
        allow = {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        assert responses == [(1, allow), (2, allow)]
        assert len(self._dialogs(conn)) == 1

    async def test_another_tool_still_asks(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """The approval is for one tool of one stack, not for the stack."""
        conn, _ = _connection()
        folder = str(tmp_path / "study")
        await self._answer(
            conn,
            self._request(1, "medmcp-cohort_summarize_dataset", index_path=folder),
            server._STACK_TOOL_CHAT_OPTION,
        )
        await self._answer(
            conn, self._request(2, "medmcp-cohort_select_series", index_path=folder), "allow_once"
        )
        assert len(self._dialogs(conn)) == 2

    async def test_a_call_that_would_overwrite_a_file_asks_again(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """An approved tool is shown again, without the chat option, when it would overwrite."""
        conn, _ = _connection()
        scan = tmp_path / "study" / "scan.nii.gz"
        scan.write_bytes(b"")
        existing = tmp_path / "study" / "out.nii.gz"
        await self._answer(
            conn,
            self._request(
                1, "medmcp-tptbox_process_nifti", input_path=str(scan), output_path=str(existing)
            ),
            server._STACK_TOOL_CHAT_OPTION,
        )
        existing.write_bytes(b"")
        await self._answer(
            conn,
            self._request(
                2, "medmcp-tptbox_process_nifti", input_path=str(scan), output_path=str(existing)
            ),
            "allow_once",
        )
        dialogs = self._dialogs(conn)
        assert len(dialogs) == 2
        assert server._STACK_TOOL_CHAT_OPTION not in {o["optionId"] for o in dialogs[1]["options"]}

    async def test_a_path_outside_the_workspace_is_never_covered(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """Neither offered for nor covered by the chat approval."""
        conn, _ = _connection()
        await self._answer(
            conn, self._request(1, "medmcp-cohort_index_dataset", dataset_dir="/etc"), "allow_once"
        )
        assert server._STACK_TOOL_CHAT_OPTION not in {
            o["optionId"] for o in self._dialogs(conn)[0]["options"]
        }

    async def test_the_shell_is_never_offered_it(self, responses: list[tuple[int, object]]) -> None:
        """Built-in tools keep their own rules."""
        conn, _ = _connection()
        req = self._request(1, "bash", command="python3 script.py")
        await self._answer(conn, req, "allow_once")
        assert server._STACK_TOOL_CHAT_OPTION not in {
            o["optionId"] for o in self._dialogs(conn)[0]["options"]
        }

    async def test_the_kill_switch_restores_per_call_approval(
        self,
        responses: list[tuple[int, object]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``MEDMCP_STACK_TOOL_APPROVAL=0`` removes the option."""
        monkeypatch.setenv(server.STACK_TOOL_APPROVAL_ENV_VAR, "0")
        conn, _ = _connection()
        req = self._request(1, "medmcp-cohort_index_dataset", dataset_dir=str(tmp_path / "study"))
        await self._answer(conn, req, "allow_once")
        assert server._STACK_TOOL_CHAT_OPTION not in {
            o["optionId"] for o in self._dialogs(conn)[0]["options"]
        }


class TestWhatAChatLongApprovalNeverCovers:
    """Jobs, external servers, and another tool's ``command`` argument still ask."""

    responses = TestStackToolGrantForTheChat.responses
    _request = staticmethod(TestStackToolGrantForTheChat._request)
    _dialogs = staticmethod(TestStackToolGrantForTheChat._dialogs)
    _answer = TestStackToolGrantForTheChat._answer

    async def test_a_tool_that_starts_a_job_is_asked_every_time(
        self,
        responses: list[tuple[int, object]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A job runs for hours on its own arguments; one approval is one job."""

        class _Jobs:
            def declares(self, stack: str, tool: str) -> bool:
                return (stack, tool) == ("medmcp-foo", "start_long_run")

        monkeypatch.setattr(server, "_jobs", _Jobs())
        conn, _ = _connection()
        req = self._request(1, "medmcp-foo_start_long_run", data=str(tmp_path / "study"))
        await self._answer(conn, req, "allow_once")
        offered = {o["optionId"] for o in self._dialogs(conn)[0]["options"]}
        assert server._STACK_TOOL_CHAT_OPTION not in offered

    async def test_an_external_server_is_asked_every_time(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """Each call of an external server sends data out; none is covered."""
        conn, _ = _connection()
        conn.servers = [{"name": "remote", "url": "https://mcp.example/mcp"}]
        await self._answer(conn, self._request(1, "remote_lookup", query="x"), "allow_once")
        offered = {o["optionId"] for o in self._dialogs(conn)[0]["options"]}
        assert server._STACK_TOOL_CHAT_OPTION not in offered

    async def test_the_read_only_grant_is_for_the_shell_only(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """A stack tool with a ``command`` argument is not a read-only shell command."""
        conn, _ = _connection()
        conn.servers = [{"name": "medmcp-foo"}]
        conn._read_only_granted = True
        task = asyncio.create_task(
            conn._handle_permission(self._request(1, "medmcp-foo_run", command="ls"))
        )
        await asyncio.sleep(0.05)
        assert len(self._dialogs(conn)) == 1
        conn._pending_perms[1].set_result("reject_once")
        await task


class TestAssistantCode:
    """Code the assistant wrote itself is always put to the user, or refused."""

    @pytest.fixture
    def responses(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, object]]:
        """Capture what is answered to vibe; provenance and explanations off."""
        sent: list[tuple[int, object]] = []

        async def _respond(req_id: int, result: object) -> None:
            sent.append((req_id, result))

        monkeypatch.setattr(server._client, "respond", _respond)
        monkeypatch.setattr(settings, "load_provenance_enabled", lambda: False)
        monkeypatch.setattr(settings, "load_explain_enabled", lambda: False)
        monkeypatch.delenv(server.READ_ONLY_APPROVAL_ENV_VAR, raising=False)
        monkeypatch.delenv(server.ASSISTANT_CODE_ENV_VAR, raising=False)
        return sent

    @staticmethod
    def _dialogs(conn: server._ChatConnection) -> list[dict[str, Any]]:
        return [m for m in cast("_StubWs", conn.ws).sent if m.get("type") == "permission_request"]

    async def test_a_label_only_the_model_gave_is_not_shown(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """The dialog carries the fixed rules' labels, not the model's own reading."""
        conn, _ = _connection()
        req = TestReadOnlyGrantForTheChat._request(1, "python3 - <<'PY'\nprint(1)\nPY")
        cast("dict[str, Any]", req["params"])["toolCall"]["risks"] = ["privacy"]
        task = asyncio.create_task(conn._handle_permission(req))
        await asyncio.sleep(0.05)
        keys = {r["key"] for r in self._dialogs(conn)[-1]["risks"]}
        assert "assistant_code" in keys
        assert "privacy" not in keys
        conn._pending_perms[1].set_result("reject_once")
        await task

    async def test_script_is_labelled_in_the_dialog(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """A Python heredoc is shown with the assistant_code label."""
        conn, _ = _connection()
        req = TestReadOnlyGrantForTheChat._request(1, "python3 - <<'PY'\nprint(1)\nPY")
        task = asyncio.create_task(conn._handle_permission(req))
        await asyncio.sleep(0.05)
        dialog = self._dialogs(conn)[-1]
        assert "assistant_code" in {r["key"] for r in dialog["risks"]}
        assert [o["optionId"] for o in dialog["options"]] == ["allow_once", "reject_once"]
        conn._pending_perms[1].set_result("reject_once")
        await task

    async def test_deny_mode_refuses_without_a_dialog(
        self, responses: list[tuple[int, object]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MEDMCP_ASSISTANT_CODE=deny answers reject at once; nothing is shown."""
        monkeypatch.setenv(server.ASSISTANT_CODE_ENV_VAR, "deny")
        conn, _ = _connection()
        await conn._handle_permission(TestReadOnlyGrantForTheChat._request(1, "Rscript analysis.R"))
        assert responses[-1] == (1, {"outcome": {"outcome": "selected", "optionId": "reject_once"}})
        assert self._dialogs(conn) == []

    async def test_read_only_grant_never_covers_code(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """After "allow looking at files", an awk program still asks."""
        conn, _ = _connection()
        await TestReadOnlyGrantForTheChat()._answer(
            conn,
            TestReadOnlyGrantForTheChat._request(1, "ls", ("command_pattern", "ls *")),
            "allow_always",
        )
        req = TestReadOnlyGrantForTheChat._request(
            2, "awk -F, '{s+=$2} END {print s}' a.csv", ("command_pattern", "awk *")
        )
        task = asyncio.create_task(conn._handle_permission(req))
        await asyncio.sleep(0.05)
        assert len(self._dialogs(conn)) == 2
        conn._pending_perms[2].set_result("reject_once")
        await task


def test_workspace_is_exported_without_the_stack_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The running server exports the resolved workspace, pool or no pool.

    Container-stack manifests mount ``${MEDMCP_WORKSPACE}``; unset, the literal
    reaches ``docker run`` and every container stack fails discovery.
    """
    resolved = tmp_path.resolve()
    monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
    monkeypatch.setattr(settings, "stack_pool_enabled", lambda: False)
    # setenv so monkeypatch restores the variable the lifespan assigns.
    monkeypatch.setenv("MEDMCP_WORKSPACE", "relative/elsewhere")

    with TestClient(server.app, base_url="http://127.0.0.1:8100"):
        assert os.environ["MEDMCP_WORKSPACE"] == str(resolved)


def test_the_frontend_page_is_revalidated_but_its_assets_are_not(tmp_path: Path) -> None:
    """After an update the browser fetches the new page instead of reusing an old copy."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>")
    (tmp_path / "assets" / "index-abc.css").write_text("body{}")
    app = FastAPI()
    app.mount("/", server._FrontendFiles(directory=tmp_path, html=True))
    client = TestClient(app)
    assert client.get("/").headers.get("cache-control") == "no-cache"
    assert "cache-control" not in client.get("/assets/index-abc.css").headers


class TestAllowEverythingInThisChat:
    """Allow everything in this chat: the core answers ordinary approvals itself."""

    ALLOW: ClassVar[dict[str, object]] = {
        "outcome": {"outcome": "selected", "optionId": "allow_once"}
    }
    responses = TestStackToolGrantForTheChat.responses
    _request = staticmethod(TestStackToolGrantForTheChat._request)
    _dialogs = staticmethod(TestStackToolGrantForTheChat._dialogs)
    _answer = TestStackToolGrantForTheChat._answer

    @pytest.fixture(autouse=True)
    def _setting_on(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Settings > Advanced has it switched on, consent given."""
        monkeypatch.delenv(server.CHAT_AUTO_APPROVE_ENV_VAR, raising=False)
        monkeypatch.setattr(settings, "CHAT_AUTO_APPROVE_PATH", tmp_path / "chat_auto_approve.json")
        settings.acknowledge_chat_auto_approve()
        settings.set_chat_auto_approve_enabled(True)

    async def _turn_on(self, conn: server._ChatConnection) -> None:
        await self._answer(
            conn, self._request(1, "bash", command="mkdir x"), server._CHAT_AUTO_APPROVE_OPTION
        )

    async def test_ordinary_dialogs_offer_it(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """A shell command, a new file and a stack call all offer it."""
        conn, _ = _connection()
        await self._answer(conn, self._request(1, "bash", command="mkdir x"), "allow_once")
        await self._answer(
            conn, self._request(2, "write_file", path=str(tmp_path / "a.txt")), "allow_once"
        )
        await self._answer(
            conn,
            self._request(3, "medmcp-cohort_index_dataset", dataset_dir=str(tmp_path / "study")),
            "allow_once",
        )
        for dialog in self._dialogs(conn):
            offered = {o["optionId"]: o["name"] for o in dialog["options"]}
            assert offered[server._CHAT_AUTO_APPROVE_OPTION] == "Allow everything in this chat"

    async def test_later_ordinary_calls_run_without_a_dialog_and_are_recorded(
        self,
        responses: list[tuple[int, object]],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Each call gets allow_once, no dialog; the browser hears it is on; audited as auto."""
        conn, _ = _connection()
        await self._turn_on(conn)
        with caplog.at_level(logging.INFO, logger="medmcp.audit"):
            await conn._handle_permission(self._request(2, "bash", command="mkdir y"))
            await conn._handle_permission(
                self._request(3, "write_file", path=str(tmp_path / "a.txt"))
            )
            await conn._handle_permission(
                self._request(4, "medmcp-cohort_index_dataset", dataset_dir=str(tmp_path / "study"))
            )
        assert responses == [(n, self.ALLOW) for n in (1, 2, 3, 4)]
        assert len(self._dialogs(conn)) == 1
        frames = [m for m in cast("_StubWs", conn.ws).sent if m.get("type") == "auto_approve"]
        assert frames[0]["enabled"] is True and frames[0]["since"]
        assert sum("auto-approved" in r.getMessage() for r in caplog.records) == 3

    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [
            ("bash", {"command": "rm -r y"}),
            ("bash", {"command": "python3 - <<'EOF'\nprint(1)\nEOF"}),
            ("medmcp-cohort_index_dataset", {"dataset_dir": "/etc"}),
            ("edit", {"file_path": "results.csv", "old_string": "a", "new_string": "b"}),
            ("write_file", {"path": "results.csv", "content": "x"}),
            ("bash", {"command": "mv new.csv results.csv"}),
            ("bash", {"command": "cp /dev/null results.csv"}),
            ("bash", {"command": "echo x > results.csv"}),
            ("bash", {"command": "sed -i 's/a/b/' results.csv"}),
            ("web_fetch", {"url": "https://example.org/?q=sub-01"}),
            ("bash", {"command": "curl https://example.org/?id=sub-01"}),
        ],
        ids=[
            "delete",
            "assistant-code",
            "outside-workspace",
            "edit",
            "overwrite",
            "mv",
            "cp",
            "redirect",
            "sed-i",
            "web-fetch",
            "curl",
        ],
    )
    async def test_risky_calls_still_ask_without_the_option(
        self,
        responses: list[tuple[int, object]],
        tmp_path: Path,
        tool: str,
        arguments: dict[str, str],
    ) -> None:
        """Changing or deleting files, the network, assistant code and outside paths ask."""
        (tmp_path / "results.csv").write_text("a")
        conn, _ = _connection()
        await self._turn_on(conn)
        await self._answer(conn, self._request(2, tool, **arguments), "allow_once")
        dialogs = self._dialogs(conn)
        assert len(dialogs) == 2
        assert server._CHAT_AUTO_APPROVE_OPTION not in {
            o["optionId"] for o in dialogs[1]["options"]
        }

    async def test_a_destructive_stack_tool_still_asks(
        self,
        responses: list[tuple[int, object]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A tool its stack declares destructive is never covered, as under #206."""

        def _effect(stack: str, tool: str) -> str | None:
            return "destructive" if tool == "erase" else None

        monkeypatch.setattr(server, "tool_effect", _effect)
        conn, _ = _connection()
        await self._turn_on(conn)
        req = self._request(2, "medmcp-cohort_erase", path=str(tmp_path / "study"))
        await self._answer(conn, req, "allow_once")
        assert len(self._dialogs(conn)) == 2

    async def test_it_can_be_turned_off_and_a_new_chat_asks_again(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """Off from the browser restores the dialog; a new chat connection starts off."""
        conn, _ = _connection()
        await self._turn_on(conn)
        await conn._set_auto_approve(False)
        await self._answer(conn, self._request(2, "bash", command="mkdir y"), "reject_once")
        assert len(self._dialogs(conn)) == 2
        off = {"type": "auto_approve", "enabled": False, "since": None}
        assert off in cast("_StubWs", conn.ws).sent
        fresh, _ = _connection()
        await self._answer(fresh, self._request(3, "bash", command="mkdir z"), "allow_once")
        assert len(self._dialogs(fresh)) == 1

    async def test_off_in_settings_it_is_not_offered_and_a_forged_choice_is_refused(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """Without the setting (the default) no dialog offers it, and its id grants nothing."""
        settings.set_chat_auto_approve_enabled(False)
        conn, _ = _connection()
        await self._turn_on(conn)
        offered = {o["optionId"] for o in self._dialogs(conn)[0]["options"]}
        assert server._CHAT_AUTO_APPROVE_OPTION not in offered
        assert conn._auto_approve is None
        assert responses == [(1, {"outcome": {"outcome": "cancelled"}})]

    async def test_switching_the_setting_off_ends_it_in_an_open_chat(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """The next call asks again, and the browser hears it is off."""
        conn, _ = _connection()
        await self._turn_on(conn)
        settings.set_chat_auto_approve_enabled(False)
        await self._answer(conn, self._request(2, "bash", command="mkdir y"), "allow_once")
        assert len(self._dialogs(conn)) == 2
        assert conn._auto_approve is None
        off = {"type": "auto_approve", "enabled": False, "since": None}
        assert off in cast("_StubWs", conn.ws).sent

    async def test_the_kill_switch_hides_it_and_a_forged_choice_is_refused(
        self, responses: list[tuple[int, object]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MEDMCP_CHAT_AUTO_APPROVE=0: not offered, and sending its id anyway grants nothing."""
        monkeypatch.setenv(server.CHAT_AUTO_APPROVE_ENV_VAR, "0")
        conn, _ = _connection()
        await self._turn_on(conn)
        offered = {o["optionId"] for o in self._dialogs(conn)[0]["options"]}
        assert server._CHAT_AUTO_APPROVE_OPTION not in offered
        assert conn._auto_approve is None
        assert responses == [(1, {"outcome": {"outcome": "cancelled"}})]


class TestChatAutoApproveSetting:
    """Settings > Advanced: off by default, on only after consent, off clears consent."""

    @pytest.fixture(autouse=True)
    def _state(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv(server.CHAT_AUTO_APPROVE_ENV_VAR, raising=False)
        monkeypatch.setattr(settings, "CHAT_AUTO_APPROVE_PATH", tmp_path / "chat_auto_approve.json")

    def test_off_by_default(self, client: TestClient) -> None:
        """No state file: off, not acknowledged, available."""
        assert client.get("/api/chat-auto-approve").json() == {
            "enabled": False,
            "acknowledged": False,
            "available": True,
        }

    def test_on_needs_the_acknowledgement_and_off_clears_it(self, client: TestClient) -> None:
        """Enabling without consent is a 400; disabling means consenting again next time."""
        assert client.put("/api/chat-auto-approve", json={"enabled": True}).status_code == 400
        assert client.post("/api/chat-auto-approve/acknowledge").status_code == 200
        on = client.put("/api/chat-auto-approve", json={"enabled": True})
        assert on.json() == {"enabled": True, "acknowledged": True, "available": True}
        off = client.put("/api/chat-auto-approve", json={"enabled": False})
        assert off.json()["acknowledged"] is False
        assert client.put("/api/chat-auto-approve", json={"enabled": True}).status_code == 400

    def test_enabled_without_acknowledgement_counts_as_off(self) -> None:
        """A state file carrying one without the other is not trusted."""
        settings.atomic_write_json(settings.CHAT_AUTO_APPROVE_PATH, {"enabled": True})
        assert settings.chat_auto_approve_enabled() is False

    def test_the_environment_forbids_it(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MEDMCP_CHAT_AUTO_APPROVE=0: unavailable, and neither consent nor enabling is taken."""
        settings.acknowledge_chat_auto_approve()
        settings.set_chat_auto_approve_enabled(True)
        monkeypatch.setenv(server.CHAT_AUTO_APPROVE_ENV_VAR, "0")
        assert client.get("/api/chat-auto-approve").json()["available"] is False
        assert client.get("/api/chat-auto-approve").json()["enabled"] is False
        assert client.put("/api/chat-auto-approve", json={"enabled": True}).status_code == 400
        assert client.post("/api/chat-auto-approve/acknowledge").status_code == 400
