"""Tests for the explorer's lazy folder listing (``GET /api/tree/dir``).

The listing must stay as confined as the full tree it replaces in the UI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medmcp import server


@pytest.fixture
def workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the server's workspace root at a temp dir (resolved, no symlinks)."""
    resolved = (tmp_path / "ws").resolve()
    resolved.mkdir()
    monkeypatch.setattr(server, "WORKSPACE_ROOT", resolved)
    return resolved


@pytest.fixture
def outside(workspace_root: Path) -> Path:
    """A folder next to the workspace holding a file."""
    outside = workspace_root.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    return outside


def _listing(client: TestClient, path: str = "") -> dict[str, dict[str, object]]:
    resp = client.get("/api/tree/dir", params={"path": path})
    assert resp.status_code == 200, resp.text
    return {n["name"]: n for n in resp.json()["children"]}


# ── /api/tree/dir: one folder level ──────────────────────────────────────────


class TestTreeDir:
    """The explorer loads one folder at a time instead of the whole workspace."""

    def test_lists_one_level_folders_first(self, workspace_root: Path, client: TestClient) -> None:
        """Folders first, by name; files carry sizes; nothing below the level is walked."""
        (workspace_root / "b.txt").write_bytes(b"12345")
        (workspace_root / "A.nii").write_bytes(b"x")
        (workspace_root / "study" / "sub-01").mkdir(parents=True)
        (workspace_root / "study" / "sub-01" / "t1.nii.gz").write_bytes(b"x")
        (workspace_root / "empty").mkdir()
        (workspace_root / ".git").mkdir()
        body = client.get("/api/tree/dir").json()
        assert [n["name"] for n in body["children"]] == ["empty", "study", "A.nii", "b.txt"]
        assert body["total"] == 4
        nodes = {n["name"]: n for n in body["children"]}
        assert nodes["study"] == {"id": "study", "name": "study", "dir": True, "has_children": True}
        assert nodes["empty"]["has_children"] is False
        assert nodes["b.txt"] == {"id": "b.txt", "name": "b.txt", "size": 5}

    def test_lists_a_subfolder_with_full_ids(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Children carry workspace-relative ids, as the full tree's did."""
        (workspace_root / "study" / "sub-01").mkdir(parents=True)
        (workspace_root / "study" / "sub-01" / "t1.nii.gz").write_bytes(b"xy")
        nodes = _listing(client, "study/sub-01")
        assert nodes["t1.nii.gz"] == {
            "id": "study/sub-01/t1.nii.gz",
            "name": "t1.nii.gz",
            "size": 2,
        }

    def test_a_folder_holding_only_skipped_entries_has_no_children(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A folder showing nothing must not offer an expander."""
        (workspace_root / "proj" / "node_modules").mkdir(parents=True)
        assert _listing(client)["proj"]["has_children"] is False

    def test_rejects_traversal_and_missing_folders(
        self, workspace_root: Path, outside: Path, client: TestClient
    ) -> None:
        """Nothing outside the workspace can be listed."""
        del outside
        (workspace_root / "f.txt").write_text("x")
        for bad in ("../outside", "/etc", "a/../../outside"):
            assert client.get("/api/tree/dir", params={"path": bad}).status_code == 400
        assert client.get("/api/tree/dir", params={"path": "nope"}).status_code == 404
        assert client.get("/api/tree/dir", params={"path": "f.txt"}).status_code == 404

    def test_does_not_list_through_an_outside_link(
        self, workspace_root: Path, outside: Path, client: TestClient
    ) -> None:
        """The link is shown, but not the host folder (or file) behind it."""
        (workspace_root / "host").symlink_to(outside)
        (workspace_root / "secret").symlink_to(outside / "secret.txt")
        nodes = _listing(client)
        assert nodes["host"]["has_children"] is False
        assert "size" not in nodes["secret"]
        assert client.get("/api/tree/dir", params={"path": "host"}).status_code == 400

    def test_lists_through_an_inside_link_but_not_a_cycle(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A link to another workspace folder is browsable; one to an ancestor is not."""
        (workspace_root / "data").mkdir()
        (workspace_root / "data" / "a.nii").write_bytes(b"x")
        (workspace_root / "alias").symlink_to(workspace_root / "data")
        (workspace_root / "data" / "loop").symlink_to(workspace_root)
        alias = _listing(client, "alias")
        assert list(alias) == ["loop", "a.nii"]
        assert alias["a.nii"]["id"] == "alias/a.nii"
        assert alias["loop"]["has_children"] is False
        assert client.get("/api/tree/dir", params={"path": "data/loop"}).json()["children"] == []
        # Nor deeper down, through the cycle.
        deeper = client.get("/api/tree/dir", params={"path": "data/loop/data"}).json()
        assert deeper["children"] == []

    def test_caps_a_huge_folder_and_reports_the_total(
        self, workspace_root: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cut listing says how many entries there are in all."""
        monkeypatch.setattr(server, "_DIR_MAX_ENTRIES", 3)
        for i in range(5):
            (workspace_root / f"f{i}.txt").write_text("x")
        body = client.get("/api/tree/dir").json()
        assert len(body["children"]) == 3
        assert body["total"] == 5
