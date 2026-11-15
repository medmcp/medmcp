"""Tests for the zip upload (``POST /api/files/upload-zip``).

The extraction must never write outside the chosen folder, whatever the
archive claims.
"""

from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medmcp import server, zipupload


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


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _upload(client: TestClient, data: bytes, dir: str = "") -> tuple[int, dict[str, object]]:
    resp = client.post(
        "/api/files/upload-zip",
        params={"dir": dir},
        files={"file": ("case.zip", io.BytesIO(data), "application/zip")},
    )
    return resp.status_code, resp.json()


def _only_dir(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


class TestZipUpload:
    """A zip is unpacked into the chosen folder, and nowhere else."""

    def test_extracts_nested_folders_and_reports_counts(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """The folder tree inside the archive is recreated under ``dir``."""
        data = _zip(
            {
                "case1/": b"",
                "case1/series2/IM0001.dcm": b"dicom",
                "case1/notes.txt": b"hi",
                "__MACOSX/case1/._notes.txt": b"junk",
                "case1/.DS_Store": b"junk",
            }
        )
        status, body = _upload(client, data, "incoming")
        assert status == 200, body
        assert body["files"] == 2
        assert body["bytes"] == 7
        assert body["path"] == "incoming"
        root = workspace_root / "incoming"
        assert (root / "case1" / "series2" / "IM0001.dcm").read_bytes() == b"dicom"
        assert _only_dir(root) == [
            "case1",
            "case1/notes.txt",
            "case1/series2",
            "case1/series2/IM0001.dcm",
        ]

    @pytest.mark.parametrize(
        "name",
        ["../evil.txt", "a/../../evil.txt", "/abs/evil.txt", "C:/evil.txt", "..\\evil.txt"],
    )
    def test_zip_slip_is_refused_before_anything_is_written(
        self, workspace_root: Path, client: TestClient, name: str
    ) -> None:
        """A member escaping the folder rejects the whole archive, naming the member."""
        data = _zip({"good.txt": b"ok", name: b"evil"})
        status, body = _upload(client, data, "in")
        assert status == 400
        assert name in str(body["detail"])
        assert not (workspace_root / "in").exists()
        assert not (workspace_root.parent / "evil.txt").exists()
        assert not list(workspace_root.parent.rglob("evil.txt"))

    def test_symlink_member_is_refused(self, workspace_root: Path, client: TestClient) -> None:
        """A link stored in the archive could point anywhere on the host."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
        status, body = _upload(client, buf.getvalue())
        assert status == 400
        assert "link" in str(body["detail"])
        assert not (workspace_root / "link").exists()

    def test_device_member_is_refused(self, workspace_root: Path, client: TestClient) -> None:
        """Only regular files and folders are unpacked."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("fifo")
            info.external_attr = (stat.S_IFIFO | 0o644) << 16
            zf.writestr(info, b"")
        status, _ = _upload(client, buf.getvalue())
        assert status == 400
        assert not (workspace_root / "fifo").exists()

    def test_encrypted_member_is_refused(self, workspace_root: Path, client: TestClient) -> None:
        """An encrypted entry cannot be read without a password: refused up front, not midway."""
        data = bytearray(_zip({"secret.txt": b"x"}))
        # zipfile cannot write encrypted entries: set the "encrypted" flag bit by
        # hand in the local header (flags at +6) and the central directory (+8).
        data[6] |= 0x1
        data[data.index(b"PK\x01\x02") + 8] |= 0x1
        status, body = _upload(client, bytes(data))
        assert status == 400
        assert "encrypted" in str(body["detail"])
        assert not (workspace_root / "secret.txt").exists()

    def test_does_not_write_through_a_link_in_the_target(
        self, workspace_root: Path, outside: Path, client: TestClient
    ) -> None:
        """An existing link named like an archive folder or file is not followed."""
        (workspace_root / "case").symlink_to(outside)
        status, body = _upload(client, _zip({"case/secret.txt": b"pwned"}))
        assert status == 400, body
        assert (outside / "secret.txt").read_text() == "s"
        (workspace_root / "notes.txt").symlink_to(outside / "secret.txt")
        status, _ = _upload(client, _zip({"notes.txt": b"pwned"}))
        assert status == 400
        assert (outside / "secret.txt").read_text() == "s"

    def test_size_cap(
        self, workspace_root: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An archive unpacking past the cap is refused with a clear reason."""
        monkeypatch.setattr(zipupload, "MAX_BYTES", 1000)
        # Compresses to almost nothing: the cap is on what it unpacks to.
        status, body = _upload(client, _zip({"big.bin": b"\0" * 5000}))
        assert status == 413
        assert "unpacks to more than" in str(body["detail"])
        assert not (workspace_root / "big.bin").exists()

    def test_file_count_cap(self, workspace_root: Path) -> None:
        """Too many members are refused before extraction."""
        data = _zip({f"f{i}.txt": b"x" for i in range(5)})
        with (
            zipfile.ZipFile(io.BytesIO(data)) as zf,
            pytest.raises(zipupload.ZipRejectedError) as e,
        ):
            zipupload.check_members(zf, max_files=3)
        assert e.value.status == 413
        assert "more than 3 files" in str(e.value)
        assert not list(workspace_root.iterdir())

    def test_not_a_zip_and_empty_zip(self, client: TestClient) -> None:
        """Garbage and an archive without files get their own messages."""
        status, body = _upload(client, b"not a zip at all")
        assert status == 400
        assert "not a valid .zip" in str(body["detail"])
        status, body = _upload(client, _zip({"only/": b""}))
        assert status == 400
        assert "no files" in str(body["detail"])

    def test_dir_outside_the_workspace_is_refused(self, outside: Path, client: TestClient) -> None:
        """The target folder is confined like any other upload's."""
        status, _ = _upload(client, _zip({"a.txt": b"x"}), "../outside")
        assert status == 400
        assert not (outside / "a.txt").exists()

    def test_leaves_only_the_extracted_files(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """No temporary copy of the archive is left in the target folder."""
        status, _ = _upload(client, _zip({"a.txt": b"x"}), "in")
        assert status == 200
        assert _only_dir(workspace_root / "in") == ["a.txt"]

    def test_root_target_reports_an_empty_path(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Unpacked into the workspace itself, the reported folder is "" like everywhere else."""
        status, body = _upload(client, _zip({"a.txt": b"x"}))
        assert status == 200
        assert body["path"] == ""
        assert (workspace_root / "a.txt").read_bytes() == b"x"
