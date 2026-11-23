"""Tests for the zip upload (``POST /api/files/upload-zip``).

The extraction must never write outside the chosen folder, whatever the
archive claims.
"""

from __future__ import annotations

import errno
import io
import logging
import stat
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO

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


def _stored(entries: dict[str, bytes]) -> bytes:
    """A zip whose members are stored uncompressed, so their bytes can be found and patched."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _tree(root: Path) -> dict[str, bytes | None]:
    """Everything under *root*: file contents by relative path, None for a folder."""
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


@pytest.fixture
def audit(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the ``medmcp.audit`` logger, which does not propagate to the root."""
    logger = logging.getLogger("medmcp.audit")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


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
        assert status == 409, body
        assert (outside / "secret.txt").read_text() == "s"
        (workspace_root / "notes.txt").symlink_to(outside / "secret.txt")
        status, _ = _upload(client, _zip({"notes.txt": b"pwned"}))
        assert status == 409
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

    def test_file_count_cap_at_the_endpoint(
        self, workspace_root: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Too many files are refused with a reason, and nothing is written."""
        monkeypatch.setattr(zipupload, "MAX_FILES", 3)
        status, body = _upload(client, _zip({f"f{i}.txt": b"x" for i in range(5)}), "in")
        assert status == 413
        assert "more than 3 files" in str(body["detail"])
        assert _tree(workspace_root) == {}

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

    def test_not_a_zip_and_empty_zip(self, workspace_root: Path, client: TestClient) -> None:
        """Garbage and an archive without files get their own messages."""
        del workspace_root
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


class TestZipNeverReplaces:
    """An unpack adds files; it never replaces, truncates or half-writes anything."""

    def test_an_existing_file_refuses_the_whole_archive(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """The clashing file stays byte-identical and no other member is written."""
        (workspace_root / "in" / "case").mkdir(parents=True)
        (workspace_root / "in" / "case" / "notes.txt").write_bytes(b"mine, do not touch")
        before = _tree(workspace_root)
        data = _zip({"a_first.txt": b"new", "case/notes.txt": b"theirs", "case/z.txt": b"new"})
        status, body = _upload(client, data, "in")
        assert status == 409, body
        assert "'case/notes.txt' already exists" in str(body["detail"])
        assert _tree(workspace_root) == before

    def test_the_refusal_names_a_few_clashes_and_counts_the_rest(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A long list of clashes is cut, with the number left out."""
        names = [f"f{i}.txt" for i in range(8)]
        for name in names:
            (workspace_root / name).write_bytes(b"old")
        status, body = _upload(client, _zip(dict.fromkeys(names, b"new")))
        assert status == 409
        detail = str(body["detail"])
        assert "'f0.txt', 'f1.txt', 'f2.txt', 'f3.txt', 'f4.txt' and 3 more already exist" in detail
        assert "f5.txt" not in detail
        assert all((workspace_root / n).read_bytes() == b"old" for n in names)

    def test_an_existing_folder_is_merged_into(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Folders that exist are not clashes: new files land beside what is there."""
        (workspace_root / "case").mkdir()
        (workspace_root / "case" / "old.txt").write_bytes(b"old")
        status, body = _upload(client, _zip({"case/": b"", "case/new.txt": b"new"}))
        assert status == 200, body
        assert body["dirs"] == 0
        assert _tree(workspace_root) == {
            "case": None,
            "case/new.txt": b"new",
            "case/old.txt": b"old",
        }

    def test_a_folder_where_a_file_goes_and_a_file_where_a_folder_goes(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Either mismatch with what exists refuses the archive."""
        (workspace_root / "a").mkdir()
        (workspace_root / "b").write_bytes(b"file")
        before = _tree(workspace_root)
        status, body = _upload(client, _zip({"a": b"x"}))
        assert status == 409, body
        status, body = _upload(client, _zip({"ok.txt": b"x", "b/inner.txt": b"x"}))
        assert status == 409, body
        assert "'b' already exists" in str(body["detail"])
        assert _tree(workspace_root) == before

    def test_a_failure_midway_removes_what_was_written(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A member with a bad checksum after good ones: files and new folders are gone again."""
        (workspace_root / "in" / "kept").mkdir(parents=True)
        (workspace_root / "in" / "kept" / "old.txt").write_bytes(b"old")
        before = _tree(workspace_root)
        data = bytearray(
            _stored(
                {
                    "kept/good.txt": b"good",
                    "new/deep/good.txt": b"good",
                    "new/deep/bad.txt": b"PAYLOAD-TO-CORRUPT",
                }
            )
        )
        at = data.index(b"PAYLOAD-TO-CORRUPT")
        data[at] ^= 0xFF
        status, body = _upload(client, bytes(data), "in")
        assert status == 400, body
        assert "damaged" in str(body["detail"])
        assert _tree(workspace_root) == before

    def test_a_write_error_midway_rolls_back_and_is_not_a_500(
        self, workspace_root: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full disk during the second file: 507, and the first file is removed."""
        calls = 0
        real_copy = zipupload._copy  # pyright: ignore[reportPrivateUsage]

        def copy(src: IO[bytes], dst: IO[bytes]) -> int:
            nonlocal calls
            calls += 1
            if calls == 2:
                dst.write(b"partial")
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_copy(src, dst)

        monkeypatch.setattr(zipupload, "_copy", copy)
        status, body = _upload(client, _zip({"d/one.txt": b"1", "d/two.txt": b"2"}), "fresh")
        assert status == 507, body
        assert "no space" in str(body["detail"])
        assert _tree(workspace_root) == {}

    def test_unsupported_compression_is_refused_before_writing(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Deflate64 (method 9) cannot be read by zipfile: a reason, not a crash."""
        data = bytearray(_zip({"good.txt": b"ok", "odd.bin": b"x" * 10}))
        local = data.index(b"PK\x03\x04", 1)  # the second member's local header
        central = data.index(b"PK\x01\x02", data.index(b"PK\x01\x02") + 1)
        data[local + 8 : local + 10] = (9).to_bytes(2, "little")
        data[central + 10 : central + 12] = (9).to_bytes(2, "little")
        status, body = _upload(client, bytes(data))
        assert status == 400, body
        assert "'odd.bin'" in str(body["detail"])
        assert "compression" in str(body["detail"])
        assert _tree(workspace_root) == {}

    def test_a_name_twice_in_the_archive_is_refused(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """The second copy would replace the first; neither is written."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf, pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("d/a.txt", b"one")
            zf.writestr("d/a.txt", b"two")
        status, body = _upload(client, buf.getvalue())
        assert status == 400, body
        assert "'d/a.txt' is in the archive more than once" in str(body["detail"])
        # The same path spelled differently is the same file.
        status, body = _upload(client, _zip({"d/a.txt": b"one", "d\\a.txt": b"two"}))
        assert status == 400, body
        assert _tree(workspace_root) == {}

    def test_one_name_as_file_and_folder_is_refused(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """An archive holding ``a`` and ``a/b`` cannot be unpacked as it stands."""
        for entries in ({"a": b"x", "a/b.txt": b"y"}, {"a/b.txt": b"y", "a": b"x"}):
            status, body = _upload(client, _zip(entries))
            assert status == 400, body
            assert "'a' is both a file and a folder" in str(body["detail"])
        assert _tree(workspace_root) == {}

    def test_a_name_too_long_for_the_disk_is_refused(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Refused up front rather than failing on the write."""
        status, body = _upload(client, _zip({"ok.txt": b"x", "n" * 300 + ".txt": b"x"}))
        assert status == 400, body
        assert "too long" in str(body["detail"])
        assert _tree(workspace_root) == {}

    def test_a_colon_in_a_name_is_not_a_drive_letter(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """Only a drive prefix is refused; ``a:b.txt`` is a plain file name."""
        entries = {"a:b.txt": b"1", "scan 10:42.nii": b"2", "d/x:y": b"3"}
        status, body = _upload(client, _zip(entries))
        assert status == 200, body
        assert (workspace_root / "a:b.txt").read_bytes() == b"1"
        assert (workspace_root / "d" / "x:y").read_bytes() == b"3"
        for name in ("C:/evil.txt", "c:\\evil.txt", "C:"):
            status, _ = _upload(client, _zip({name: b"x"}), "drive")
            assert status == 400, name
        assert not (workspace_root / "drive").exists()

    def test_dir_that_is_a_file_is_refused(self, workspace_root: Path, client: TestClient) -> None:
        """A target that is a file, or lies under one, is a plain refusal."""
        (workspace_root / "f.txt").write_bytes(b"file")
        for target in ("f.txt", "f.txt/sub"):
            status, body = _upload(client, _zip({"a.txt": b"x"}), target)
            assert status == 400, body
            assert "'f.txt' is a file, not a folder" in str(body["detail"])
        assert _tree(workspace_root) == {"f.txt": b"file"}

    def test_a_refusal_removes_every_folder_made_for_the_target(
        self, workspace_root: Path, client: TestClient
    ) -> None:
        """A nested target that did not exist is gone again, all levels of it."""
        status, _ = _upload(client, _zip({"../evil.txt": b"x"}), "new/deeper/still")
        assert status == 400
        assert _tree(workspace_root) == {}


class TestZipAudit:
    """Every unpack and every refusal leaves one line in the audit log."""

    def test_an_unpack_is_logged(
        self, workspace_root: Path, client: TestClient, audit: pytest.LogCaptureFixture
    ) -> None:
        """What was unpacked, how much, and where."""
        del workspace_root
        status, _ = _upload(client, _zip({"a.txt": b"x"}), "in")
        assert status == 200
        lines = [r.getMessage() for r in audit.records if r.name == "medmcp.audit"]
        assert lines == ["zip upload: case.zip, 1 file(s) into in"]

    def test_each_refusal_is_logged_on_one_line(
        self,
        workspace_root: Path,
        outside: Path,
        client: TestClient,
        audit: pytest.LogCaptureFixture,
    ) -> None:
        """A refused member, a clash and a target outside the workspace all leave a line."""
        del outside
        (workspace_root / "a.txt").write_bytes(b"old")
        assert _upload(client, _zip({"x\ny/../../evil.txt": b"x"}))[0] == 400
        assert _upload(client, _zip({"a.txt": b"new"}))[0] == 409
        assert _upload(client, _zip({"b.txt": b"x"}), "../outside")[0] == 400
        assert _upload(client, b"garbage")[0] == 400
        lines = [r.getMessage() for r in audit.records if r.name == "medmcp.audit"]
        assert len(lines) == 4
        assert all(line.startswith("zip upload refused (") and "\n" not in line for line in lines)
        assert "(400): case.zip into .: 'x?y/../../evil.txt': '..' is not allowed" in lines[0]
        assert "(409): case.zip into .: 'a.txt' already exists here" in lines[1]
        assert "into ../outside: path escapes the workspace" in lines[2]
        assert "not a valid .zip archive" in lines[3]
