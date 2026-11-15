"""Safe extraction of an uploaded ``.zip`` into a workspace folder.

The explorer's folder upload depends on the browser handing over a folder's
files, which a sandboxed browser (snap or flatpak Firefox, going through the
desktop portal) does not always do. A zip is one plain file and always arrives,
so the server unpacks it instead.

Nothing in the archive is trusted. Every member is checked before anything is
written — absolute names, ``..``, links, device files and encrypted members are
refused, as are archives over the file-count or size caps — and the files are
then written through directory descriptors opened with ``O_NOFOLLOW``, so a
link already sitting in the target folder cannot redirect a write either.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import stat
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO

# Generous for imaging data, but a bound: an archive declaring more is refused
# before anything is written.
MAX_FILES: int = 100_000
MAX_BYTES: int = 64 * 1024**3

_CHUNK: int = 1024 * 1024
# Finder's resource-fork shadow tree and folder metadata: noise, not data.
_JUNK_DIRS: frozenset[str] = frozenset({"__MACOSX"})
_JUNK_FILES: frozenset[str] = frozenset({".DS_Store", "Thumbs.db"})


class ZipRejectedError(Exception):
    """The archive was refused; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        """Keep the client-facing *message* and its HTTP *status*."""
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Member:
    """One archive entry that passed the checks, with its path split into parts."""

    info: zipfile.ZipInfo
    parts: tuple[str, ...]
    is_dir: bool


@dataclass(frozen=True)
class Extracted:
    """What an extraction wrote."""

    files: int
    dirs: int
    bytes: int


def _human(n: int) -> str:
    """*n* bytes as a short human-readable size."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - loop always returns


def _member_parts(name: str) -> tuple[str, ...]:
    """The path components of archive name *name*, or raise if it is unsafe."""
    if "\x00" in name:
        raise ZipRejectedError(f"'{name}': the name contains a NUL byte")
    # Some Windows tools write backslashes; treat them as the separators they mean.
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        raise ZipRejectedError(f"'{name}': absolute paths are not allowed in the archive")
    parts = tuple(p for p in PurePosixPath(normalized).parts if p not in {"", "."})
    if ".." in parts:
        raise ZipRejectedError(f"'{name}': '..' is not allowed in archive paths")
    return parts


def check_members(
    zf: zipfile.ZipFile, *, max_files: int | None = None, max_bytes: int | None = None
) -> list[Member]:
    """Validate every member of *zf* and return the ones to extract.

    The caps default to :data:`MAX_FILES` and :data:`MAX_BYTES`.

    Raises:
        ZipRejectedError: naming the first offending member, or the cap exceeded.
    """
    max_files = MAX_FILES if max_files is None else max_files
    max_bytes = MAX_BYTES if max_bytes is None else max_bytes
    members: list[Member] = []
    files = 0
    total = 0
    for info in zf.infolist():
        parts = _member_parts(info.filename)
        if not parts or parts[0] in _JUNK_DIRS or parts[-1] in _JUNK_FILES:
            continue
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind == stat.S_IFLNK:
            raise ZipRejectedError(f"'{info.filename}': links are not allowed in the archive")
        is_dir = info.is_dir() or kind == stat.S_IFDIR
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ZipRejectedError(
                f"'{info.filename}': only regular files and folders can be extracted"
            )
        if info.flag_bits & 0x1:
            raise ZipRejectedError(f"'{info.filename}': encrypted archives are not supported")
        if not is_dir:
            files += 1
            total += info.file_size
            if files > max_files:
                raise ZipRejectedError(
                    f"the archive holds more than {max_files} files; split it into smaller ones",
                    status=413,
                )
            if total > max_bytes:
                raise ZipRejectedError(
                    f"the archive unpacks to more than {_human(max_bytes)}; "
                    "split it into smaller ones",
                    status=413,
                )
        members.append(Member(info=info, parts=parts, is_dir=is_dir))
    return members


class _DirFds:
    """Directory descriptors under one root, opened (and created) without following links."""

    def __init__(self, root_fd: int) -> None:
        """Start from the already-open *root_fd* (owned by the caller)."""
        self._fds: dict[tuple[str, ...], int] = {(): root_fd}
        self.created = 0

    def get(self, parts: tuple[str, ...]) -> int:
        """The descriptor of folder *parts*, creating missing folders on the way."""
        if parts in self._fds:
            return self._fds[parts]
        parent = self.get(parts[:-1])
        name = parts[-1]
        try:
            os.mkdir(name, 0o777, dir_fd=parent)
            self.created += 1
        except FileExistsError:
            pass
        try:
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ZipRejectedError(
                    f"'{PurePosixPath(*parts)}' already exists in the target folder "
                    "as a link or a file, not a folder"
                ) from exc
            raise
        self._fds[parts] = fd
        return fd

    def close(self) -> None:
        """Close every descriptor this opened (not the caller's root)."""
        for key, fd in self._fds.items():
            if key:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._fds = {}


def extract(zf: zipfile.ZipFile, members: list[Member], root_fd: int) -> Extracted:
    """Write the checked *members* of *zf* under the directory open as *root_fd*.

    Raises:
        ZipRejectedError: an entry would replace a link, or a folder is a file.
    """
    dirs = _DirFds(root_fd)
    files = 0
    written = 0
    try:
        for member in members:
            if member.is_dir:
                dirs.get(member.parts)
                continue
            parent = dirs.get(member.parts[:-1])
            name = member.parts[-1]
            try:
                fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                    0o666,
                    dir_fd=parent,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ZipRejectedError(
                        f"'{member.info.filename}': a link with that name exists "
                        "in the target folder"
                    ) from exc
                if exc.errno == errno.EISDIR:
                    raise ZipRejectedError(
                        f"'{member.info.filename}': a folder with that name exists "
                        "in the target folder"
                    ) from exc
                raise
            with os.fdopen(fd, "wb") as dst, zf.open(member.info) as src:
                written += _copy(src, dst)
            files += 1
    finally:
        created = dirs.created
        dirs.close()
    return Extracted(files=files, dirs=created, bytes=written)


def _copy(src: IO[bytes], dst: IO[bytes]) -> int:
    """Copy *src* to *dst* in chunks; return the number of bytes copied."""
    n = 0
    while chunk := src.read(_CHUNK):
        dst.write(chunk)
        n += len(chunk)
    return n


def extract_archive(
    archive: IO[bytes],
    target_dir: Path,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> Extracted:
    """Check the zip in *archive* (seekable) and unpack it into *target_dir*.

    Nothing is written unless every member passes and the unpacked size fits
    on the disk.

    Raises:
        ZipRejectedError: not a zip, a refused member, a cap exceeded, no room,
            or the archive is damaged.
    """
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ZipRejectedError("not a valid .zip archive") from exc
    with zf:
        members = check_members(zf, max_files=max_files, max_bytes=max_bytes)
        if not any(not m.is_dir for m in members):
            raise ZipRejectedError("the archive holds no files")
        need = sum(m.info.file_size for m in members)
        free = shutil.disk_usage(target_dir).free
        if need > free:
            raise ZipRejectedError(
                f"not enough free space: the archive unpacks to {_human(need)}, "
                f"{_human(free)} are free",
                status=507,
            )
        root_fd = os.open(target_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            return extract(zf, members, root_fd)
        except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
            raise ZipRejectedError(f"the archive is damaged: {exc}") from exc
        finally:
            os.close(root_fd)
