"""Safe extraction of an uploaded ``.zip`` into a workspace folder.

The explorer's folder upload depends on the browser handing over a folder's
files, which a sandboxed browser (snap or flatpak Firefox, going through the
desktop portal) does not always do. A zip is one plain file and always arrives,
so the server unpacks it instead.

Nothing in the archive is trusted. Every member is checked before anything is
written — absolute names, ``..``, links, device files, encrypted members,
unsupported compression, a name given twice and archives over the file-count or
size caps are refused. An unpack never replaces anything: an archive naming a
file that already exists in the target folder is refused whole. Files are
created exclusively through directory descriptors opened with ``O_NOFOLLOW``,
so a link already sitting in the target folder cannot redirect a write, and an
unpack that fails midway removes every file and folder it had made.
"""

from __future__ import annotations

import contextlib
import errno
import lzma
import os
import re
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
# Folders nested deeper than any real dataset; each level holds a descriptor open.
MAX_DEPTH: int = 64
# The longest single name common filesystems take, in bytes.
MAX_NAME_BYTES: int = 255

_CHUNK: int = 1024 * 1024
# Finder's resource-fork shadow tree and folder metadata: noise, not data.
_JUNK_DIRS: frozenset[str] = frozenset({"__MACOSX"})
_JUNK_FILES: frozenset[str] = frozenset({".DS_Store", "Thumbs.db"})
# What this Python's zipfile can decompress (Deflate64, PPMd, … it cannot).
_COMPRESSION: frozenset[int] = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}
)
# A Windows drive prefix ("C:/…", "C:\…" or a bare "C:"); a colon anywhere else
# is an ordinary character of the name.
_DRIVE_PREFIX: re.Pattern[str] = re.compile(r"^[A-Za-z]:(/|$)")
# How many clashing names a refusal spells out before it only counts them.
_CLASHES_NAMED: int = 5


class ZipRejectedError(Exception):
    """The archive was refused; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        """Keep the client-facing *message* and its HTTP *status*."""
        super().__init__(message)
        self.status = status


class _NotAFolderError(ZipRejectedError):
    """A path the archive needs as a folder exists as a file or a link."""

    def __init__(self, parts: tuple[str, ...]) -> None:
        """Remember the offending path, relative to the target folder."""
        self.path = "/".join(parts)
        super().__init__(f"'{self.path}' already exists here and is not a folder", status=409)


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
    if normalized.startswith("/") or _DRIVE_PREFIX.match(normalized):
        raise ZipRejectedError(f"'{name}': absolute paths are not allowed in the archive")
    parts = tuple(p for p in PurePosixPath(normalized).parts if p not in {"", "."})
    if ".." in parts:
        raise ZipRejectedError(f"'{name}': '..' is not allowed in archive paths")
    if len(parts) > MAX_DEPTH:
        raise ZipRejectedError(f"'{name}': folders are nested more than {MAX_DEPTH} deep")
    if any(len(os.fsencode(p)) > MAX_NAME_BYTES for p in parts):
        raise ZipRejectedError(f"'{name[:60]}…': a name in this path is too long")
    return parts


def check_members(
    zf: zipfile.ZipFile, *, max_files: int | None = None, max_bytes: int | None = None
) -> list[Member]:
    """Validate every member of *zf* and return the ones to extract.

    The caps default to :data:`MAX_FILES` and :data:`MAX_BYTES`. Only the
    archive's directory is read, none of the members' data.

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
            if info.compress_type not in _COMPRESSION:
                raise ZipRejectedError(
                    f"'{info.filename}': its compression method is not supported; "
                    "pack the archive again with standard (Deflate) compression"
                )
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
    _check_names(members)
    return members


def _check_names(members: list[Member]) -> None:
    """Refuse an archive naming a file twice, or one name as both file and folder.

    Raises:
        ZipRejectedError: naming the first such path.
    """
    files: set[tuple[str, ...]] = set()
    folders: set[tuple[str, ...]] = set()
    for member in members:
        if member.is_dir:
            folders.add(member.parts)
        elif member.parts in files:
            raise ZipRejectedError(f"'{'/'.join(member.parts)}' is in the archive more than once")
        else:
            files.add(member.parts)
        for depth in range(1, len(member.parts)):
            folders.add(member.parts[:depth])
    for path in sorted(files & folders):
        raise ZipRejectedError(f"'{'/'.join(path)}' is both a file and a folder in the archive")


class _Walker:
    """Reaches folders under one root through descriptors, never following a link.

    Only the chain of folders down to the one last asked for stays open, so the
    number of descriptors held is the nesting depth, not the folder count.
    """

    def __init__(self, root_fd: int) -> None:
        """Start from the already-open *root_fd* (owned by the caller)."""
        self._root = root_fd
        self._chain: list[tuple[str, int]] = []
        # Folders this walker created, in creation order (parents first).
        self.made: list[tuple[str, ...]] = []

    def _descend(self, parts: tuple[str, ...], *, create: bool) -> int | None:
        keep = 0
        for (name, _), part in zip(self._chain, parts, strict=False):
            if name != part:
                break
            keep += 1
        self._pop_to(keep)
        fd = self._chain[-1][1] if self._chain else self._root
        for depth in range(keep, len(parts)):
            name = parts[depth]
            if create:
                try:
                    os.mkdir(name, 0o777, dir_fd=fd)
                    self.made.append(parts[: depth + 1])
                except FileExistsError:
                    pass
            try:
                fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if create:
                    raise
                return None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise _NotAFolderError(parts[: depth + 1]) from exc
                raise
            self._chain.append((name, fd))
        return fd

    def find(self, parts: tuple[str, ...]) -> int | None:
        """The descriptor of the existing folder *parts*, or None when it is missing.

        Raises:
            _NotAFolderError: a component exists as a file or a link.
        """
        return self._descend(parts, create=False)

    def make(self, parts: tuple[str, ...]) -> int:
        """The descriptor of folder *parts*, creating the missing folders on the way.

        Raises:
            _NotAFolderError: a component exists as a file or a link.
        """
        fd = self._descend(parts, create=True)
        assert fd is not None  # a missing folder raises when creating
        return fd

    def _pop_to(self, depth: int) -> None:
        while len(self._chain) > depth:
            with contextlib.suppress(OSError):
                os.close(self._chain.pop()[1])

    def close(self) -> None:
        """Close every descriptor this opened (not the caller's root)."""
        self._pop_to(0)


def _existing(walker: _Walker, members: list[Member]) -> list[str]:
    """The paths *members* would write that already exist under the walker's root.

    A file member clashes with anything of its name; an existing folder is only
    in the way where a file is to go, and a file or a link where a folder is.
    """
    clashes: dict[str, None] = {}
    for member in members:
        try:
            if member.is_dir:
                walker.find(member.parts)
                continue
            parent = walker.find(member.parts[:-1])
        except _NotAFolderError as exc:
            clashes[exc.path] = None
            continue
        if parent is None:
            continue
        try:
            os.stat(member.parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            continue
        clashes["/".join(member.parts)] = None
    return list(clashes)


def _clash_message(clashes: list[str]) -> str:
    """The refusal naming the first few of *clashes* and counting the rest."""
    named = ", ".join(f"'{c}'" for c in clashes[:_CLASHES_NAMED])
    more = len(clashes) - _CLASHES_NAMED
    if more > 0:
        named += f" and {more} more"
    verb = "already exists" if len(clashes) == 1 else "already exist"
    return f"{named} {verb} here; nothing was unpacked"


def _remove(walker: _Walker, parts: tuple[str, ...], *, is_dir: bool) -> None:
    """Best-effort removal of one path this unpack created."""
    with contextlib.suppress(OSError, ZipRejectedError):
        parent = walker.find(parts[:-1])
        if parent is None:
            return
        if is_dir:
            os.rmdir(parts[-1], dir_fd=parent)
        else:
            os.unlink(parts[-1], dir_fd=parent)


def _roll_back(walker: _Walker, files: list[tuple[str, ...]]) -> None:
    """Remove the *files* an unpack created and the folders the walker made."""
    for parts in reversed(files):
        _remove(walker, parts, is_dir=False)
    for parts in reversed(walker.made):
        _remove(walker, parts, is_dir=True)


def extract(zf: zipfile.ZipFile, members: list[Member], root_fd: int) -> Extracted:
    """Write the checked *members* of *zf* under the directory open as *root_fd*.

    All or nothing. Nothing is written when a member would land on something
    that already exists; every file is created exclusively, so an existing file
    is never opened, let alone truncated; and when anything fails midway the
    files and folders made so far are removed before the error is passed on.

    Raises:
        ZipRejectedError: a member's path already exists (status 409).
    """
    walker = _Walker(root_fd)
    made_files: list[tuple[str, ...]] = []
    written = 0
    try:
        clashes = _existing(walker, members)
        if clashes:
            raise ZipRejectedError(_clash_message(clashes), status=409)
        for member in members:
            if member.is_dir:
                walker.make(member.parts)
                continue
            parent = walker.make(member.parts[:-1])
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            fd = os.open(member.parts[-1], flags, 0o666, dir_fd=parent)
            made_files.append(member.parts)
            with os.fdopen(fd, "wb") as dst, zf.open(member.info) as src:
                written += _copy(src, dst)
        return Extracted(files=len(made_files), dirs=len(walker.made), bytes=written)
    except BaseException:
        _roll_back(walker, made_files)
        raise
    finally:
        walker.close()


def _copy(src: IO[bytes], dst: IO[bytes]) -> int:
    """Copy *src* to *dst* in chunks; return the number of bytes copied."""
    n = 0
    while chunk := src.read(_CHUNK):
        dst.write(chunk)
        n += len(chunk)
    return n


def _os_rejection(exc: OSError) -> ZipRejectedError:
    """The refusal to answer with when the filesystem turned a write down."""
    if isinstance(exc, FileExistsError):
        # Something appeared between the check and the write.
        return ZipRejectedError(
            f"'{exc.filename}' already exists here; nothing was unpacked", status=409
        )
    if exc.errno in {errno.ENOSPC, errno.EDQUOT}:
        return ZipRejectedError("no space left to unpack the archive", status=507)
    if exc.errno == errno.ENAMETOOLONG:
        return ZipRejectedError("a name in the archive is too long")
    if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
        return ZipRejectedError("the target folder cannot be written to", status=403)
    reason = exc.strerror or str(exc)
    return ZipRejectedError(f"the archive could not be unpacked: {reason}", status=500)


def extract_archive(
    archive: IO[bytes],
    target_dir: Path,
    *,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> Extracted:
    """Check the zip in *archive* (seekable) and unpack it into *target_dir*.

    Nothing is written unless every member passes, none of them exists in
    *target_dir* yet and the unpacked size fits on the disk; a failure while
    writing leaves *target_dir* as it was (see :func:`extract`).

    Raises:
        ZipRejectedError: not a zip, a refused member, a cap exceeded, a file
            that already exists, no room, a damaged archive, or a write the
            filesystem refused.
    """
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ZipRejectedError("not a valid .zip archive") from exc
    with zf:
        members = check_members(zf, max_files=max_files, max_bytes=max_bytes)
        if not any(not m.is_dir for m in members):
            raise ZipRejectedError("the archive holds no files")
        try:
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
            finally:
                os.close(root_fd)
        except (zipfile.BadZipFile, zlib.error, lzma.LZMAError, EOFError) as exc:
            raise ZipRejectedError(f"the archive is damaged: {exc}") from exc
        except (NotImplementedError, RuntimeError) as exc:
            raise ZipRejectedError(f"the archive cannot be unpacked: {exc}") from exc
        except OSError as exc:
            raise _os_rejection(exc) from exc


def unpack_into(archive: IO[bytes], target_dir: Path) -> Extracted:
    """Unpack the zip in *archive* into *target_dir*, creating that folder if needed.

    A refused or failed unpack also removes the folders created here for the
    target, so it leaves nothing behind.

    Raises:
        ZipRejectedError: as :func:`extract_archive`, or *target_dir* is (or
            lies under) something that is not a folder.
    """
    missing: list[Path] = []
    probe = target_dir
    while not os.path.lexists(probe):
        missing.append(probe)
        probe = probe.parent
    if not probe.is_dir():
        raise ZipRejectedError(f"'{probe.name}' is a file, not a folder")
    try:
        for folder in reversed(missing):
            folder.mkdir()
        return extract_archive(archive, target_dir)
    except BaseException as exc:
        for folder in missing:
            with contextlib.suppress(OSError):
                folder.rmdir()
        if isinstance(exc, OSError):
            raise _os_rejection(exc) from exc
        raise
