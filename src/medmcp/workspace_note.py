"""The viewer "workspace context" note and its inverse.

When a file is open in the viewer, the workspace server sends a
``[workspace context: …]`` note with the prompt (as a content block flagged
``automatic``, which vibe ≥2.23 keeps out of auto-title derivation). The note
lets the agent resolve references like "this image" *and* hands it the absolute
on-disk path the stack tools expect (the workspace is bind-mounted at the
identical absolute path — path parity). The note is live-turn metadata, not part
of the user's request: replayed turns carry the note-free text natively (the
prompt's ``user_display_content`` meta), while the persisted prompt text still
contains the note, so :func:`strip_workspace_note` remains the fallback for
pre-2.23 transcripts and for distillation's seed request.

The note's format (:func:`build_workspace_note`) and the pattern that removes it
(:func:`strip_workspace_note`) live here together so they stay in sync — they
were previously duplicated across ``server.py`` and ``distill.py``.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

# The note is always appended last, separated by a blank line, so the strip
# pattern cuts from its marker to the end of the text — this also handles a title
# truncated mid-note (no closing "]"). The "\n\n" anchor avoids touching a
# bracketed phrase a user happened to type inline.
_NOTE_RE = re.compile(r"\n\n\[workspace context:.*$", re.DOTALL)


def build_workspace_note(absolute_path: str) -> str:
    """Build the ``[workspace context: …]`` note for the file open in the viewer.

    *absolute_path* must already be resolved to its on-disk absolute path (path
    parity) — handing the agent the absolute path lets its first tool call hit
    instead of recovering after a filesystem search.
    """
    return (
        f'\n\n[workspace context: the file "{absolute_path}" is currently open in the '
        'viewer; references like "this image" or "the current image" mean that file]'
    )


def strip_workspace_note(text: str) -> str:
    """Remove the appended ``[workspace context: …]`` note from *text*."""
    return _NOTE_RE.sub("", text).strip()


def display_content_text(display: dict[str, Any]) -> str:
    """Join the text blocks of a ``user_display_content`` payload (``""`` if none).

    The payload is the ``{version, host, content: [...]}`` dict the server sends
    with a prompt; vibe persists it on the user message and echoes it in replayed
    ``user_message_chunk`` frames, so both the server (replay) and distillation
    (the seed request) can recover the note-free user text from it.
    """
    blocks = display.get("content")
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in cast("list[object]", blocks):
        if isinstance(block, dict):
            b = cast("dict[str, Any]", block)
            if b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
    return "\n\n".join(p for p in parts if p).strip()


# ── dataset notes ────────────────────────────────────────────────────────────────
# A stack may keep an index of a folder of scans: a table with one row per scan and,
# beside it, a small summary file. It says so in its image label (``dataset_index``,
# see :func:`parse_dataset_index`). When a prompt refers to a workspace folder and
# an active stack declares such an index, the agent is told what the summary says —
# or that there is none yet and which tool builds it — so it does not explore the
# folder by shell first. Without a declaring stack nothing is added: the core knows
# no stack's files or tools by name. Like the viewer note above, each line starts
# with ``[workspace context:`` and is therefore removed by ``strip_workspace_note``.

# At most this many folders are described per prompt; a prompt naming more is rare
# and the note must stay a few lines.
_MAX_DATASET_NOTES = 3

# How many directory entries the "does this folder hold scans?" check may visit.
_SCAN_PROBE_BUDGET = 3000

# How much of a summary's own description is quoted.
_MAX_DESCRIPTION_CHARS = 400

# Folder-name words too common to identify a folder on their own.
_GENERIC_NAME_TOKENS = frozenset(
    {"data", "dataset", "datasets", "master", "main", "test", "tests", "testdata", "raw", "new"}
)

_ABS_PATH_RE = re.compile(r"(/[^\s\"'`<>|]+)")
_IMAGE_SUFFIXES = (".nii", ".nii.gz", ".dcm", ".ima")
_TOOL_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_MAX_QUERY_TOOLS = 4


@dataclass(frozen=True)
class DatasetIndex:
    """A stack's declaration that it indexes folders of scans.

    Attributes:
        stack: The declaring stack.
        summary: Where the summary lives, relative to the indexed folder. A JSON
            object; its optional ``description`` string is what the agent is told.
        build: The tool that indexes a folder.
        query: The tools that answer from an index.
    """

    stack: str
    summary: PurePosixPath
    build: str
    query: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        """The declaration as it is stored in a stack manifest."""
        return {"summary": str(self.summary), "build": self.build, "query": list(self.query)}


def parse_dataset_index(stack: str, raw: object) -> DatasetIndex | None:
    """Read a ``dataset_index`` declaration, or ``None`` when absent or malformed.

    The shape, in the image label or the manifest::

        {"summary": ".index/summary.json", "build": "build_index",
         "query": ["summarize_index", "pick_scans"]}

    ``summary`` must be a relative path that stays inside the folder; ``build`` is
    the tool that indexes a folder; ``query`` lists up to four tools that answer
    from an index (may be empty). Tool names are plain identifiers, since they are
    written into the prompt. A declaration that does not fit is dropped and the
    stack installs without it.

    The summary file is a JSON object; the core reads only ``description``, one
    line on what the folder holds (e.g. ``"3 subjects, 18 sessions, 315 scans"``),
    quoted to the assistant on one line, square brackets replaced, cut at 400
    characters. With a summary the assistant is told the folder has a table of
    its scans and to call the ``query`` tools instead of reading its files; a
    folder with scans but no summary, to call ``build`` first; a folder without
    scans, nothing.
    """
    if not isinstance(raw, dict):
        return None
    decl = cast("dict[str, Any]", raw)
    summary, build, query = decl.get("summary"), decl.get("build"), decl.get("query", [])
    if not isinstance(summary, str) or not isinstance(build, str) or not isinstance(query, list):
        return None
    relpath = PurePosixPath(summary.strip())
    if not relpath.parts or relpath.is_absolute() or ".." in relpath.parts:
        return None
    tools = cast("list[object]", query)
    names = [build, *tools]
    if len(tools) > _MAX_QUERY_TOOLS or not all(
        isinstance(n, str) and _TOOL_NAME_RE.fullmatch(n) for n in names
    ):
        return None
    return DatasetIndex(stack, relpath, build, tuple(cast("list[str]", tools)))


def dataset_indexes(servers: Iterable[dict[str, Any]]) -> list[DatasetIndex]:
    """The dataset indexes the given (active) stacks declare."""
    found: list[DatasetIndex] = []
    for server in servers:
        index = parse_dataset_index(str(server.get("name", "")), server.get("dataset_index"))
        if index is not None:
            found.append(index)
    return found


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _mentions_folder(prompt_tokens: set[str], prompt_flat: str, folder: str) -> bool:
    """Whether a prompt names the top-level workspace folder *folder*.

    People type "the atss testdata folder" for ``atss_testdata`` and "the open ms
    dataset" for ``open_ms_data-master``, so the match ignores punctuation and the
    generic words of a folder name. A single short distinctive word is not enough.
    """
    tokens = _tokens(folder)
    if not tokens:
        return False
    if len(" ".join(tokens)) >= 5 and " ".join(tokens) in prompt_flat:
        return True
    distinctive = [t for t in tokens if t not in _GENERIC_NAME_TOKENS]
    if not distinctive or not all(t in prompt_tokens for t in distinctive):
        return False
    return len(distinctive) >= 2 or len(distinctive[0]) >= 5


def _holds_scans(folder: Path) -> bool:
    """A bounded look for imaging files, so a folder of spreadsheets is left alone."""
    seen = 0
    for _root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            seen += 1
            lower = name.lower()
            if lower.endswith(_IMAGE_SUFFIXES) or lower == "dicomdir":
                return True
        seen += len(dirs)
        if seen > _SCAN_PROBE_BUDGET:
            return False
    return False


def _indexed_ancestor(
    path: Path, workspace_root: Path, indexes: Sequence[DatasetIndex]
) -> Path | None:
    """The nearest folder at or above *path*, inside the workspace, that has an index."""
    current = path if path.is_dir() else path.parent
    while current != workspace_root and workspace_root in current.parents:
        if any((current / index.summary).is_file() for index in indexes):
            return current
        current = current.parent
    return None


def find_dataset_dirs(
    text: str,
    viewed_path: str | None,
    workspace_root: Path,
    indexes: Sequence[DatasetIndex],
) -> list[Path]:
    """Workspace folders a prompt refers to, most specific mention first.

    Three sources: absolute paths written in the prompt, top-level workspace
    folders named in it, and the file open in the viewer. A path with an indexed
    folder at or above it resolves to that folder; otherwise a named directory
    stands for itself. Nothing outside *workspace_root* is ever returned.
    """
    root = workspace_root.resolve()
    found: list[Path] = []

    def add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if resolved == root or root not in resolved.parents or not resolved.exists():
            return
        target = _indexed_ancestor(resolved, root, indexes) or (
            resolved if resolved.is_dir() else None
        )
        if target is not None and target not in found:
            found.append(target)

    for match in _ABS_PATH_RE.findall(text):
        add(Path(match.rstrip(".,;:)!?")))
    # A folder is named by the words of the prompt, not by the parts of a path
    # written in it: "results to /ws/stack-tests/run-5/" names no "stack-tests".
    # The path itself was handled above (and is skipped there when it does not
    # exist yet, as an output folder usually does not).
    flat = " ".join(_tokens(_ABS_PATH_RE.sub(" ", text)))
    prompt_tokens = set(flat.split())
    try:
        top_level = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        top_level = []
    for folder in top_level:
        if _mentions_folder(prompt_tokens, flat, folder.name):
            add(folder)
    if viewed_path and Path(viewed_path).exists():
        indexed = _indexed_ancestor(Path(viewed_path), root, indexes)
        if indexed is not None and indexed not in found:
            found.append(indexed)
    return found[:_MAX_DATASET_NOTES]


def _summary_description(summary_file: Path) -> str | None:
    """The summary's own one-line description; ``""`` without one, ``None`` if unreadable.

    The text is the stack's, so it is flattened to one line, kept free of the
    brackets that delimit a note, and clipped.
    """
    try:
        loaded: object = json.loads(summary_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    text = cast("dict[str, Any]", loaded).get("description")
    if not isinstance(text, str):
        return ""
    flat = " ".join(text.replace("[", "(").replace("]", ")").split())
    if len(flat) > _MAX_DESCRIPTION_CHARS:
        flat = flat[: _MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
    return flat


def _tool_list(index: DatasetIndex, tools: Sequence[str]) -> str:
    return f"{' or '.join(tools)} ({index.stack})"


def describe_dataset(folder: Path, indexes: Sequence[DatasetIndex]) -> str | None:
    """One ``[workspace context: …]`` line for *folder*, or ``None`` when it needs none.

    An indexed folder is described from its summary; a folder that holds scans
    but has no index gets the instruction to build one first; a folder without
    imaging files, or no stack that indexes folders, gets nothing.
    """
    if not indexes:
        return None
    for index in indexes:
        summary_file = folder / index.summary
        if not summary_file.is_file():
            continue
        description = _summary_description(summary_file)
        if description is None:
            continue
        said = f" ({description})" if description else ""
        ask = (
            f" For any question about what it contains, or to pick scans, call "
            f'{_tool_list(index, index.query)} with "{folder}";'
            if index.query
            else ""
        )
        return (
            f'[workspace context: the folder "{folder}" already has a table of its scans'
            f"{said}.{ask} do not list or read its files yourself]"
        )
    if not _holds_scans(folder):
        return None
    index = indexes[0]
    then = f", then {' or '.join(index.query)}" if index.query else ""
    return (
        f'[workspace context: the folder "{folder}" holds scans but has no table of them yet. '
        "Before answering anything about its contents or selecting scans, call "
        f'{_tool_list(index, [index.build])} for "{folder}"{then}; do not list or read its '
        "files yourself]"
    )


def build_dataset_notes(
    text: str,
    viewed_path: str | None,
    workspace_root: Path,
    indexes: Sequence[DatasetIndex],
) -> str:
    """The dataset lines for one prompt, joined by blank lines (``""`` when none).

    *indexes* are the declarations of the active stacks (:func:`dataset_indexes`);
    with none, no folder is looked at and nothing is added.
    """
    if not indexes:
        return ""
    lines = [
        line
        for folder in find_dataset_dirs(text, viewed_path, workspace_root, indexes)
        if (line := describe_dataset(folder, indexes)) is not None
    ]
    return "\n\n".join(lines)
