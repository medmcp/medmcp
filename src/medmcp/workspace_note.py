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
from pathlib import Path
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
# The cohort stack's ``index_dataset`` writes ``<folder>/.medmcp/dataset_summary.json``
# next to its table of scans. When a prompt refers to a workspace folder, the agent
# is told what that table already says — or that there is none yet — so it does not
# explore the folder by shell first. Like the viewer note above, each line starts
# with ``[workspace context:`` and is therefore removed by ``strip_workspace_note``.

DATASET_SUMMARY_RELPATH = Path(".medmcp") / "dataset_summary.json"

# At most this many folders are described per prompt; a prompt naming more is rare
# and the note must stay a few lines.
_MAX_DATASET_NOTES = 3

# How many directory entries the "does this folder hold scans?" check may visit.
_SCAN_PROBE_BUDGET = 3000

# Folder-name words too common to identify a folder on their own.
_GENERIC_NAME_TOKENS = frozenset(
    {"data", "dataset", "datasets", "master", "main", "test", "tests", "testdata", "raw", "new"}
)

_ABS_PATH_RE = re.compile(r"(/[^\s\"'`<>|]+)")
_IMAGE_SUFFIXES = (".nii", ".nii.gz", ".dcm", ".ima")


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


def _indexed_ancestor(path: Path, workspace_root: Path) -> Path | None:
    """The nearest folder at or above *path*, inside the workspace, that has an index."""
    current = path if path.is_dir() else path.parent
    while current != workspace_root and workspace_root in current.parents:
        if (current / DATASET_SUMMARY_RELPATH).is_file():
            return current
        current = current.parent
    return None


def find_dataset_dirs(text: str, viewed_path: str | None, workspace_root: Path) -> list[Path]:
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
        target = _indexed_ancestor(resolved, root) or (resolved if resolved.is_dir() else None)
        if target is not None and target not in found:
            found.append(target)

    for match in _ABS_PATH_RE.findall(text):
        add(Path(match.rstrip(".,;:)!?")))
    flat = " ".join(_tokens(text))
    prompt_tokens = set(flat.split())
    try:
        top_level = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        top_level = []
    for folder in top_level:
        if _mentions_folder(prompt_tokens, flat, folder.name):
            add(folder)
    if viewed_path:
        indexed = _indexed_ancestor(Path(viewed_path), root) if Path(viewed_path).exists() else None
        if indexed is not None and indexed not in found:
            found.append(indexed)
    return found[:_MAX_DATASET_NOTES]


def _tally(summary: dict[str, Any], key: str, limit: int = 6) -> str:
    counts = summary.get(key)
    if not isinstance(counts, dict) or not counts:
        return ""
    pairs = sorted(
        ((str(k), v) for k, v in cast("dict[str, Any]", counts).items() if isinstance(v, int)),
        key=lambda kv: -kv[1],
    )
    shown = ", ".join(f"{k} {v}" for k, v in pairs[:limit])
    return shown + (", …" if len(pairs) > limit else "")


def describe_dataset(folder: Path) -> str | None:
    """One ``[workspace context: …]`` line for *folder*, or ``None`` when it needs none.

    An indexed folder is described from its ``dataset_summary.json``; a folder
    that holds scans but has no index gets the instruction to build one first; a
    folder without imaging files gets nothing.
    """
    summary_file = folder / DATASET_SUMMARY_RELPATH
    if summary_file.is_file():
        try:
            summary = cast("dict[str, Any]", json.loads(summary_file.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            summary = None
        if isinstance(summary, dict):
            parts = [
                f"{summary.get('n_subjects', '?')} subjects, {summary.get('n_sessions', '?')} "
                f"sessions, {summary.get('n_series', '?')} scans"
            ]
            for label, key in (
                ("sequences", "sequences"),
                ("orientation", "planes"),
                ("contrast", "contrast"),
            ):
                tally = _tally(summary, key)
                if tally:
                    parts.append(f"{label}: {tally}")
            coverage = summary.get("date_coverage")
            if coverage:
                parts.append(f"dates: {coverage}")
            unknown = summary.get("n_unknown_sequence")
            if isinstance(unknown, int) and unknown:
                parts.append(f"{unknown} scans with unknown sequence")
            return (
                f'[workspace context: the folder "{folder}" already has a table of its scans '
                f"({'; '.join(parts)}). For any question about what it contains, how far apart "
                "the sessions are, or to pick scans, call summarize_dataset or select_series "
                f'with dataset "{folder}"; do not list or read its files yourself]'
            )
    if not _holds_scans(folder):
        return None
    return (
        f'[workspace context: the folder "{folder}" holds scans but has no table of them yet. '
        "Before answering anything about its contents or selecting scans, call index_dataset "
        f'with dataset_dir "{folder}", then summarize_dataset or select_series; do not list '
        "or read its files yourself]"
    )


def build_dataset_notes(text: str, viewed_path: str | None, workspace_root: Path) -> str:
    """The dataset lines for one prompt, joined by blank lines (``""`` when none)."""
    lines = [
        line
        for folder in find_dataset_dirs(text, viewed_path, workspace_root)
        if (line := describe_dataset(folder)) is not None
    ]
    return "\n\n".join(lines)
