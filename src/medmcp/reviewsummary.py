"""What a ``confirm_selection`` dialog shows: the review the call would confirm.

``confirm_selection`` (the cohort stack) writes the roll-out manifest from a
``review_selection`` table: it is the person's decision about which series go
into an analysis. In the rehearsals the model made that decision itself, calling
``confirm_selection`` straight after the review with a note it invented ("User
approved selection as is"). So the core never lets a plan's Go cover it, and its
dialog shows the review itself, read here from the ``<name>_review.json`` beside
the file the call names: how many rows per status, and the rows that still need
a decision. The person reads what they confirm in the dialog that confirms it.

Read-only and best effort: a review that cannot be read is said so in the dialog
(the call itself would then fail in the stack too), never an error here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

# The review_selection outputs a call may name; the record is <name>_review.json.
_SUFFIXES = ("_review.csv", "_review.html", "_review.json")
_KIND = "medmcp-cohort review_selection"
# Statuses a confirm refuses to pass without an override (medmcp-cohort's UNRESOLVED).
UNRESOLVED = ("needs_decision", "multiple_stations")
# Rows listed in the dialog; the rest are counted.
MAX_ROWS = 10
# Candidates named per listed row.
MAX_CANDIDATES = 4
# A review record larger than this is not read (the dialog says so).
MAX_BYTES = 64 * 1024 * 1024
_MAX_REASON = 240


def _record_path(review_path: str, workspace: Path) -> Path:
    raw = Path(review_path) if Path(review_path).is_absolute() else workspace / review_path
    path = Path(os.path.normpath(str(raw)))
    for suffix in _SUFFIXES:
        if path.name.endswith(suffix):
            return path.with_name(path.name.removesuffix(suffix) + "_review.json")
    raise ValueError(f"{review_path} is not a review_selection output (<name>_review.csv)")


def _series(candidate: object) -> str:
    if not isinstance(candidate, dict):
        return ""
    data = cast("JsonDict", candidate)
    number = str(data.get("series_number") or "").strip()
    description = str(data.get("description") or "").strip()
    text = " ".join(t for t in (f"#{number}" if number else "", description) if t)
    return text or str(data.get("path") or "").rsplit("/", 1)[-1]


def _row(raw: JsonDict) -> JsonDict:
    reason = str(raw.get("reason") or "")
    if len(reason) > _MAX_REASON:
        reason = reason[: _MAX_REASON - 1] + "…"
    candidates = [
        s for s in (_series(c) for c in cast("list[object]", raw.get("candidates") or [])) if s
    ]
    return {
        "subject": str(raw.get("subject_id") or ""),
        "session": str(raw.get("session_id") or ""),
        "role": str(raw.get("role") or ""),
        "status": str(raw.get("status") or ""),
        "reason": reason,
        "candidates": candidates[:MAX_CANDIDATES],
        "moreCandidates": max(0, len(candidates) - MAX_CANDIDATES),
    }


def summarize(arguments: JsonDict, workspace: Path) -> JsonDict:
    """The review a ``confirm_selection`` call refers to, for its dialog.

    Returns ``{"path", "counts", "rows", "rowsTotal", "overrides", "note",
    "error"}``: *counts* rows per status, *rows* up to :data:`MAX_ROWS` rows that
    need a decision (``needs_decision``, ``multiple_stations``) out of
    *rowsTotal*, *overrides* how many decisions the call passes, *note* the note
    it would record. *error* says why the review could not be read (the rest is
    then empty); the path must lie in the workspace.
    """
    out: JsonDict = {
        "path": "",
        "counts": {},
        "rows": [],
        "rowsTotal": 0,
        "overrides": 0,
        "note": str(arguments.get("note") or ""),
        "error": None,
    }
    overrides = arguments.get("overrides")
    if isinstance(overrides, list):
        out["overrides"] = len(cast("list[object]", overrides))
    review_path = arguments.get("review_path")
    if not isinstance(review_path, str) or not review_path.strip():
        out["error"] = "The call names no review (review_path is missing)."
        return out
    try:
        path = _record_path(review_path.strip(), workspace)
    except ValueError as exc:
        out["error"] = f"The review could not be read: {exc}."
        return out
    out["path"] = str(path)
    root = Path(os.path.normpath(str(workspace)))
    try:
        real = path.resolve()
        if not real.is_relative_to(root.resolve()):
            out["error"] = f"The review could not be read: {path} is outside the workspace."
            return out
        if real.stat().st_size > MAX_BYTES:
            out["error"] = f"The review could not be read: {path.name} is too large to show."
            return out
        record = json.loads(real.read_text(encoding="utf-8"))
    except FileNotFoundError:
        out["error"] = f"The review could not be read: {path.name} was not found."
        return out
    except (OSError, ValueError) as exc:
        out["error"] = f"The review could not be read: {path.name}: {exc}."
        return out
    if not isinstance(record, dict) or cast("JsonDict", record).get("kind") != _KIND:
        out["error"] = (
            f"The review could not be read: {path.name} is not a review_selection record."
        )
        return out
    data = cast("JsonDict", record)
    rows = [
        cast("JsonDict", r)
        for r in cast("list[object]", data.get("rows") or [])
        if isinstance(r, dict)
    ]
    counts = data.get("counts")
    if isinstance(counts, dict):
        out["counts"] = {
            str(k): v for k, v in cast("JsonDict", counts).items() if isinstance(v, int)
        }
    else:
        tally: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status") or "")
            tally[status] = tally.get(status, 0) + 1
        out["counts"] = tally
    open_rows = [r for r in rows if r.get("status") in UNRESOLVED]
    out["rowsTotal"] = len(open_rows)
    out["rows"] = [_row(r) for r in open_rows[:MAX_ROWS]]
    return out
