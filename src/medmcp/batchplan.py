"""Turn a cohort batch manifest into per-item replay bindings.

The cohort stack's ``plan_batch`` tool writes a reviewable ``batch_plan.csv`` —
one row per map unit (subject, or subject-session), each with the resolved input
files and a ``status``. This module converts that manifest into the ``runs`` a
:func:`medmcp.replay.run_batch` call consumes: a list of ``{in_N: value}``
bindings, one per *ok* row, with the recipe's ``{{in_N}}`` inputs matched to the
manifest's role columns.

Only ``ok`` rows become runs; flagged rows (``missing``/``ambiguous``) are
returned separately so the caller escalates them instead of silently feeding a
tool the wrong (or no) file — the deterministic roll-out has no LLM to notice.
This is the glue that lets a single-subject workflow, once distilled, be mapped
over a whole cohort without hand-writing the per-subject input list.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from medmcp.workflow import Recipe

# Manifest columns that are metadata, not resolvable input roles.
_RESERVED_COLUMNS: frozenset[str] = frozenset(
    {"subject", "session", "status", "reason", "output_dir"}
)


class BatchPlanError(Exception):
    """Raised when a manifest can't be mapped onto a recipe's inputs."""


@dataclass
class BatchBinding:
    """The result of binding a manifest to a recipe.

    Attributes:
        runs: One ``{in_N: value}`` binding per ``ok`` manifest row, ready for
            :func:`medmcp.replay.run_batch`.
        labels: A ``subject[/session]`` label per run (aligned with ``runs``) for
            progress reporting.
        skipped: The flagged rows (each a ``{subject, session?, status, reason}``
            dict) that were not turned into runs.
        column_map: The resolved ``in_N -> manifest-column`` mapping used.
    """

    runs: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    labels: list[str] = field(default_factory=list[str])
    skipped: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    column_map: dict[str, str] = field(default_factory=dict[str, str])


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a ``batch_plan.csv`` manifest into a list of row dicts."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def _role_columns(rows: list[dict[str, str]]) -> list[str]:
    """The manifest's resolvable input-role columns (order preserved, no metadata)."""
    if not rows:
        return []
    return [c for c in rows[0] if c not in _RESERVED_COLUMNS]


def infer_column_map(recipe: Recipe, rows: list[dict[str, str]]) -> dict[str, str]:
    """Match each recipe input (``in_N``) to a manifest role column.

    Two strategies, in order, applied per input:

    1. **Exact example** — the column whose value in some row equals the input's
       recorded ``example`` path (unambiguous when the prototype subject is in the
       cohort).
    2. **Role token** — the column whose name appears in the example's file name
       (e.g. example ``…_FLAIR.nii.gz`` → column ``flair``), accepted only when a
       single column matches.

    Returns the inferred ``in_N -> column`` map (possibly partial); inputs that
    stay unresolved are reported by :func:`runs_from_manifest`.
    """
    columns = _role_columns(rows)
    mapping: dict[str, str] = {}
    for inp in recipe.inputs:
        example = inp.example
        exact = [c for c in columns if any(row.get(c) == example for row in rows)]
        if len(exact) == 1:
            mapping[inp.name] = exact[0]
            continue
        token = Path(example).name.lower()
        loose = [c for c in columns if c.lower() in token]
        if len(loose) == 1:
            mapping[inp.name] = loose[0]
    return mapping


def runs_from_manifest(
    recipe: Recipe,
    rows: list[dict[str, str]],
    *,
    column_map: dict[str, str] | None = None,
) -> BatchBinding:
    """Build per-item replay bindings from a manifest, skipping flagged rows.

    Args:
        recipe: The distilled workflow whose ``{{in_N}}`` inputs are bound per row.
        rows: Manifest rows from :func:`read_manifest`.
        column_map: Optional explicit ``in_N -> manifest-column`` mapping; any
            input not covered is inferred via :func:`infer_column_map`.

    Returns:
        A :class:`BatchBinding`.

    Raises:
        BatchPlanError: If any recipe input can't be matched to a manifest column
            (the caller should re-run with an explicit ``column_map``), or a
            mapped column is absent from the manifest.
    """
    resolved = dict(column_map or {})
    for name, col in infer_column_map(recipe, rows).items():
        resolved.setdefault(name, col)

    unresolved = [i.name for i in recipe.inputs if i.name not in resolved]
    if unresolved:
        available = ", ".join(_role_columns(rows)) or "(none)"
        raise BatchPlanError(
            f"could not map recipe input(s) {', '.join(unresolved)} to a manifest "
            f"column; available columns: {available}. Pass an explicit column_map."
        )
    # Inference only offers role columns, but an explicit column_map may target
    # any column actually present (e.g. output_dir), so validate against all of them.
    present: set[str] = set(rows[0]) if rows else set()
    bad = {name: col for name, col in resolved.items() if col not in present}
    if bad:
        pairs = ", ".join(f"{n}->{c}" for n, c in bad.items())
        raise BatchPlanError(f"mapped column(s) not in manifest: {pairs}")

    binding = BatchBinding(column_map=resolved)
    for row in rows:
        if row.get("status") != "ok":
            entry = {k: row.get(k, "") for k in ("subject", "session", "status", "reason")}
            binding.skipped.append({k: v for k, v in entry.items() if v})
            continue
        binding.runs.append({name: row[col] for name, col in resolved.items()})
        subject = row.get("subject", "?")
        session = row.get("session", "")
        binding.labels.append(f"{subject}/{session}" if session else subject)
    return binding
