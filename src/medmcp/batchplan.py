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

**Outputs are never read from an input column.** A recipe input a tool *writes*
(``output_path``, ``output_dir``) is not matched against the manifest's role
columns at all — those hold the files to read, and a name coincidence there once
bound every subject's output to its own input file. It is placed instead under
an output root, one folder per item: ``<root>/<subject>[/<session>]`` for a
folder, and that folder plus the recorded file name for a file.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from medmcp import workflow
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
        output_inputs: The recipe inputs that were placed under the output root
            rather than read from a column.
        output_root: The root those were placed under ("" when the manifest's
            own ``output_dir`` column, or the recipe's default, decided).
    """

    runs: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    labels: list[str] = field(default_factory=list[str])
    skipped: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    column_map: dict[str, str] = field(default_factory=dict[str, str])
    output_inputs: list[str] = field(default_factory=list[str])
    output_root: str = ""


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a ``batch_plan.csv`` manifest into a list of row dicts."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(handle)]


def role_columns(rows: list[dict[str, str]]) -> list[str]:
    """The manifest's resolvable input-role columns (order preserved, no metadata)."""
    if not rows:
        return []
    return [c for c in rows[0] if c not in _RESERVED_COLUMNS]


_UNSAFE_PART_RE = re.compile(r"[^A-Za-z0-9._-]+")


def folder_part(text: str) -> str:
    """One path component made from an id: nothing that could leave its folder."""
    part = _UNSAFE_PART_RE.sub("_", text.strip()).strip("._")
    return part or "item"


def output_inputs(recipe: Recipe) -> dict[str, workflow.OutputKind]:
    """The recipe inputs a tool writes to, with whether each is a folder or a file."""
    names = workflow.input_argument_names(recipe)
    return {
        name: workflow.output_kind(names[name])
        for name, role in workflow.input_roles(recipe).items()
        if role == "output"
    }


def place_output(
    recipe: Recipe, name: str, kind: workflow.OutputKind, folder: PurePosixPath
) -> str:
    """Where output input *name* goes for an item whose results folder is *folder*.

    A folder output is the folder; a file output is the file name the session
    wrote, inside it — the name is the one thing about the recorded path that
    carries over to another subject.
    """
    if kind == "dir":
        return str(folder)
    example = next((i.example for i in recipe.inputs if i.name == name), "")
    return str(folder / (PurePosixPath(example).name or name))


def infer_column_map(recipe: Recipe, rows: list[dict[str, str]]) -> dict[str, str]:
    """Match each recipe input (``in_N``) a tool *reads* to a manifest role column.

    Two strategies, in order, applied per input:

    1. **Exact example** — the column whose value in some row equals the input's
       recorded ``example`` path (unambiguous when the prototype subject is in the
       cohort).
    2. **Role token** — the column whose name appears in the example's file name
       (e.g. example ``…_FLAIR.nii.gz`` → column ``flair``), accepted only when a
       single column matches.

    An input a tool writes to is never offered a column (see the module
    docstring). Returns the inferred ``in_N -> column`` map (possibly partial);
    inputs that stay unresolved are reported by :func:`runs_from_manifest`.
    """
    columns = role_columns(rows)
    written = output_inputs(recipe)
    mapping: dict[str, str] = {}
    for inp in recipe.inputs:
        if inp.name in written:
            continue
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
    output_root: str | Path | None = None,
) -> BatchBinding:
    """Build per-item replay bindings from a manifest, skipping flagged rows.

    Args:
        recipe: The distilled workflow whose ``{{in_N}}`` inputs are bound per row.
        rows: Manifest rows from :func:`read_manifest`.
        column_map: Optional explicit ``in_N -> manifest-column`` mapping; any
            input not covered is inferred via :func:`infer_column_map`.
        output_root: Where results go: each item's outputs are placed in
            ``<output_root>/<subject>[/<session>]``. Without it the manifest's
            ``output_dir`` column is used when it has one; failing both, an output
            that has a recipe default keeps it and one that has none is an error.

    Returns:
        A :class:`BatchBinding`.

    Raises:
        BatchPlanError: If a recipe input can't be matched to a manifest column
            (the caller should re-run with an explicit ``column_map``), a mapped
            column is absent from the manifest, or an output has nowhere to go.
    """
    resolved = dict(column_map or {})
    for name, col in infer_column_map(recipe, rows).items():
        resolved.setdefault(name, col)

    present: set[str] = set(rows[0]) if rows else set()
    root = PurePosixPath(str(output_root)) if output_root else None
    planned = "output_dir" in present
    # An output the caller mapped by hand stays mapped; the rest are placed.
    written: dict[str, workflow.OutputKind] = {
        n: k for n, k in output_inputs(recipe).items() if n not in resolved
    }
    defaults = {i.name for i in recipe.inputs if i.default}
    placed: dict[str, workflow.OutputKind] = written if root is not None or planned else {}
    homeless = [n for n in written if n not in placed and n not in defaults]
    if homeless:
        raise BatchPlanError(
            f"no place for the output(s) {', '.join(homeless)}: give an output root "
            "(the manifest has no output_dir column)"
        )

    unresolved = [
        i.name
        for i in recipe.inputs
        if i.name not in resolved and i.name not in written and i.name not in defaults
    ]
    if unresolved:
        available = ", ".join(role_columns(rows)) or "(none)"
        raise BatchPlanError(
            f"could not map recipe input(s) {', '.join(unresolved)} to a manifest "
            f"column; available columns: {available}. Pass an explicit column_map."
        )
    # Inference only offers role columns, but an explicit column_map may target
    # any column actually present (e.g. output_dir), so validate against all of them.
    bad = {name: col for name, col in resolved.items() if col not in present}
    if bad:
        pairs = ", ".join(f"{n}->{c}" for n, c in bad.items())
        raise BatchPlanError(f"mapped column(s) not in manifest: {pairs}")

    binding = BatchBinding(
        column_map=resolved,
        output_inputs=sorted(placed),
        output_root=str(root) if root is not None else "",
    )
    for row in rows:
        if row.get("status") != "ok":
            entry = {k: row.get(k, "") for k in ("subject", "session", "status", "reason")}
            binding.skipped.append({k: v for k, v in entry.items() if v})
            continue
        subject = row.get("subject", "?")
        session = row.get("session", "")
        run = {name: row[col] for name, col in resolved.items()}
        if placed:
            if root is not None:
                folder = root / folder_part(subject)
                if session:
                    folder = folder / folder_part(session)
            else:
                folder = PurePosixPath(row.get("output_dir", ""))
            if not str(folder).strip("."):
                entry = {
                    "subject": subject,
                    "session": session,
                    "status": "missing",
                    "reason": "no output_dir planned for this row",
                }
                binding.skipped.append({k: v for k, v in entry.items() if v})
                continue
            for name, kind in placed.items():
                run[name] = place_output(recipe, name, kind, folder)
        binding.runs.append(run)
        binding.labels.append(f"{subject}/{session}" if session else subject)
    return binding
