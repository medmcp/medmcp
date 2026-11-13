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

**Selection manifests.** The cohort stack's ``select_series`` writes a different
table: one row per selected *series* (``subject_id``, ``session_id``,
``sequence``, ``contrast``, ``plane``, ``path``, …; ``collection`` when several
collections are indexed together), with no ``status`` — every row was selected
on purpose. :func:`runs_from_selection` turns that into runs without a single
filename glob: a one-input recipe runs once per row on its ``path``; a recipe
with several inputs is pivoted by subject and session, each input bound to the
series of the *role* the person assigned it (``FLAIR``, ``T1+contrast``), and a
subject-session where a role is absent or present twice is flagged, not guessed.
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
        kind: ``"plan"`` (a ``plan_batch`` manifest) or ``"selection"``.
        role_map: For a selection, the ``in_N -> role`` assignment used.
    """

    runs: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    labels: list[str] = field(default_factory=list[str])
    skipped: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    column_map: dict[str, str] = field(default_factory=dict[str, str])
    output_inputs: list[str] = field(default_factory=list[str])
    output_root: str = ""
    kind: str = "plan"
    role_map: dict[str, str] = field(default_factory=dict[str, str])


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


# ── Selection manifests (select_series) ──────────────────────────────────────


class NeedsRolesError(BatchPlanError):
    """A selection cannot be bound until the person says which series feeds which input.

    Attributes:
        unmapped: The recipe inputs that still have no role.
        role_map: The assignments already known (given or inferred).
    """

    def __init__(self, unmapped: list[str], role_map: dict[str, str]) -> None:
        """Remember which inputs lack a role, for the caller to ask about."""
        super().__init__(
            f"say which series feeds the input(s) {', '.join(unmapped)} "
            "(for example in_1=FLAIR, in_2=T1+contrast)"
        )
        self.unmapped = unmapped
        self.role_map = role_map


@dataclass(frozen=True)
class Role:
    """Which series an input takes: a sequence, optionally narrowed.

    Written ``SEQUENCE``, ``SEQUENCE+contrast``, ``SEQUENCE-contrast`` or
    ``SEQUENCE?contrast`` (contrast unknown), with an optional ``@plane`` — the
    notation ``select_series`` uses for ``require_in_session``. ``contrast`` and
    ``plane`` empty mean "any".
    """

    sequence: str
    contrast: str = ""
    plane: str = ""

    def matches(self, row: dict[str, str]) -> bool:
        """Whether the series in *row* is one this role takes."""
        if row.get("sequence", "").strip().lower() != self.sequence.lower():
            return False
        if self.contrast and row.get("contrast", "").strip().lower() != self.contrast:
            return False
        return not self.plane or row.get("plane", "").strip().lower() == self.plane


_CONTRAST_MARKS = {"+": "yes", "-": "no", "?": "unknown"}
_ROLE_RE = re.compile(r"^(?P<seq>[^@+?-]+?)(?:(?P<mark>[+?-])contrast)?(?:@(?P<plane>\w+))?$", re.I)


def parse_role(text: str) -> Role:
    """Read a role as written (``"T1+contrast"``, ``"FLAIR"``, ``"T2@axial"``).

    Raises:
        BatchPlanError: The text is not a role.
    """
    match = _ROLE_RE.match(text.strip())
    if match is None or not match.group("seq").strip():
        raise BatchPlanError(
            f"not a series role: {text!r} (write SEQUENCE, SEQUENCE+contrast or SEQUENCE-contrast)"
        )
    return Role(
        sequence=match.group("seq").strip(),
        contrast=_CONTRAST_MARKS.get(match.group("mark") or "", ""),
        plane=(match.group("plane") or "").lower(),
    )


def role_of(row: dict[str, str]) -> str:
    """The role a selection row is, as written: its sequence and, when known, contrast."""
    sequence = row.get("sequence", "").strip() or "unknown"
    mark = {"yes": "+contrast", "no": "-contrast"}.get(row.get("contrast", "").strip().lower(), "")
    return f"{sequence}{mark}"


def is_selection(rows: list[dict[str, str]]) -> bool:
    """Whether *rows* are a ``select_series`` manifest rather than a ``plan_batch`` one."""
    return bool(rows) and "path" in rows[0] and "subject_id" in rows[0]


def _unit(row: dict[str, str]) -> tuple[str, str, str]:
    """Who a series belongs to: ``(collection, subject, session)``.

    The same subject id in two collections is two people, so the collection is
    part of the identity whenever the manifest has one.
    """
    return (
        row.get("collection", "").strip(),
        row.get("subject_id", "").strip(),
        row.get("session_id", "").strip(),
    )


def _unit_label(unit: tuple[str, str, str]) -> str:
    return "/".join(part for part in unit if part) or "?"


def _unit_folder(root: PurePosixPath, unit: tuple[str, str, str]) -> PurePosixPath:
    folder = root
    for part in unit:
        if part:
            folder = folder / folder_part(part)
    return folder


def selection_roles(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """The roles a selection offers, for a person to assign to inputs.

    One entry per sequence-and-contrast combination present, plus the bare
    sequence where it comes with more than one contrast state — each with how
    many series and how many subject-sessions have it.
    """
    series: dict[str, int] = {}
    units: dict[str, set[tuple[str, str, str]]] = {}
    contrasts: dict[str, set[str]] = {}
    for row in rows:
        exact = role_of(row)
        sequence = row.get("sequence", "").strip() or "unknown"
        contrasts.setdefault(sequence, set()).add(exact)
        for name in {exact, sequence}:
            series[name] = series.get(name, 0) + 1
            units.setdefault(name, set()).add(_unit(row))
    out: list[dict[str, object]] = []
    for sequence in sorted(contrasts):
        exact = sorted(contrasts[sequence])
        names = exact if len(exact) == 1 else [sequence, *[n for n in exact if n != sequence]]
        for name in names:
            out.append({"role": name, "series": series[name], "units": len(units[name])})
    return out


def data_inputs(recipe: Recipe) -> list[str]:
    """The recipe inputs that need a file to read: not an output, and without a default."""
    written = output_inputs(recipe)
    return [i.name for i in recipe.inputs if i.name not in written and not i.default]


def _abs_path(value: str, base_dir: Path | None) -> str:
    """A manifest path as given, or against the dataset folder when it is relative."""
    if not value or base_dir is None or PurePosixPath(value).is_absolute():
        return value
    return str(PurePosixPath(str(base_dir)) / value)


def dataset_dir(manifest: str | Path) -> Path:
    """The dataset a selection manifest belongs to (it lives in ``<dataset>/.medmcp/``)."""
    folder = Path(manifest).resolve().parent
    return folder.parent if folder.name == ".medmcp" else folder


def _flag(unit: tuple[str, str, str], status: str, reason: str) -> dict[str, str]:
    entry = {
        "subject": "/".join(p for p in unit[:2] if p),
        "session": unit[2],
        "status": status,
        "reason": reason,
    }
    return {k: v for k, v in entry.items() if v}


def runs_from_selection(
    recipe: Recipe,
    rows: list[dict[str, str]],
    *,
    role_map: dict[str, str] | None = None,
    output_root: str | Path | None = None,
    base_dir: Path | None = None,
) -> BatchBinding:
    """Build replay bindings from a ``select_series`` manifest.

    Args:
        recipe: The distilled workflow.
        rows: The manifest rows (one per selected series).
        role_map: ``in_N -> role`` (see :class:`Role`). Optional for a one-input
            recipe, where it narrows which rows run; needed for every reading
            input of a multi-input recipe unless the input's recorded example is
            itself a row of the manifest.
        output_root: Each item's outputs go to
            ``<output_root>/[<collection>/]<subject>[/<session>]``.
        base_dir: What a relative ``path`` is read against (the dataset folder).

    Returns:
        A :class:`BatchBinding` of kind ``"selection"``. Subject-sessions that
        lack a role, or have it twice, are in ``skipped`` with the reason.

    Raises:
        NeedsRolesError: A multi-input recipe has inputs without a role.
        BatchPlanError: A role is malformed, the recipe reads nothing, or an
            output has nowhere to go.
    """
    readers = data_inputs(recipe)
    if not readers:
        raise BatchPlanError("this workflow has no input to bind a selected series to")
    root = PurePosixPath(str(output_root)) if output_root else None
    written = output_inputs(recipe)
    defaults = {i.name for i in recipe.inputs if i.default}
    placed: dict[str, workflow.OutputKind] = written if root is not None else {}
    homeless = [n for n in written if n not in placed and n not in defaults]
    if homeless:
        raise BatchPlanError(
            f"no place for the output(s) {', '.join(homeless)}: give an output root"
        )

    usable: list[dict[str, str]] = []
    binding = BatchBinding(
        kind="selection",
        output_inputs=sorted(placed),
        output_root=str(root) if root is not None else "",
    )
    for row in rows:
        status = row.get("status", "").strip()
        if status and status != "ok":
            binding.skipped.append(_flag(_unit(row), status, row.get("reason", "")))
        elif not row.get("path", "").strip():
            binding.skipped.append(_flag(_unit(row), "missing", "the row has no path"))
        else:
            usable.append(row)

    # An input whose recorded example is a selected series has that series' role.
    roles = dict(role_map or {})
    by_path = {_abs_path(r["path"], base_dir): r for r in usable}
    for inp in recipe.inputs:
        if inp.name in readers and inp.name not in roles and inp.example in by_path:
            roles[inp.name] = role_of(by_path[inp.example])
    parsed = {name: parse_role(text) for name, text in roles.items() if name in readers}

    def _outputs(folder: PurePosixPath) -> dict[str, str]:
        return {name: place_output(recipe, name, kind, folder) for name, kind in placed.items()}

    if len(readers) == 1:
        # One run per selected series. A role, when given, only narrows the rows.
        name = readers[0]
        chosen = [r for r in usable if name not in parsed or parsed[name].matches(r)]
        per_unit: dict[tuple[str, str, str], int] = {}
        for row in chosen:
            per_unit[_unit(row)] = per_unit.get(_unit(row), 0) + 1
        taken: set[str] = set()
        for row in chosen:
            unit = _unit(row)
            label = _unit_label(unit)
            folder = _unit_folder(root, unit) if root is not None else None
            if per_unit[unit] > 1:
                # Several series of one session: each gets its own folder, or two
                # would write the same file.
                series = folder_part(role_of(row))
                tag, n = series, 1
                while f"{label}/{tag}" in taken:
                    n += 1
                    tag = f"{series}_{n}"
                taken.add(f"{label}/{tag}")
                label = f"{label} · {tag}"
                folder = folder / tag if folder is not None else None
            run = {name: _abs_path(row["path"], base_dir)}
            if folder is not None:
                run.update(_outputs(folder))
            binding.runs.append(run)
            binding.labels.append(label)
        binding.role_map = {k: v for k, v in roles.items() if k in readers}
        return binding

    unmapped = [n for n in readers if n not in parsed]
    if unmapped:
        raise NeedsRolesError(unmapped, {k: v for k, v in roles.items() if k in readers})
    binding.role_map = {n: roles[n] for n in readers}
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in usable:
        grouped.setdefault(_unit(row), []).append(row)
    for unit, members in grouped.items():
        run: dict[str, str] = {}
        problem: tuple[str, str] | None = None
        for name in readers:
            hits = [r for r in members if parsed[name].matches(r)]
            if not hits:
                problem = ("missing", f"no {roles[name]} series for {name}")
                break
            if len(hits) > 1:
                what = ", ".join(
                    r.get("series_description", "").strip() or PurePosixPath(r["path"]).name
                    for r in hits[:3]
                )
                problem = (
                    "ambiguous",
                    f"{len(hits)} {roles[name]} series for {name} ({what}); "
                    "narrow the selection or the role",
                )
                break
            run[name] = _abs_path(hits[0]["path"], base_dir)
        if problem is not None:
            binding.skipped.append(_flag(unit, *problem))
            continue
        if root is not None:
            run.update(_outputs(_unit_folder(root, unit)))
        binding.runs.append(run)
        binding.labels.append(_unit_label(unit))
    return binding


def bind_manifest(
    recipe: Recipe,
    manifest: str | Path,
    *,
    column_map: dict[str, str] | None = None,
    role_map: dict[str, str] | None = None,
    output_root: str | Path | None = None,
) -> BatchBinding:
    """Read *manifest* and bind it, whichever of the two kinds it is."""
    rows = read_manifest(manifest)
    if is_selection(rows):
        return runs_from_selection(
            recipe,
            rows,
            role_map=role_map,
            output_root=output_root,
            base_dir=dataset_dir(manifest),
        )
    return runs_from_manifest(recipe, rows, column_map=column_map, output_root=output_root)
