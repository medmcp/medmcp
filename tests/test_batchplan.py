"""Tests for batchplan: turning a cohort manifest into replay bindings."""

from __future__ import annotations

from pathlib import Path

import pytest

from medmcp import batchplan
from medmcp.workflow import Recipe, RecipeStep, WorkflowInput


def _recipe() -> Recipe:
    """A two-input, one-step recipe (segment FLAIR guided by a mask)."""
    return Recipe(
        name="lesion-volume",
        description="segment then measure",
        inputs=[
            WorkflowInput(name="in_1", example="/ws/scans/P6/T1/P6_T1_FLAIR.nii.gz"),
            WorkflowInput(name="in_2", example="/ws/scans/P6/T1/P6_T1_MASK.nii.gz"),
        ],
        steps=[
            RecipeStep(
                server="medmcp-cohort",
                tool="extract_lesion_volume",
                arguments={"flair": "{{in_1}}", "mask": "{{in_2}}"},
            )
        ],
    )


def _rows(subject_paths: list[tuple[str, str, str, str, str]]) -> list[dict[str, str]]:
    """Build manifest rows: (subject, session, flair, mask, status)."""
    return [
        {"subject": s, "session": t, "flair": f, "mask": m, "status": st, "reason": ""}
        for s, t, f, m, st in subject_paths
    ]


def test_infer_by_exact_example_and_build_runs() -> None:
    """The prototype subject's exact paths anchor the in_N -> column mapping."""
    f6, m6 = "/ws/P6/T1/P6_T1_FLAIR.nii.gz", "/ws/P6/T1/P6_T1_MASK.nii.gz"
    f9, m9 = "/ws/P9/T1/P9_T1_FLAIR.nii.gz", "/ws/P9/T1/P9_T1_MASK.nii.gz"
    rows = _rows([("P6", "T1", f6, m6, "ok"), ("P9", "T1", f9, m9, "ok")])
    binding = batchplan.runs_from_manifest(_recipe(), rows)
    assert binding.column_map == {"in_1": "flair", "in_2": "mask"}
    assert binding.runs == [
        {"in_1": f6, "in_2": m6},
        {"in_1": f9, "in_2": m9},
    ]
    assert binding.labels == ["P6/T1", "P9/T1"]
    assert binding.skipped == []


def test_infer_by_role_token_for_a_different_cohort() -> None:
    """When the prototype subject isn't in the cohort, role-name tokens still map."""
    rows = _rows(
        [("P40", "T1", "/d/P40/T1/P40_T1_FLAIR.nii.gz", "/d/P40/T1/P40_T1_MASK.nii.gz", "ok")]
    )
    binding = batchplan.runs_from_manifest(_recipe(), rows)
    assert binding.column_map == {"in_1": "flair", "in_2": "mask"}
    assert binding.runs[0]["in_1"].endswith("P40_T1_FLAIR.nii.gz")


def test_flagged_rows_are_skipped_not_run() -> None:
    """Missing/ambiguous items are returned as skipped, never turned into runs."""
    rows = _rows(
        [
            ("P6", "T1", "/d/P6/T1/P6_T1_FLAIR.nii.gz", "/d/P6/T1/P6_T1_MASK.nii.gz", "ok"),
            ("P9", "T1", "", "", "missing"),
        ]
    )
    rows[1]["reason"] = "no file for role(s): flair, mask"
    binding = batchplan.runs_from_manifest(_recipe(), rows)
    assert len(binding.runs) == 1
    assert binding.skipped == [
        {
            "subject": "P9",
            "session": "T1",
            "status": "missing",
            "reason": "no file for role(s): flair, mask",
        }
    ]


def test_explicit_column_map_overrides_inference() -> None:
    """An explicit map wins even when it disagrees with what would be inferred."""
    rows = [
        {
            "subject": "P6",
            "session": "",
            "img": "/d/P6/flair.nii.gz",
            "seg": "/d/P6/mask.nii.gz",
            "status": "ok",
            "reason": "",
        }
    ]
    binding = batchplan.runs_from_manifest(
        _recipe(), rows, column_map={"in_1": "img", "in_2": "seg"}
    )
    assert binding.runs[0] == {"in_1": "/d/P6/flair.nii.gz", "in_2": "/d/P6/mask.nii.gz"}


def test_unresolvable_input_raises() -> None:
    """A recipe input that matches no column is an error, not a silent drop."""
    rows = [
        {
            "subject": "P6",
            "session": "",
            "img": "/d/a.nii",
            "seg": "/d/b.nii",
            "status": "ok",
            "reason": "",
        }
    ]
    with pytest.raises(batchplan.BatchPlanError, match="could not map recipe input"):
        batchplan.runs_from_manifest(_recipe(), rows)


def test_bad_explicit_column_raises() -> None:
    """A mapped column absent from the manifest is reported clearly."""
    rows = [
        {
            "subject": "P6",
            "session": "",
            "flair": "/d/a.nii",
            "mask": "/d/b.nii",
            "status": "ok",
            "reason": "",
        }
    ]
    with pytest.raises(batchplan.BatchPlanError, match="not in manifest"):
        batchplan.runs_from_manifest(_recipe(), rows, column_map={"in_1": "nope", "in_2": "mask"})


def test_read_manifest_roundtrip(tmp_path: Path) -> None:
    """read_manifest parses a CSV into row dicts with empty-string fills."""
    csv = tmp_path / "batch_plan.csv"
    csv.write_text(
        "subject,session,flair,mask,status,reason\n"
        "P6,T1,/d/P6_FLAIR.nii.gz,/d/P6_MASK.nii.gz,ok,\n",
        encoding="utf-8",
    )
    rows = batchplan.read_manifest(csv)
    assert rows == [
        {
            "subject": "P6",
            "session": "T1",
            "flair": "/d/P6_FLAIR.nii.gz",
            "mask": "/d/P6_MASK.nii.gz",
            "status": "ok",
            "reason": "",
        }
    ]


# ── outputs are placed, never read from an input column ──────────────────────


def _file_output_recipe() -> Recipe:
    """Reorient a mask: ``in_1`` is read, ``in_2`` is the file that is written."""
    return Recipe(
        name="reorient",
        description="d",
        inputs=[
            WorkflowInput(name="in_1", example="/ws/raw/patient01/consensus_gt.nii.gz"),
            WorkflowInput(name="in_2", example="/ws/out/patient01/consensus_gt_RAS.nii.gz"),
        ],
        steps=[
            RecipeStep(
                server="medmcp-tptbox",
                tool="process_nifti",
                arguments={"input_path": "{{in_1}}", "output_path": "{{in_2}}"},
            )
        ],
    )


def _plan(with_output_dir: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for subject in ("1", "2"):
        row = {
            "subject": subject,
            "consensus_gt": f"/ws/raw/patient0{subject}/consensus_gt.nii.gz",
            "status": "ok",
            "reason": "",
        }
        if with_output_dir:
            row["output_dir"] = f"/ws/planned/{subject}"
        rows.append(row)
    return rows


def test_an_output_is_never_inferred_from_an_input_column() -> None:
    """``consensus_gt`` is in the output's file name; it must not be bound to it."""
    recipe = _file_output_recipe()
    assert batchplan.infer_column_map(recipe, _plan(False)) == {"in_1": "consensus_gt"}
    assert batchplan.output_inputs(recipe) == {"in_2": "file"}


def test_output_goes_under_the_output_root_per_subject() -> None:
    """Each item writes ``<root>/<subject>/<recorded file name>``."""
    binding = batchplan.runs_from_manifest(
        _file_output_recipe(), _plan(False), output_root="/ws/results/reorient"
    )
    assert binding.column_map == {"in_1": "consensus_gt"}
    assert binding.output_inputs == ["in_2"]
    assert [r["in_2"] for r in binding.runs] == [
        "/ws/results/reorient/1/consensus_gt_RAS.nii.gz",
        "/ws/results/reorient/2/consensus_gt_RAS.nii.gz",
    ]
    assert all(r["in_1"] != r["in_2"] for r in binding.runs)


def test_the_plans_output_dir_column_is_used_without_a_root() -> None:
    """``plan_batch`` planned a folder per subject; a file output lands inside it."""
    binding = batchplan.runs_from_manifest(_file_output_recipe(), _plan(True))
    assert binding.runs[0]["in_2"] == "/ws/planned/1/consensus_gt_RAS.nii.gz"
    assert binding.output_root == ""


def test_an_output_with_nowhere_to_go_is_an_error() -> None:
    """No root, no ``output_dir`` column, no recipe default: say so, do not guess."""
    with pytest.raises(batchplan.BatchPlanError, match="output root"):
        batchplan.runs_from_manifest(_file_output_recipe(), _plan(False))


def test_a_folder_output_gets_the_item_folder_and_a_session_level() -> None:
    """An ``output_dir`` input is the item's folder; a session adds a level."""
    recipe = Recipe(
        name="seg",
        description="d",
        inputs=[
            WorkflowInput(name="in_1", example="/ws/P6/T1/P6_FLAIR.nii.gz"),
            WorkflowInput(name="in_2", example="/ws/P6/T1", default="{{dir(in_1)}}"),
        ],
        steps=[
            RecipeStep(
                server="s", tool="t", arguments={"input_path": "{{in_1}}", "output_dir": "{{in_2}}"}
            )
        ],
    )
    rows = [
        {"subject": "P 9", "session": "T1", "flair": "/d/P9/T1/P9_FLAIR.nii.gz", "status": "ok"}
    ]
    placed = batchplan.runs_from_manifest(recipe, rows, output_root="/ws/out")
    assert placed.runs == [{"in_1": "/d/P9/T1/P9_FLAIR.nii.gz", "in_2": "/ws/out/P_9/T1"}]
    # Without a root the recipe's own default (next to the scan) still applies.
    kept = batchplan.runs_from_manifest(recipe, rows)
    assert kept.runs == [{"in_1": "/d/P9/T1/P9_FLAIR.nii.gz"}]


def test_folder_part_cannot_leave_its_folder() -> None:
    """An id never becomes a path that climbs out of the output root."""
    assert batchplan.folder_part("../../etc") == "etc"
    assert batchplan.folder_part("sub 01/ses") == "sub_01_ses"
    assert batchplan.folder_part("..") == "item"
