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
