"""Tests for batchplan: turning a cohort manifest into replay bindings."""

from __future__ import annotations

import hashlib
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


# ── selection manifests (select_series) ──────────────────────────────────────


def _series(
    subject: str, session: str, sequence: str, contrast: str, path: str, **extra: str
) -> dict[str, str]:
    """One row of a ``select_series`` manifest (the columns the roll-out reads)."""
    return {
        "subject_id": subject,
        "session_id": session,
        "sequence": sequence,
        "contrast": contrast,
        "plane": "axial",
        "path": path,
        **extra,
    }


def _one_input_recipe() -> Recipe:
    return Recipe(
        name="describe",
        description="d",
        inputs=[WorkflowInput(name="in_1", example="/elsewhere/scan.nii.gz")],
        steps=[RecipeStep(server="s", tool="describe", arguments={"image_path": "{{in_1}}"})],
    )


def _two_input_recipe() -> Recipe:
    """FLAIR resampled onto the post-contrast T1, written to a file."""
    return Recipe(
        name="onto-t1",
        description="d",
        inputs=[
            WorkflowInput(name="in_1", example="/elsewhere/FLAIR.nii.gz"),
            WorkflowInput(name="in_2", example="/elsewhere/T1WKS.nii.gz"),
            WorkflowInput(name="in_3", example="/elsewhere/out/FLAIR_on_T1.nii.gz"),
        ],
        steps=[
            RecipeStep(
                server="s",
                tool="process_nifti",
                arguments={
                    "input_path": "{{in_1}}",
                    "operations": [{"op": "resample_to", "reference_path": "{{in_2}}"}],
                    "output_path": "{{in_3}}",
                },
            )
        ],
    )


def test_a_selection_is_recognised_and_a_plan_is_not() -> None:
    """The two manifests are told apart by their columns."""
    assert batchplan.is_selection([_series("P1", "", "FLAIR", "no", "/d/P1/FLAIR.nii.gz")])
    assert not batchplan.is_selection(_plan(False))
    assert not batchplan.is_selection([])


def test_one_input_runs_once_per_selected_series_without_a_status_column() -> None:
    """Every row is a run on its ``path``; nothing has to be mapped."""
    rows = [
        _series("P1", "ses-1", "FLAIR", "no", "/d/P1/FLAIR.nii.gz"),
        _series("P2", "ses-1", "FLAIR", "no", "/d/P2/FLAIR.nii.gz"),
    ]
    binding = batchplan.runs_from_selection(_one_input_recipe(), rows)
    assert binding.kind == "selection"
    assert binding.runs == [{"in_1": "/d/P1/FLAIR.nii.gz"}, {"in_1": "/d/P2/FLAIR.nii.gz"}]
    assert binding.labels == ["P1/ses-1", "P2/ses-1"]
    assert binding.skipped == []


def test_several_series_of_one_session_get_their_own_folder() -> None:
    """Two scans of one visit must not write the same output file."""
    recipe = _file_output_recipe()
    rows = [
        _series("P1", "s1", "FLAIR", "no", "/d/P1/FLAIR.nii.gz"),
        _series("P1", "s1", "T2", "no", "/d/P1/T2W.nii.gz"),
        _series("P1", "s1", "T2", "no", "/d/P1/T2W_b.nii.gz"),
        _series("P2", "s1", "FLAIR", "no", "/d/P2/FLAIR.nii.gz"),
    ]
    binding = batchplan.runs_from_selection(recipe, rows, output_root="/ws/out")
    assert binding.labels == [
        "P1/s1 · FLAIR-contrast",
        "P1/s1 · T2-contrast",
        "P1/s1 · T2-contrast_2",
        "P2/s1",
    ]
    outputs = [r["in_2"] for r in binding.runs]
    assert outputs == [
        "/ws/out/P1/s1/FLAIR-contrast/consensus_gt_RAS.nii.gz",
        "/ws/out/P1/s1/T2-contrast/consensus_gt_RAS.nii.gz",
        "/ws/out/P1/s1/T2-contrast_2/consensus_gt_RAS.nii.gz",
        "/ws/out/P2/s1/consensus_gt_RAS.nii.gz",
    ]
    assert len(set(outputs)) == len(outputs)


def test_a_role_narrows_a_one_input_selection() -> None:
    """``in_1=FLAIR`` runs the FLAIR rows of a mixed selection only."""
    rows = [
        _series("P1", "", "FLAIR", "no", "/d/P1/FLAIR.nii.gz"),
        _series("P1", "", "T2", "no", "/d/P1/T2W.nii.gz"),
    ]
    binding = batchplan.runs_from_selection(_one_input_recipe(), rows, role_map={"in_1": "FLAIR"})
    assert binding.runs == [{"in_1": "/d/P1/FLAIR.nii.gz"}]
    assert binding.labels == ["P1"]


def test_a_plain_selection_is_refused_for_several_inputs() -> None:
    """Several inputs need a confirmed selection: no roles are asked for, none guessed."""
    rows = [
        _series("P1", "", "FLAIR", "no", "/d/P1/FLAIR.nii.gz"),
        _series("P1", "", "T1", "yes", "/d/P1/T1WKS.nii.gz"),
    ]
    for role_map in (None, {"in_1": "FLAIR", "in_2": "T1+contrast"}):
        with pytest.raises(batchplan.NeedsConfirmationError) as caught:
            batchplan.runs_from_selection(
                _two_input_recipe(), rows, role_map=role_map, output_root="/ws/out"
            )
        assert caught.value.inputs == ["in_1", "in_2"]
        assert "review_selection and confirm_selection first" in str(caught.value)


def _confirmed(subject: str, session: str, role: str, path: str, **extra: str) -> dict[str, str]:
    """One row of a ``confirm_selection`` manifest: a selection row plus its ``role``."""
    sequence, _, rest = role.partition("+")
    contrast = "yes" if rest else ("no" if role.endswith("-contrast") else "unknown")
    return {
        **_series(subject, session, sequence.removesuffix("-contrast"), contrast, path, **extra),
        "role": role,
        "decided_by": "rules",
    }


def test_a_confirmed_selection_asks_for_its_own_role_labels() -> None:
    """The inputs still to assign are named; the roles offered are the ``role`` column's."""
    rows = [
        _confirmed("P1", "", "FLAIR", "/d/P1/FLAIR.nii.gz"),
        _confirmed("P1", "", "T1+contrast", "/d/P1/T1WKS.nii.gz"),
    ]
    with pytest.raises(batchplan.NeedsRolesError) as caught:
        batchplan.runs_from_selection(_two_input_recipe(), rows, output_root="/ws/out")
    assert caught.value.unmapped == ["in_1", "in_2"]
    assert [r["role"] for r in batchplan.selection_roles(rows)] == ["FLAIR", "T1+contrast"]


def test_a_confirmed_selection_binds_by_role_label_only() -> None:
    """``T1`` is not a role of a selection confirmed as ``T1+contrast``: refused, not matched."""
    rows = [
        _confirmed("P1", "", "FLAIR", "/d/P1/FLAIR.nii.gz"),
        _confirmed("P1", "", "T1+contrast", "/d/P1/T1WKS.nii.gz"),
    ]
    with pytest.raises(batchplan.BatchPlanError, match="not a role of this confirmed selection"):
        batchplan.runs_from_selection(
            _two_input_recipe(), rows, role_map={"in_1": "FLAIR", "in_2": "T1"}, output_root="/o"
        )


def test_pivot_binds_each_role_and_flags_missing_and_ambiguous_sessions() -> None:
    """One run per subject-session; a session without, or with two of, a role is flagged."""
    rows = [
        _confirmed("P1", "s1", "FLAIR", "/d/P1/FLAIR.nii.gz"),
        _confirmed("P1", "s1", "T1-contrast", "/d/P1/T1W.nii.gz"),
        _confirmed("P1", "s1", "T1+contrast", "/d/P1/T1WKS.nii.gz"),
        _confirmed("P2", "s1", "FLAIR", "/d/P2/FLAIR.nii.gz"),
        _confirmed("P2", "s1", "T1-contrast", "/d/P2/T1W.nii.gz"),
        _confirmed("P3", "s1", "FLAIR", "/d/P3/FLAIR.nii.gz", series_description="ax FLAIR"),
        _confirmed("P3", "s1", "FLAIR", "/d/P3/FLAIR_rep.nii.gz", series_description="repeat"),
        _confirmed("P3", "s1", "T1+contrast", "/d/P3/T1WKS.nii.gz"),
    ]
    binding = batchplan.runs_from_selection(
        _two_input_recipe(),
        rows,
        role_map={"in_1": "FLAIR", "in_2": "T1+contrast"},
        output_root="/ws/out",
    )
    assert binding.runs == [
        {
            "in_1": "/d/P1/FLAIR.nii.gz",
            "in_2": "/d/P1/T1WKS.nii.gz",
            "in_3": "/ws/out/P1/s1/FLAIR_on_T1.nii.gz",
        }
    ]
    assert binding.labels == ["P1/s1"]
    assert binding.role_map == {"in_1": "FLAIR", "in_2": "T1+contrast"}
    by_subject = {s["subject"]: s for s in binding.skipped}
    assert by_subject["P2"]["status"] == "missing"
    assert "T1+contrast" in by_subject["P2"]["reason"]
    assert by_subject["P3"]["status"] == "ambiguous"
    assert "ax FLAIR, repeat" in by_subject["P3"]["reason"]


def test_roles_are_inferred_when_the_recorded_scans_are_in_the_selection() -> None:
    """The prototype subject's confirmed rows say which role each input had."""
    rows = [
        _confirmed("P0", "", "FLAIR", "/elsewhere/FLAIR.nii.gz"),
        _confirmed("P0", "", "T1+contrast", "/elsewhere/T1WKS.nii.gz"),
    ]
    binding = batchplan.runs_from_selection(_two_input_recipe(), rows, output_root="/o")
    assert binding.role_map == {"in_1": "FLAIR", "in_2": "T1+contrast"}
    assert len(binding.runs) == 1


def test_collection_is_part_of_who_a_subject_is() -> None:
    """The same subject id in two collections is two items with two folders."""
    rows = [
        _series("001", "", "FLAIR", "no", "/d/a/001/FLAIR.nii.gz", collection="siteA"),
        _series("001", "", "FLAIR", "no", "/d/b/001/FLAIR.nii.gz", collection="siteB"),
    ]
    binding = batchplan.runs_from_selection(_file_output_recipe(), rows, output_root="/o")
    assert binding.labels == ["siteA/001", "siteB/001"]
    assert [r["in_2"] for r in binding.runs] == [
        "/o/siteA/001/consensus_gt_RAS.nii.gz",
        "/o/siteB/001/consensus_gt_RAS.nii.gz",
    ]


def test_selection_roles_lists_what_can_be_assigned() -> None:
    """A sequence with two contrast states is offered bare and per state."""
    rows = [
        _series("P1", "", "FLAIR", "no", "/a"),
        _series("P1", "", "T1", "no", "/b"),
        _series("P1", "", "T1", "yes", "/c"),
        _series("P2", "", "T1", "unknown", "/d"),
    ]
    offered = {r["role"]: (r["series"], r["units"]) for r in batchplan.selection_roles(rows)}
    assert offered == {
        "FLAIR-contrast": (1, 1),
        "T1": (3, 2),
        "T1+contrast": (1, 1),
        "T1-contrast": (1, 1),
    }


def test_parse_role() -> None:
    """The notation of select_series' require_in_session, plus a plane."""
    assert batchplan.parse_role("T1+contrast") == batchplan.Role("T1", "yes")
    assert batchplan.parse_role("T1-contrast") == batchplan.Role("T1", "no")
    assert batchplan.parse_role("T1?contrast") == batchplan.Role("T1", "unknown")
    assert batchplan.parse_role(" flair@Coronal ") == batchplan.Role("flair", "", "coronal")
    assert batchplan.Role("flair").matches({"sequence": "FLAIR", "contrast": "no"})
    with pytest.raises(batchplan.BatchPlanError):
        batchplan.parse_role("+contrast")


def test_bind_manifest_reads_either_kind_and_resolves_relative_paths(tmp_path: Path) -> None:
    """A manifest in ``<dataset>/.medmcp/`` has its relative paths read against the dataset."""
    dataset = tmp_path / "study"
    (dataset / ".medmcp").mkdir(parents=True)
    manifest = dataset / ".medmcp" / "selection_series.csv"
    manifest.write_text(
        "subject_id,session_id,sequence,contrast,plane,path\nP1,,FLAIR,no,axial,P1/FLAIR.nii.gz\n"
    )
    binding = batchplan.bind_manifest(_one_input_recipe(), manifest)
    assert binding.runs == [{"in_1": f"{dataset}/P1/FLAIR.nii.gz"}]

    plan = tmp_path / "batch_plan.csv"
    plan.write_text("subject,scan,status,reason\n1,/d/1/scan.nii.gz,ok,\n")
    assert batchplan.bind_manifest(_one_input_recipe(), plan).kind == "plan"


# ── confirmed selections (confirm_selection) ─────────────────────────────────
#
# tests/fixtures/confirmed/ holds what the cohort stack wrote (medmcp-cohort
# 759e3db: review_selection then confirm_selection with one override) for three
# subjects and the roles T1, T1+contrast, FLAIR: the source selection
# lesions_series.csv, the confirmed lesions_confirmed.csv and its record
# lesions_confirmed.json (paths in the record rewritten to /workspace/study).
# P01 is decided by the rules, P02 by an override, P03 lacks a FLAIR and is
# left out. The CSVs are byte-for-byte the tool's output.

FIXTURES = Path(__file__).parent / "fixtures" / "confirmed"


def _study(tmp_path: Path) -> Path:
    """The fixture in ``<study>/.medmcp/``, where the cohort tools write it."""
    folder = tmp_path / "study" / ".medmcp"
    folder.mkdir(parents=True)
    for name in ("lesions_series.csv", "lesions_confirmed.csv", "lesions_confirmed.json"):
        (folder / name).write_bytes((FIXTURES / name).read_bytes())
    return folder


def _three_input_recipe() -> Recipe:
    """A lesion pipeline reading a T1, a T1 with contrast and a FLAIR, writing a mask."""
    return Recipe(
        name="lesions",
        description="d",
        inputs=[
            WorkflowInput(name="in_1", example="/elsewhere/T1.nii.gz"),
            WorkflowInput(name="in_2", example="/elsewhere/T1KM.nii.gz"),
            WorkflowInput(name="in_3", example="/elsewhere/FLAIR.nii.gz"),
            WorkflowInput(name="in_4", example="/elsewhere/out/lesions.nii.gz"),
        ],
        steps=[
            RecipeStep(
                server="s",
                tool="segment",
                arguments={
                    "t1_path": "{{in_1}}",
                    "t1ce_path": "{{in_2}}",
                    "flair_path": "{{in_3}}",
                    "output_path": "{{in_4}}",
                },
            )
        ],
    )


ROLES = {"in_1": "T1-contrast", "in_2": "T1+contrast", "in_3": "FLAIR"}


def test_a_confirmed_selection_rolls_out_with_its_record(tmp_path: Path) -> None:
    """The cohort tools' confirmed CSV binds by role, and its checked record rides along."""
    folder = _study(tmp_path)
    manifest = folder / "lesions_confirmed.csv"
    rows, confirmation = batchplan.load_manifest(manifest)
    assert [r["role"] for r in batchplan.selection_roles(rows)] == [
        "T1-contrast",
        "T1+contrast",
        "FLAIR",
    ]
    assert confirmation is not None

    binding = batchplan.bind_manifest(
        _three_input_recipe(), manifest, role_map=ROLES, output_root="/ws/out"
    )
    study = folder.parent
    assert binding.labels == ["P01/ses-01", "P02/ses-01"]  # P03 was left out at confirmation
    assert binding.skipped == []
    assert len(binding.runs) == 2
    assert binding.runs[1] == {
        "in_1": f"{study}/P02/ses-01/s06.nii.gz",  # the series the override picked
        "in_2": f"{study}/P02/ses-01/s07.nii.gz",
        "in_3": f"{study}/P02/ses-01/s08.nii.gz",
        "in_4": "/ws/out/P02/ses-01/lesions.nii.gz",
    }
    assert binding.confirmation == confirmation

    summary = confirmation.summary()
    assert summary["path"] == str(folder / "lesions_confirmed.json")
    assert summary["confirmed_csv_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert summary["sha256"] == hashlib.sha256(confirmation.path.read_bytes()).hexdigest()
    assert summary["rule_version"] == "2"
    assert summary["overrides"] == 1 and summary["excluded"] == 1
    assert summary["roles"] == ["T1-contrast", "T1+contrast", "FLAIR"]
    assert summary["confirmed_at"]
    assert (
        summary["source_manifest"]["sha256"]
        == hashlib.sha256((folder / "lesions_series.csv").read_bytes()).hexdigest()
    )
    assert summary["index"]["path"] == "/workspace/study/.medmcp/dataset_index.csv"
    assert summary["index"]["sha256"]


def test_a_confirmed_selection_changed_after_confirmation_is_refused(tmp_path: Path) -> None:
    """One edited path is enough: the hash in the record no longer matches."""
    folder = _study(tmp_path)
    manifest = folder / "lesions_confirmed.csv"
    manifest.write_text(manifest.read_text().replace("s06.nii.gz", "s05.nii.gz"))
    for read in (
        lambda: batchplan.load_manifest(manifest),
        lambda: batchplan.bind_manifest(_three_input_recipe(), manifest, role_map=ROLES),
        lambda: batchplan.check_run_manifest(manifest, _three_input_recipe()),
    ):
        with pytest.raises(batchplan.ConfirmationError) as caught:
            read()
        assert "the confirmed selection was changed after confirmation" in str(caught.value)


def test_a_confirmed_selection_without_its_record_is_refused(tmp_path: Path) -> None:
    """No record, a record of something else, or a role-less ``_confirmed.csv``: refused."""
    folder = _study(tmp_path)
    manifest = folder / "lesions_confirmed.csv"
    record = folder / "lesions_confirmed.json"
    text = record.read_text()

    record.write_text(text.replace("medmcp-cohort confirm_selection", "something else"))
    with pytest.raises(batchplan.ConfirmationError, match="not a confirm_selection record"):
        batchplan.load_manifest(manifest)

    record.unlink()
    with pytest.raises(batchplan.ConfirmationError, match="has no confirmation record"):
        batchplan.load_manifest(manifest)

    # A copy under another name is still bound only with a record beside it.
    renamed = folder / "mine.csv"
    renamed.write_bytes(manifest.read_bytes())
    with pytest.raises(batchplan.ConfirmationError, match="has no confirmation record"):
        batchplan.load_manifest(renamed)

    plain_named = folder / "other_confirmed.csv"
    plain_named.write_bytes((folder / "lesions_series.csv").read_bytes())
    with pytest.raises(batchplan.ConfirmationError, match="no role column"):
        batchplan.load_manifest(plain_named)


def test_the_unconfirmed_source_selection_is_refused_for_several_inputs(tmp_path: Path) -> None:
    """The same selection before confirmation: refused, with what to do instead."""
    source = _study(tmp_path) / "lesions_series.csv"
    with pytest.raises(batchplan.NeedsConfirmationError, match="confirm_selection first"):
        batchplan.bind_manifest(_three_input_recipe(), source, role_map=ROLES, output_root="/o")
    with pytest.raises(batchplan.NeedsConfirmationError):
        batchplan.check_run_manifest(source, _three_input_recipe())


def test_a_one_input_roll_out_takes_a_plain_selection_or_one_confirmed_role(
    tmp_path: Path,
) -> None:
    """One input is never ambiguous on a plain selection; a confirmed one names its role."""
    folder = _study(tmp_path)
    source = folder / "lesions_series.csv"
    plain = batchplan.bind_manifest(_one_input_recipe(), source)
    assert len(plain.runs) == 10 and plain.confirmation is None
    assert batchplan.check_run_manifest(source, _one_input_recipe()) is None

    confirmed = folder / "lesions_confirmed.csv"
    with pytest.raises(batchplan.NeedsRolesError) as caught:
        batchplan.bind_manifest(_one_input_recipe(), confirmed)
    assert caught.value.unmapped == ["in_1"]
    flair = batchplan.bind_manifest(_one_input_recipe(), confirmed, role_map={"in_1": "FLAIR"})
    assert [Path(r["in_1"]).name for r in flair.runs] == ["s04.nii.gz", "s08.nii.gz"]
    assert flair.confirmation is not None


def test_a_plan_is_not_held_to_the_selection_rules(tmp_path: Path) -> None:
    """A plan_batch manifest of several inputs binds as before, with no record."""
    plan = tmp_path / "batch_plan.csv"
    plan.write_text(
        "subject,flair,mask,status,reason\n"
        "1,/ws/scans/P1/P1_T1_FLAIR.nii.gz,/ws/scans/P1/P1_T1_MASK.nii.gz,ok,\n"
    )
    binding = batchplan.bind_manifest(_recipe(), plan)
    assert binding.kind == "plan" and len(binding.runs) == 1
    assert binding.confirmation is None
    assert batchplan.check_run_manifest(plan, _recipe()) is None
    assert batchplan.check_run_manifest(tmp_path / "gone.csv", _recipe()) is None
