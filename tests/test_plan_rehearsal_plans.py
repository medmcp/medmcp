"""The plans of the plan2 rehearsal (U1-U9, D7), checked as the core now checks them.

Two things the rehearsal showed the plan check missing, besides the plan tool
answering "approved" without a card (see ``test_plan_guards``):

* required inputs left empty — ``{}``, ``""``, ``[]`` or a placeholder like
  ``"<selection_path from previous step>"`` (U1, U4, U5, D7);
* parameters the tools do not take, whatever the spelling of the tool (U7, U9).

The schemas are the cohort and dicom stacks' real ones, cut down to their
parameter names and ``required`` lists (select_series keeps its ``$defs``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medmcp import plan

JsonDict = dict[str, Any]


def _schema(props: list[str], required: list[str], **extra: object) -> JsonDict:
    return {
        "type": "object",
        "properties": {p: {} for p in props},
        "required": required,
        **extra,
    }


SCHEMAS: dict[str, dict[str, JsonDict]] = {
    "medmcp-cohort": {
        "index_dataset": _schema(["dataset_dir", "output_dir"], ["dataset_dir"]),
        "summarize_dataset": _schema(
            ["index_path", "require", "collection", "output_dir", "group_by"], ["index_path"]
        ),
        "select_series": {
            "$defs": {"SeriesCriteria": {"type": "object", "properties": {"sequence": {}}}},
            "type": "object",
            "properties": {
                "index_path": {"type": "string"},
                "criteria": {"$ref": "#/$defs/SeriesCriteria"},
                "output_dir": {},
                "name": {},
            },
            "required": ["index_path", "criteria"],
        },
        "review_selection": _schema(
            ["manifest_path", "roles", "prefer", "output_dir"], ["manifest_path", "roles"]
        ),
        "confirm_selection": _schema(
            ["review_path", "user_reply", "overrides", "note"], ["review_path", "user_reply"]
        ),
        "check_dataset": _schema(
            ["index_path", "collection", "output_dir", "deep_check"], ["index_path"]
        ),
        "plan_batch": _schema(
            ["cohort_path", "scans_dir", "inputs", "output_dir"],
            ["cohort_path", "scans_dir", "inputs", "output_dir"],
        ),
        "flow_diagram": _schema(["sources", "output_dir", "title"], ["sources", "output_dir"]),
    },
    "medmcp-dicom": {
        "build_bids_dataset": _schema(
            ["dicom_root", "output_dir", "dataset_name", "anonymize"], ["dicom_root", "output_dir"]
        ),
        "convert_dcm_to_nifti": _schema(["input_dir", "output_dir"], ["input_dir", "output_dir"]),
    },
}


def _problems(steps: list[JsonDict], ws: Path) -> list[str]:
    """Every problem the core finds: parse_plan's, then check_tools'."""
    args = {"understanding": "x", "steps": steps, "output_dir": str(ws / "results")}
    try:
        parsed = plan.parse_plan(args)
    except plan.PlanError as exc:
        return exc.problems
    return plan.check_tools(parsed, SCHEMAS, complete=True)


class TestEmptyRequiredInputs:
    """A required input must be shown with a value."""

    def test_u1_steps_with_no_inputs_are_refused(self, tmp_path: Path) -> None:
        """U1: summarize_dataset and select_series planned with ``{}``."""
        (problem,) = _problems(
            [{"action": "Summarize", "tool": "summarize_dataset", "inputs": {}}], tmp_path
        )
        assert "summarize_dataset requires index_path" in problem

    def test_u4_blank_required_values_are_refused(self, tmp_path: Path) -> None:
        """U4: review_selection with ``manifest_path: ""`` and ``roles: []``."""
        (problem,) = _problems(
            [
                {
                    "action": "Review",
                    "tool": "review_selection",
                    "inputs": {"manifest_path": "", "roles": []},
                }
            ],
            tmp_path,
        )
        assert "'manifest_path', 'roles' is empty" in problem

    def test_u5_a_placeholder_is_refused(self, tmp_path: Path) -> None:
        """U5: ``"<selection_path from previous step>"`` stands for no value."""
        (problem,) = _problems(
            [
                {
                    "action": "Review",
                    "tool": "medmcp-cohort_review_selection",
                    "inputs": {
                        "manifest_path": "<selection_path from previous step>",
                        "roles": ["angio"],
                    },
                }
            ],
            tmp_path,
        )
        assert "'manifest_path' is empty" in problem

    def test_d7_a_list_of_blanks_is_refused(self, tmp_path: Path) -> None:
        """D7: flow_diagram ``sources: [""]``."""
        (problem,) = _problems(
            [
                {
                    "action": "Flow",
                    "tool": "mcp_medmcp_cohort.flow_diagram",
                    "inputs": {"sources": [""], "output_dir": str(tmp_path / "results")},
                }
            ],
            tmp_path,
        )
        assert "'sources' is empty" in problem

    def test_partial_inputs_and_tools_without_required_pass(self, tmp_path: Path) -> None:
        """Key arguments are enough; an optional blank, or a tool with nothing required, passes."""
        steps: list[JsonDict] = [
            {
                "action": "Summarize",
                "tool": "summarize_dataset",
                "inputs": {"index_path": str(tmp_path), "collection": ""},
            },
            {"action": "Check", "tool": "check_dataset", "inputs": {"index_path": str(tmp_path)}},
        ]
        assert _problems(steps, tmp_path) == []
        assert (
            plan.check_tools(
                plan.parse_plan({"understanding": "x", "steps": [{"action": "a", "tool": "t"}]}),
                {"s": {"t": {"type": "object", "properties": {}}}},
                complete=True,
            )
            == []
        )


class TestUnknownParameters:
    """Every spelling of a tool is checked against its schema, $defs or not."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "select_series",
            "medmcp-cohort_select_series",
            "mcp_medmcp_cohort.select_series",
            "tools.mcp_medmcp_cohort.select_series",
        ],
    )
    def test_u9_invented_parameters_are_refused(self, tmp_path: Path, spelling: str) -> None:
        """U9: ``dataset_dir`` and ``sequence_name`` on select_series."""
        (problem,) = _problems(
            [
                {
                    "action": "Select DWI",
                    "tool": spelling,
                    "inputs": {"dataset_dir": str(tmp_path), "sequence_name": "DWI"},
                }
            ],
            tmp_path,
        )
        assert "no parameter 'dataset_dir', 'sequence_name'" in problem
        assert "index_path, criteria, output_dir, name" in problem

    def test_u7_dataset_dir_on_select_series(self, tmp_path: Path) -> None:
        """U7: select_series with ``dataset_dir`` and criteria."""
        problems = _problems(
            [
                {"action": "Index", "tool": "index_dataset", "inputs": {"dataset_dir": "/d"}},
                {
                    "action": "Select T2",
                    "tool": "select_series",
                    "inputs": {"criteria": {"sequence": ["T2"]}, "dataset_dir": "/d"},
                },
            ],
            tmp_path,
        )
        assert any("select_series has no parameter 'dataset_dir'" in p for p in problems)
