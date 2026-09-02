"""Tests for the workflow schema round trip and the on-disk layout."""

from __future__ import annotations

from pathlib import Path

import yaml

from medmcp import workflow
from medmcp.workflow import Recipe, RecipeStep, StackRequirement, WorkflowInput


def _recipe(name: str = "strip") -> Recipe:
    return Recipe(
        name=name,
        description="Skull strip a brain MRI.",
        inputs=[
            WorkflowInput(name="in_1", example="/d/t1.nii.gz", description="the scan"),
            WorkflowInput(name="in_2", example="/d", default="{{dir(in_1)}}"),
        ],
        steps=[
            RecipeStep(
                server="medmcp-neuro",
                tool="skull_strip",
                arguments={"input_path": "{{in_1}}", "output_dir": "{{in_2}}"},
                produces={"brain_path": "step1.brain_path"},
            )
        ],
        requires=[StackRequirement(stack="medmcp-neuro", image="ghcr.io/x:main", digest="sha")],
        manual_steps=["builtin:bash `ls`"],
    )


# ── schema ───────────────────────────────────────────────────────────────────


def test_from_dict_round_trips_to_dict() -> None:
    """to_dict → from_dict → to_dict is the identity, defaults and pins included."""
    original = _recipe()
    assert Recipe.from_dict(original.to_dict()) == original


def test_from_dict_ignores_keys_it_does_not_know() -> None:
    """An export envelope's marker and the blocks older exports carried load cleanly."""
    data = _recipe().to_dict()
    data.update({"medmcp_workflow": 1, "documentation": "# Steps", "prose": {"x": 1}})
    assert Recipe.from_dict(data) == _recipe()


def test_from_dict_of_an_empty_mapping_is_an_empty_recipe() -> None:
    """A recipe.yaml with nothing in it loads rather than raising."""
    recipe = Recipe.from_dict({})
    assert (recipe.name, recipe.description, recipe.steps) == ("", "", [])


def test_write_then_read_recipe(tmp_path: Path) -> None:
    """write_recipe creates the directory and read_recipe gets the same recipe back."""
    d = tmp_path / "wf"
    workflow.write_recipe(d, _recipe())
    assert sorted(p.name for p in d.iterdir()) == ["recipe.yaml"]
    assert workflow.read_recipe(d) == _recipe()
    # Field order is kept, so a diff of two recipe.yaml files reads naturally.
    assert list(yaml.safe_load((d / "recipe.yaml").read_text())) == [
        "name",
        "description",
        "inputs",
        "steps",
        "requires",
        "manual_steps",
    ]


# ── naming ───────────────────────────────────────────────────────────────────


def test_unique_name_numbers_a_taken_name(tmp_path: Path) -> None:
    """A free name is returned as is; a taken one gets -2, then -3."""
    assert workflow.unique_name(tmp_path, "strip") == "strip"
    (tmp_path / "strip").mkdir()
    assert workflow.unique_name(tmp_path, "strip") == "strip-2"
    (tmp_path / "strip-2").mkdir()
    assert workflow.unique_name(tmp_path, "strip") == "strip-3"


def test_unique_name_with_a_tag(tmp_path: Path) -> None:
    """With a tag the alternatives are -<tag>, then -<tag>-2."""
    (tmp_path / "strip").mkdir()
    assert workflow.unique_name(tmp_path, "strip", tag="imported") == "strip-imported"
    (tmp_path / "strip-imported").mkdir()
    assert workflow.unique_name(tmp_path, "strip", tag="imported") == "strip-imported-2"


# ── layout ───────────────────────────────────────────────────────────────────


def _legacy(root: Path, kind: str, name: str) -> Path:
    d = root / kind / name
    d.mkdir(parents=True)
    workflow.write_recipe(d, Recipe(name=name, description=kind))
    (d / "SKILL.md").write_text("---\nname: x\n---\n")
    (d / "prose.json").write_text("null")
    return d


def test_migrate_layout_folds_active_and_draft_into_the_root(tmp_path: Path) -> None:
    """Old active/ and draft/ entries move up; active wins a name, the draft is suffixed."""
    _legacy(tmp_path, "active", "strip")
    _legacy(tmp_path, "draft", "strip")
    _legacy(tmp_path, "draft", "register")

    workflow.migrate_layout(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["register", "strip", "strip-draft"]
    assert workflow.read_recipe(tmp_path / "strip").description == "active"
    assert workflow.read_recipe(tmp_path / "strip-draft").description == "draft"
    # The generated files of the skill era are gone; only the recipe remains.
    for name in ("register", "strip", "strip-draft"):
        assert sorted(p.name for p in (tmp_path / name).iterdir()) == ["recipe.yaml"]


def test_migrate_layout_leaves_the_new_layout_alone(tmp_path: Path) -> None:
    """A root already in the flat layout is untouched, and a missing root is fine."""
    workflow.write_recipe(tmp_path / "strip", _recipe())
    before = (tmp_path / "strip" / "recipe.yaml").stat().st_mtime_ns
    workflow.migrate_layout(tmp_path)
    assert (tmp_path / "strip" / "recipe.yaml").stat().st_mtime_ns == before
    workflow.migrate_layout(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_workflow_dir_requires_a_recipe(tmp_path: Path) -> None:
    """Only a directory holding recipe.yaml counts as a workflow."""
    workflow.write_recipe(tmp_path / "strip", _recipe())
    (tmp_path / "junk").mkdir()
    assert workflow.workflow_dir(tmp_path, "strip") == tmp_path / "strip"
    assert workflow.workflow_dir(tmp_path, "junk") is None
    assert workflow.workflow_dir(tmp_path, "nope") is None


def test_list_workflows_rows(tmp_path: Path) -> None:
    """Rows are sorted by name and carry the description; a broken recipe still lists."""
    workflow.write_recipe(tmp_path / "strip", _recipe())
    workflow.write_recipe(tmp_path / "align", Recipe(name="align", description="Align."))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "recipe.yaml").write_text("name: [unclosed")
    assert workflow.list_workflows(tmp_path) == [
        {"name": "align", "description": "Align."},
        {"name": "broken", "description": ""},
        {"name": "strip", "description": "Skull strip a brain MRI."},
    ]
    assert workflow.list_workflows(tmp_path / "absent") == []
