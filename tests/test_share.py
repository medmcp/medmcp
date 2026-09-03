"""Tests for workflow export/import (single inline-YAML sharing)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from medmcp import distill, share, workflow
from medmcp.workflow import Recipe, RecipeStep, StackRequirement, WorkflowInput


def _sample_recipe() -> Recipe:
    return Recipe(
        name="skull-strip-register",
        description="Skull strip then register a brain MRI to template.",
        inputs=[WorkflowInput(name="in_1", example="/data/t1.nii.gz", description="the scan")],
        steps=[
            RecipeStep(
                server="medmcp-neuro",
                tool="skull_strip",
                arguments={"input_path": "{{in_1}}"},
                produces={"brain_path": "step1.brain_path"},
            ),
            RecipeStep(
                server="medmcp-neuro",
                tool="register_to_template",
                arguments={"input_path": "{{step1.brain_path}}"},
            ),
        ],
        requires=[
            StackRequirement(
                stack="medmcp-neuro", image="ghcr.io/medmcp/neuro:main", digest="sha256:abc"
            )
        ],
        manual_steps=["builtin:bash `convert a b`"],
    )


class TestExport:
    """export_workflow serializes a workflow dir into the inline-YAML envelope."""

    def test_export_missing_raises(self, tmp_path: Path) -> None:
        """Exporting a name with no workflow dir raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            share.export_workflow("nope", workflows_root=tmp_path)

    def test_export_is_the_versioned_recipe(self, tmp_path: Path) -> None:
        """The envelope is the recipe's dict form behind a format marker — nothing else."""
        workflow.write_recipe(tmp_path / "skull-strip-register", _sample_recipe())
        text = share.export_workflow("skull-strip-register", workflows_root=tmp_path)
        env = yaml.safe_load(text)
        assert env.pop(share.FORMAT_KEY) == share.FORMAT_VERSION
        assert env == _sample_recipe().to_dict()

    def test_export_fills_missing_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A required image with no digest is resolved best-effort at export time."""
        recipe = _sample_recipe()
        recipe.requires = [
            StackRequirement(stack="medmcp-neuro", image="ghcr.io/medmcp/neuro:main")
        ]
        workflow.write_recipe(tmp_path / "skull-strip-register", recipe)

        def _fake_digest(_image: str) -> str:
            return "sha256:resolved"

        monkeypatch.setattr(distill, "resolve_digest", _fake_digest)
        env = yaml.safe_load(share.export_workflow("skull-strip-register", workflows_root=tmp_path))
        assert env["requires"][0]["digest"] == "sha256:resolved"


class TestImport:
    """import_workflow validates the envelope and reconstructs the workflow."""

    def test_round_trip_preserves_recipe(self, tmp_path: Path) -> None:
        """Export → import reconstructs an equivalent recipe as a fresh workflow."""
        original = _sample_recipe()
        workflow.write_recipe(tmp_path / "src" / "skull-strip-register", original)
        text = share.export_workflow("skull-strip-register", workflows_root=tmp_path / "src")

        target = share.import_workflow(text, workflows_root=tmp_path / "dst")
        assert target == tmp_path / "dst" / "skull-strip-register"
        assert distill.load_recipe(target).to_dict() == original.to_dict()
        assert sorted(p.name for p in target.iterdir()) == ["recipe.yaml"]

    def test_import_ignores_the_blocks_older_exports_carried(self, tmp_path: Path) -> None:
        """A file from a release that wrote documentation and prose still imports."""
        env = {share.FORMAT_KEY: 1, **_sample_recipe().to_dict()}
        env["documentation"] = "# Skull strip register workflow\n\n## Steps\n…"
        env["prose"] = {"name": "x", "steps_markdown": "1. …"}
        target = share.import_workflow(yaml.safe_dump(env), workflows_root=tmp_path)
        assert distill.load_recipe(target).to_dict() == _sample_recipe().to_dict()

    def test_import_rejects_non_mapping(self, tmp_path: Path) -> None:
        """A payload that isn't a YAML mapping is rejected."""
        with pytest.raises(share.WorkflowShareError):
            share.import_workflow("- just\n- a\n- list\n", workflows_root=tmp_path)

    def test_import_rejects_unknown_version(self, tmp_path: Path) -> None:
        """An envelope with an unrecognized format version is rejected."""
        text = yaml.safe_dump({share.FORMAT_KEY: 99, "name": "x", "steps": [{"server": "s"}]})
        with pytest.raises(share.WorkflowShareError, match="unsupported workflow format"):
            share.import_workflow(text, workflows_root=tmp_path)

    def test_import_rejects_no_steps(self, tmp_path: Path) -> None:
        """An envelope with no steps is rejected (nothing to replay)."""
        text = yaml.safe_dump({share.FORMAT_KEY: 1, "name": "x", "steps": []})
        with pytest.raises(share.WorkflowShareError, match="no steps"):
            share.import_workflow(text, workflows_root=tmp_path)

    def test_import_collision_gets_suffix(self, tmp_path: Path) -> None:
        """Importing the same workflow twice never clobbers — the second is suffixed."""
        workflow.write_recipe(tmp_path / "skull-strip-register", _sample_recipe())
        text = share.export_workflow("skull-strip-register", workflows_root=tmp_path)
        first = share.import_workflow(text, workflows_root=tmp_path)
        assert first.name == "skull-strip-register-imported"
        again = share.import_workflow(text, workflows_root=tmp_path)
        assert again.name == "skull-strip-register-imported-2"

    def test_import_into_a_legacy_root_sees_its_workflows(self, tmp_path: Path) -> None:
        """A name taken by an old draft/ entry is still a collision after the fold."""
        workflow.write_recipe(tmp_path / "draft" / "skull-strip-register", _sample_recipe())
        env = {share.FORMAT_KEY: 1, **_sample_recipe().to_dict()}
        target = share.import_workflow(yaml.safe_dump(env), workflows_root=tmp_path)
        assert target.name == "skull-strip-register-imported"
        assert not (tmp_path / "draft").exists()


def test_export_import_round_trips_a_derived_default(tmp_path: Path) -> None:
    """A shared workflow must keep its derived defaults.

    Export writes `default` (it is part of the input's dict form), so dropping it
    on import is silent: the recipient gets back the very input the default
    exists to spare them, and only notices when the run form asks for it.
    """
    recipe = Recipe(
        name="wf",
        description="d",
        inputs=[
            WorkflowInput(name="in_1", example="/a/t1.nii.gz"),
            WorkflowInput(name="in_2", example="/a", default="{{dir(in_1)}}"),
        ],
        steps=[RecipeStep(server="s", tool="t", arguments={"p": "{{in_1}}", "o": "{{in_2}}"})],
    )
    root = tmp_path / "workflows"
    workflow.write_recipe(root / "wf", recipe)

    envelope = share.export_workflow("wf", workflows_root=root)
    assert "{{dir(in_1)}}" in envelope

    target = share.import_workflow(envelope, workflows_root=tmp_path / "dest")
    loaded = distill.load_recipe(target)
    assert [i.default for i in loaded.inputs] == ["", "{{dir(in_1)}}"]
