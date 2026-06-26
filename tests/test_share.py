"""Tests for workflow export/import (single inline-YAML sharing)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from medmcp import distill, share
from medmcp.workflow import Recipe, RecipeStep, StackRequirement, WorkflowInput


def _make_draft(root: Path, recipe: Recipe) -> Path:
    """Write a draft workflow dir (recipe.yaml + SKILL.md + prose.json) under *root*."""
    draft = root / "draft" / recipe.name
    draft.mkdir(parents=True)
    (draft / "recipe.yaml").write_text(
        yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (draft / "SKILL.md").write_text(distill.render_skill_md(recipe, None), encoding="utf-8")
    (draft / "prose.json").write_text("null", encoding="utf-8")
    return draft


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
        """Exporting a name with no draft/active dir raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            share.export_workflow("nope", workflows_root=tmp_path)

    def test_export_contains_envelope_and_docs(self, tmp_path: Path) -> None:
        """Export emits the versioned envelope with steps, requires, and docs."""
        _make_draft(tmp_path, _sample_recipe())
        text = share.export_workflow("skull-strip-register", workflows_root=tmp_path)
        env = yaml.safe_load(text)
        assert env[share.FORMAT_KEY] == share.FORMAT_VERSION
        assert env["name"] == "skull-strip-register"
        assert [s["tool"] for s in env["steps"]] == ["skull_strip", "register_to_template"]
        assert env["requires"][0]["digest"] == "sha256:abc"
        assert env["manual_steps"] == ["builtin:bash `convert a b`"]
        assert "## Requirements" in env["documentation"]

    def test_export_fills_missing_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A required image with no digest is resolved best-effort at export time."""
        recipe = _sample_recipe()
        recipe.requires = [
            StackRequirement(stack="medmcp-neuro", image="ghcr.io/medmcp/neuro:main")
        ]
        _make_draft(tmp_path, recipe)

        def _fake_digest(_image: str) -> str:
            return "sha256:resolved"

        monkeypatch.setattr(distill, "resolve_digest", _fake_digest)
        env = yaml.safe_load(share.export_workflow("skull-strip-register", workflows_root=tmp_path))
        assert env["requires"][0]["digest"] == "sha256:resolved"


class TestImport:
    """import_workflow validates the envelope and reconstructs a reviewable draft."""

    def test_round_trip_preserves_recipe(self, tmp_path: Path) -> None:
        """Export → import reconstructs an equivalent recipe as a fresh draft."""
        original = _sample_recipe()
        _make_draft(tmp_path / "src", original)
        text = share.export_workflow("skull-strip-register", workflows_root=tmp_path / "src")

        draft = share.import_workflow(text, workflows_root=tmp_path / "dst")
        assert draft == tmp_path / "dst" / "draft" / "skull-strip-register"
        imported = distill.load_recipe(draft)
        assert imported.to_dict() == original.to_dict()
        assert (draft / "SKILL.md").exists()

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
        _make_draft(tmp_path, _sample_recipe())
        text = share.export_workflow("skull-strip-register", workflows_root=tmp_path)
        draft = share.import_workflow(text, workflows_root=tmp_path)
        assert draft.name == "skull-strip-register-imported"
        again = share.import_workflow(text, workflows_root=tmp_path)
        assert again.name == "skull-strip-register-imported-2"
