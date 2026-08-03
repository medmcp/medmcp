"""Tests for Tier-2 distillation (recipe extraction, parameterization, output)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

# pyright: reportPrivateUsage=false
from medmcp import distill, provcli, provenance
from medmcp.workflow import Recipe, RecipeStep, StackRequirement

JsonDict = dict[str, Any]

SESSION_ID = "abcd1234-1111-2222-3333-444455556666"


def _assistant_call(call_id: str, name: str, arguments: JsonDict) -> JsonDict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def _tool_result(call_id: str, content: str) -> JsonDict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# A two-step neuro pipeline: skull-strip a T1, then register the brain to MNI.
# Plus a read_file (exploratory, dropped) and a failed call (dropped).
_MESSAGES: list[JsonDict] = [
    {"role": "user", "content": "Skull strip and register the T1", "injected": False},
    _assistant_call("c0", "read_file", {"path": "data/x/notes.txt"}),
    _tool_result("c0", "ok: True\ncontent: hello"),
    _assistant_call(
        "c1",
        "medmcp-neuro_skull_strip",
        {"device": "cuda", "input_path": "data/x/t1.nii.gz", "output_dir": "data/x"},
    ),
    _tool_result(
        "c1",
        "ok: True\nstructured: {'brain_path': 'data/x/t1_brain.nii.gz', "
        "'input_path': 'data/x/t1.nii.gz', 'device': 'cuda'}",
    ),
    _assistant_call(
        "c2",
        "medmcp-neuro_register_to_template",
        {"input_path": "data/x/t1_brain.nii.gz", "skull_stripped": True},
    ),
    _tool_result("c2", "ok: True\nstructured: {'registered_path': 'data/x/t1_mni.nii.gz'}"),
    _assistant_call("c3", "medmcp-neuro_coregister", {"input_path": "data/x/flair.nii.gz"}),
    _tool_result("c3", "ok: False\nerror: boom"),
]


class TestBuildRecipe:
    """build_recipe extracts and parameterizes the executed pipeline."""

    def _recipe(self) -> Recipe:
        return distill.build_recipe(
            _MESSAGES, server_names=["medmcp-neuro"], name="t", description="d"
        )

    def test_drops_exploratory_and_failed(self) -> None:
        """read_file (exploratory) and the failed coregister are excluded."""
        recipe = self._recipe()
        assert [s.tool for s in recipe.steps] == ["skull_strip", "register_to_template"]

    def test_server_split(self) -> None:
        """The server prefix is stripped from each step's tool name."""
        recipe = self._recipe()
        assert all(s.server == "medmcp-neuro" for s in recipe.steps)

    def test_input_paths_lifted_to_placeholders(self) -> None:
        """First-seen input paths become workflow inputs."""
        recipe = self._recipe()
        step1 = recipe.steps[0]
        assert step1.arguments["input_path"] == "{{in_1}}"
        assert step1.arguments["device"] == "cuda"  # non-path left literal
        examples = {i.example for i in recipe.inputs}
        assert "data/x/t1.nii.gz" in examples

    def test_step_output_reused_by_later_step(self) -> None:
        """A path produced by step 1 is referenced symbolically by step 2."""
        recipe = self._recipe()
        step2 = recipe.steps[1]
        assert step2.arguments["input_path"] == "{{step1.brain_path}}"
        assert step2.arguments["skull_stripped"] is True
        assert recipe.steps[0].produces["brain_path"] == "step1.brain_path"

    def test_inputs_get_usage_descriptions(self) -> None:
        """Each lifted input is described by the tool/arg that first used it."""
        recipe = self._recipe()
        in_1 = next(i for i in recipe.inputs if i.name == "in_1")
        assert in_1.description == "the input_path for medmcp-neuro:skull_strip"

    def test_drops_rejected_call(self) -> None:
        """A tool the user rejected is excluded from the recipe."""
        messages: list[JsonDict] = [
            _assistant_call("r1", "medmcp-neuro_skull_strip", {"input_path": "data/x/t1.nii.gz"}),
            _tool_result("r1", "User rejected the tool call, provide an alternative plan"),
            *_MESSAGES[3:7],  # the two real neuro steps
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-neuro"], name="t", description="d"
        )
        assert [s.tool for s in recipe.steps] == ["skull_strip", "register_to_template"]

    def test_drops_cancelled_call(self) -> None:
        """A tool call wrapped in the user_cancellation tag is excluded."""
        messages: list[JsonDict] = [
            _assistant_call("x1", "medmcp-neuro_skull_strip", {"input_path": "data/a.nii"}),
            _tool_result("x1", "<user_cancellation>skull_strip</user_cancellation>"),
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-neuro"], name="t", description="d"
        )
        assert recipe.steps == []

    def test_drops_readonly_shell_inspection(self) -> None:
        """A bash call running a read-only inspection (ls) is excluded as noise."""
        messages: list[JsonDict] = [
            _assistant_call("e0", "bash", {"command": "ls -la data/x"}),
            _tool_result("e0", "ok: True\ncontent: t1.nii.gz"),
            *_MESSAGES[3:7],  # the two real neuro steps
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-neuro"], name="t", description="d"
        )
        assert [s.tool for s in recipe.steps] == ["skull_strip", "register_to_template"]

    def test_writing_shell_call_becomes_manual_step(self) -> None:
        """A writing bash command is dropped from steps and recorded as a manual step."""
        messages: list[JsonDict] = [
            _assistant_call("w0", "bash", {"command": "cat a.txt > b.txt"}),
            _tool_result("w0", "ok: True"),
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-neuro"], name="t", description="d"
        )
        assert recipe.steps == []
        assert recipe.manual_steps == ["builtin:bash `cat a.txt > b.txt`"]

    def test_drops_internal_warmup_tool(self) -> None:
        """Pool-machinery tools (warmup) are dropped entirely — not a step nor manual."""
        messages: list[JsonDict] = [
            _assistant_call("w0", "medmcp-neuro_warmup", {}),
            _tool_result("w0", "ok: True"),
            _assistant_call("s0", "medmcp-neuro_skull_strip", {"input_path": "/data/a.nii.gz"}),
            _tool_result("s0", "ok: True"),
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-neuro"], name="t", description="d"
        )
        assert [(s.server, s.tool) for s in recipe.steps] == [("medmcp-neuro", "skull_strip")]
        assert recipe.manual_steps == []

    def test_internal_tools_cover_proxy_hidden(self) -> None:
        """distill.INTERNAL_TOOLS must cover every tool the proxy hides (keep in sync)."""
        from medmcp.proxy import _HIDDEN_TOOLS

        assert _HIDDEN_TOOLS <= distill.INTERNAL_TOOLS

    def test_drops_tool_error_rendered_as_ok_true(self) -> None:
        """A wrapped tool exception (vibe renders ``ok: True``) is dropped as failed.

        This is the real-world skull_strip crash: HD-BET exits non-zero, the tool
        raises, FastMCP returns it as an error result, but vibe-acp's transcript
        renders ``ok: True`` with the error only in the ``text:`` field.
        """
        failed = (
            "ok: True\n"
            "server: stdio:/path/to/medmcp-neuro\n"
            "tool: skull_strip\n"
            "text: Error executing tool skull_strip: HD-BET failed (exit 1): \n"
            "structured: None"
        )
        messages: list[JsonDict] = [
            _assistant_call(
                "f1",
                "medmcp-neuro_skull_strip",
                {"device": "cuda", "input_path": "data/x/t1.nii.gz"},
            ),
            _tool_result("f1", failed),
            *_MESSAGES[3:7],  # the two real neuro steps that did succeed
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-neuro"], name="t", description="d"
        )
        assert [s.tool for s in recipe.steps] == ["skull_strip", "register_to_template"]


class TestProseParsing:
    """_parse_prose_response handles fenced and inline JSON."""

    def test_plain_json(self) -> None:
        """A bare JSON object parses."""
        parsed = distill._parse_prose_response('{"name": "x", "description": "y"}')
        assert parsed == {"name": "x", "description": "y"}

    def test_fenced_json(self) -> None:
        """A ```json fenced block is unwrapped."""
        parsed = distill._parse_prose_response('```json\n{"name": "x"}\n```')
        assert parsed == {"name": "x"}

    def test_garbage_returns_none(self) -> None:
        """Unparseable text returns None."""
        assert distill._parse_prose_response("not json at all") is None


def test_slugify() -> None:
    """Slugs are lowercased, hyphenated, and bounded."""
    assert distill.slugify("Skull Strip & Register!") == "skull-strip-register"
    assert distill.slugify("") == "workflow"


class TestFirstUserMessage:
    """_first_user_message yields a clean seed request for naming/description."""

    def test_strips_workspace_note(self) -> None:
        """The appended [workspace context: …] note is stripped from the seed request."""
        messages = [
            {
                "role": "user",
                "content": (
                    "Skull strip this scan\n\n"
                    '[workspace context: the file "/data/sub-01_T1w.nii.gz" is '
                    "currently open in the viewer]"
                ),
            }
        ]
        assert distill._first_user_message(messages) == "Skull strip this scan"

    def test_skips_injected_message(self) -> None:
        """An injected message is not treated as the user's request."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "injected context", "injected": True},
            {"role": "user", "content": "the real request"},
        ]
        assert distill._first_user_message(messages) == "the real request"

    def test_plain_message_trimmed(self) -> None:
        """A message with no note is returned, trimmed."""
        assert distill._first_user_message([{"role": "user", "content": "  do it  "}]) == "do it"


class TestRenderSkillMd:
    """render_skill_md produces a valid SKILL.md with or without prose."""

    def test_mechanical_fallback(self) -> None:
        """Without prose, steps render mechanically and inputs are listed."""
        recipe = distill.build_recipe(
            _MESSAGES, server_names=["medmcp-neuro"], name="my-flow", description="desc"
        )
        md = distill.render_skill_md(recipe, None)
        assert md.startswith("---\nname: my-flow\n")
        assert "## Steps" in md
        assert "`medmcp-neuro:skull_strip`" in md
        assert "## Inputs" in md

    def test_uses_prose_when_present(self) -> None:
        """Prose name/description/steps override the mechanical defaults."""
        recipe = distill.build_recipe(
            _MESSAGES, server_names=["medmcp-neuro"], name="my-flow", description="desc"
        )
        prose: JsonDict = {
            "description": "A nicer description.",
            "steps_markdown": "1. Do the thing.",
            "gotchas_markdown": "- Mind the gap.",
        }
        md = distill.render_skill_md(recipe, prose)
        assert "A nicer description." in md
        assert "Do the thing." in md
        assert "## Gotchas" in md

    def test_lists_required_tools(self) -> None:
        """A ## Tools section names every server:tool the workflow needs."""
        recipe = distill.build_recipe(
            _MESSAGES, server_names=["medmcp-neuro"], name="my-flow", description="desc"
        )
        md = distill.render_skill_md(recipe, None)
        assert "## Tools" in md
        assert "`medmcp-neuro:skull_strip` — from the `medmcp-neuro` stack" in md
        assert "`medmcp-neuro:register_to_template`" in md

    def test_required_tools_dedup(self) -> None:
        """Distinct MCP tools are listed once, in first-seen order."""
        recipe = Recipe(name="f", description="d")
        recipe.steps = [
            RecipeStep(server="medmcp-neuro", tool="skull_strip", arguments={}),
            RecipeStep(server="medmcp-neuro", tool="skull_strip", arguments={}),
            RecipeStep(server="medmcp-neuro", tool="coregister", arguments={}),
        ]
        assert distill._required_tools(recipe) == [
            ("medmcp-neuro", "skull_strip"),
            ("medmcp-neuro", "coregister"),
        ]

    def test_builtin_calls_recorded_as_manual_steps(self) -> None:
        """Built-in (non-MCP) calls are dropped but surfaced as manual steps in Gotchas."""
        messages: list[JsonDict] = [
            _assistant_call("b0", "bash", {"command": "convert a b"}),
            _tool_result("b0", "ok: True"),
        ]
        recipe = distill.build_recipe(messages, server_names=[], name="f", description="d")
        assert distill._required_tools(recipe) == []
        assert recipe.manual_steps == ["builtin:bash `convert a b`"]
        md = distill.render_skill_md(recipe, None)
        assert "## Gotchas" in md
        assert "builtin:bash `convert a b`" in md


class TestRequirements:
    """build_requirements + the ## Requirements rendering."""

    def test_filters_to_used_stacks_and_pins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only stacks the recipe uses are required; container→image+digest, uv→version."""
        recipe = Recipe(name="w", description="d")
        recipe.steps = [
            RecipeStep(server="medmcp-neuro", tool="skull_strip", arguments={}),
            RecipeStep(server="builtin", tool="bash", arguments={}),  # needs no stack
        ]
        manifest: JsonDict = {
            "stacks": [
                {"name": "medmcp-neuro", "image": "ghcr.io/medmcp/neuro:main"},
                {"name": "medmcp-dicom", "version": "0.1.0"},  # not used → excluded
            ]
        }

        def _fake_digest(_image: str) -> str:
            return "sha256:abc"

        monkeypatch.setattr(distill, "resolve_digest", _fake_digest)
        reqs = distill.build_requirements(recipe, manifest)
        assert [r.stack for r in reqs] == ["medmcp-neuro"]
        assert reqs[0].image == "ghcr.io/medmcp/neuro:main"
        assert reqs[0].digest == "sha256:abc"

    def test_used_stack_absent_from_manifest_listed_by_name(self) -> None:
        """A used stack with no manifest entry is still listed (importer needs it)."""
        recipe = Recipe(name="w", description="d")
        recipe.steps = [RecipeStep(server="medmcp-x", tool="t", arguments={})]
        assert distill.build_requirements(recipe, None) == [StackRequirement(stack="medmcp-x")]

    def test_requirements_rendered_in_skill_md(self) -> None:
        """## Requirements pins each stack by image(+digest) or version."""
        recipe = Recipe(name="w", description="d")
        recipe.requires = [
            StackRequirement(
                stack="medmcp-neuro", image="ghcr.io/medmcp/neuro:main", digest="sha256:abc"
            ),
            StackRequirement(stack="medmcp-dicom", version="0.1.0"),
        ]
        md = distill.render_skill_md(recipe, None)
        assert "## Requirements" in md
        assert "image `ghcr.io/medmcp/neuro:main` (`sha256:abc`)" in md
        assert "version `0.1.0`" in md


def _write_fake_session(root: Path) -> None:
    """Create a minimal vibe session log dir under *root* (a VIBE_HOME)."""
    sess = root / "logs" / "session" / "session_20260101_000000_abcd1234"
    sess.mkdir(parents=True)
    (sess / "meta.json").write_text(json.dumps({"session_id": SESSION_ID}))
    with (sess / "messages.jsonl").open("w") as f:
        for msg in _MESSAGES:
            f.write(json.dumps(msg) + "\n")


def test_distill_session_end_to_end(tmp_path: Path) -> None:
    """distill_session writes recipe.yaml + SKILL.md from a session's raw log."""
    _write_fake_session(tmp_path)
    workflows_root = tmp_path / "workflows"
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        draft_dir = distill.distill_session(
            SESSION_ID, use_llm=False, workflows_root=workflows_root
        )

    assert (draft_dir / "SKILL.md").exists()
    recipe_yaml = (draft_dir / "recipe.yaml").read_text()
    recipe = yaml.safe_load(recipe_yaml)
    assert [s["tool"] for s in recipe["steps"]] == ["skull_strip", "register_to_template"]
    assert recipe["steps"][1]["arguments"]["input_path"] == "{{step1.brain_path}}"


def test_distill_session_missing_log_raises(tmp_path: Path) -> None:
    """A session with no raw log raises FileNotFoundError."""
    (tmp_path / "logs" / "session").mkdir(parents=True)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        try:
            distill.distill_session(SESSION_ID, use_llm=False)
        except FileNotFoundError:
            return
    raise AssertionError("expected FileNotFoundError")


def test_distill_session_spans_compaction_chain(tmp_path: Path) -> None:
    """Tool calls made after a context compaction distill too.

    vibe rolls a compacted conversation over to a new session dir whose
    meta.json backlinks the original via parent_session_id; distillation must
    read the whole chain, not just the pre-compaction prefix.
    """
    _write_fake_session(tmp_path)
    cont_id = "ef567890-9999-8888-7777-666655554444"
    cont = tmp_path / "logs" / "session" / f"session_20260101_010000_{cont_id[:8]}"
    cont.mkdir(parents=True)
    (cont / "meta.json").write_text(
        json.dumps({"session_id": cont_id, "parent_session_id": SESSION_ID})
    )
    post_compaction: list[JsonDict] = [
        _assistant_call(
            "c9",
            "medmcp-neuro_coregister",
            {"input_path": "data/x/t1_mni.nii.gz", "reference": "data/x/flair.nii.gz"},
        ),
        _tool_result(
            "c9", "ok: True\nstructured: {'coregistered_path': 'data/x/flair_reg.nii.gz'}"
        ),
    ]
    with (cont / "messages.jsonl").open("w") as f:
        for msg in post_compaction:
            f.write(json.dumps(msg) + "\n")

    with patch.object(provenance, "VIBE_HOME", tmp_path):
        draft_dir = distill.distill_session(
            SESSION_ID, use_llm=False, workflows_root=tmp_path / "workflows"
        )

    recipe = yaml.safe_load((draft_dir / "recipe.yaml").read_text())
    assert [s["tool"] for s in recipe["steps"]] == [
        "skull_strip",
        "register_to_template",
        "coregister",
    ]


def test_promote_moves_draft_to_active(tmp_path: Path) -> None:
    """Promote relocates a reviewed draft into active/ for skill discovery."""
    draft = tmp_path / "workflows" / "draft" / "my-flow"
    draft.mkdir(parents=True)
    (draft / "SKILL.md").write_text("---\nname: my-flow\n---\n")
    (draft / "recipe.yaml").write_text("name: my-flow\n")
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        rc = provcli._promote("my-flow")

    assert rc == 0
    active = tmp_path / "workflows" / "active" / "my-flow"
    assert (active / "SKILL.md").exists()
    assert not draft.exists()


def test_promote_missing_draft_errors(tmp_path: Path) -> None:
    """Promoting a non-existent draft returns a non-zero exit code."""
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provcli._promote("nope") == 1


# ── Draft editing: rename / refine / discard ─────────────────────────────────


def _make_draft(tmp_path: Path) -> tuple[Path, Path]:
    """Distill a real (no-LLM) draft; return (workflows_root, draft_dir)."""
    _write_fake_session(tmp_path)
    workflows_root = tmp_path / "workflows"
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        draft_dir = distill.distill_session(
            SESSION_ID, use_llm=False, workflows_root=workflows_root
        )
    return workflows_root, draft_dir


def test_rename_draft_relocates_and_rewrites(tmp_path: Path) -> None:
    """Rename moves the draft dir and updates the name in recipe.yaml + SKILL.md."""
    workflows_root, draft_dir = _make_draft(tmp_path)
    old_name = draft_dir.name

    new_dir = distill.rename_draft(old_name, "Fancy New Name!", workflows_root=workflows_root)

    assert new_dir.name == "fancy-new-name"
    assert not draft_dir.exists()  # old dir gone
    recipe = distill.load_recipe(new_dir)
    assert recipe.name == "fancy-new-name"
    assert recipe.steps  # the pipeline is preserved across the rename
    assert (new_dir / "SKILL.md").read_text().startswith("---\nname: fancy-new-name\n")


def test_rename_draft_missing_raises(tmp_path: Path) -> None:
    """Renaming a draft that doesn't exist raises FileNotFoundError."""
    workflows_root = tmp_path / "workflows"
    try:
        distill.rename_draft("nope", "x", workflows_root=workflows_root)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_discard_draft_removes_dir(tmp_path: Path) -> None:
    """Discard deletes the draft directory."""
    workflows_root, draft_dir = _make_draft(tmp_path)
    distill.discard_draft(draft_dir.name, workflows_root=workflows_root)
    assert not draft_dir.exists()


def test_discard_draft_missing_raises(tmp_path: Path) -> None:
    """Discarding a non-existent draft raises FileNotFoundError."""
    try:
        distill.discard_draft("nope", workflows_root=tmp_path / "workflows")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_refine_draft_rewrites_prose_keeps_identity(tmp_path: Path) -> None:
    """Refine regenerates prose (mocked) while keeping the draft's name and steps."""
    workflows_root, draft_dir = _make_draft(tmp_path)
    name = draft_dir.name
    new_prose: JsonDict = {
        "name": "model-tried-to-rename",  # must be ignored — identity is preserved
        "description": "Refined description.",
        "steps_markdown": "1. Refined step.",
        "gotchas_markdown": "",
    }
    with patch.object(distill, "generate_prose", return_value=new_prose):
        out = distill.refine_draft(name, "make it generic", workflows_root=workflows_root)

    assert out == draft_dir  # same dir; refine never relocates
    recipe = distill.load_recipe(out)
    assert recipe.name == name  # identity preserved despite the model's suggestion
    assert recipe.description == "Refined description."
    assert "Refined step." in (out / "SKILL.md").read_text()


def test_unpromote_moves_active_back_to_draft(tmp_path: Path) -> None:
    """unpromote_workflow returns a promoted workflow to draft/ for editing."""
    active = tmp_path / "workflows" / "active" / "flow"
    active.mkdir(parents=True)
    (active / "SKILL.md").write_text("---\nname: flow\n---\n")
    (active / "recipe.yaml").write_text("name: flow\n")

    draft = distill.unpromote_workflow("flow", workflows_root=tmp_path / "workflows")

    assert draft == tmp_path / "workflows" / "draft" / "flow"
    assert (draft / "SKILL.md").exists()
    assert not active.exists()


def test_unpromote_missing_raises(tmp_path: Path) -> None:
    """Unpromoting a non-promoted workflow raises FileNotFoundError."""
    try:
        distill.unpromote_workflow("nope", workflows_root=tmp_path / "workflows")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_promote_then_unpromote_round_trips(tmp_path: Path) -> None:
    """A draft promoted and then unpromoted lands back in draft/ intact."""
    _, draft_dir = _make_draft(tmp_path)
    name = draft_dir.name
    distill.promote_draft(name, workflows_root=tmp_path / "workflows")
    assert not draft_dir.exists()
    back = distill.unpromote_workflow(name, workflows_root=tmp_path / "workflows")
    assert back == draft_dir
    assert (back / "SKILL.md").exists()


def test_delete_workflow_removes_active(tmp_path: Path) -> None:
    """delete_workflow removes a promoted workflow from active/."""
    active = tmp_path / "workflows" / "active" / "flow"
    active.mkdir(parents=True)
    (active / "SKILL.md").write_text("---\nname: flow\n---\n")
    removed = distill.delete_workflow("flow", workflows_root=tmp_path / "workflows")
    assert removed == active
    assert not active.exists()


def test_delete_workflow_removes_draft(tmp_path: Path) -> None:
    """delete_workflow also removes an unpromoted draft."""
    _, draft_dir = _make_draft(tmp_path)
    removed = distill.delete_workflow(draft_dir.name, workflows_root=tmp_path / "workflows")
    assert removed == draft_dir
    assert not draft_dir.exists()


def test_delete_workflow_missing_raises(tmp_path: Path) -> None:
    """Deleting a non-existent workflow raises FileNotFoundError."""
    try:
        distill.delete_workflow("nope", workflows_root=tmp_path / "workflows")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_refine_draft_model_failure_raises(tmp_path: Path) -> None:
    """When the model returns nothing, refine raises rather than silently no-op."""
    workflows_root, draft_dir = _make_draft(tmp_path)
    with patch.object(distill, "generate_prose", return_value=None):
        try:
            distill.refine_draft(draft_dir.name, "x", workflows_root=workflows_root)
        except RuntimeError:
            return
    raise AssertionError("expected RuntimeError")
