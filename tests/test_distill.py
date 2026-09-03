"""Tests for Tier-2 distillation (recipe extraction, parameterization, output)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

# pyright: reportPrivateUsage=false
from medmcp import distill, provenance
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

    def test_prefers_persisted_display_content(self) -> None:
        """The note-free user_display_content (vibe ≥2.23) wins over the stored text."""
        messages = [
            {
                "role": "user",
                "content": "Skull strip this scan\n\n[workspace context: …]",
                "user_display_content": {
                    "version": "1",
                    "host": "medmcp",
                    "content": [{"type": "text", "text": "Skull strip this scan"}],
                },
            }
        ]
        assert distill._first_user_message(messages) == "Skull strip this scan"

    def test_empty_display_content_falls_back_to_stripping(self) -> None:
        """A present-but-empty display payload does not blank the seed request."""
        empty: list[object] = []
        messages = [
            {
                "role": "user",
                "content": (
                    "Register this scan\n\n"
                    '[workspace context: the file "/data/t1.nii.gz" is currently open '
                    "in the viewer]"
                ),
                "user_display_content": {"version": "1", "host": "medmcp", "content": empty},
            }
        ]
        assert distill._first_user_message(messages) == "Register this scan"


def test_builtin_calls_recorded_as_manual_steps() -> None:
    """Built-in (non-MCP) calls are dropped from the steps but kept as manual steps."""
    messages: list[JsonDict] = [
        _assistant_call("b0", "bash", {"command": "convert a b"}),
        _tool_result("b0", "ok: True"),
    ]
    recipe = distill.build_recipe(messages, server_names=[], name="f", description="d")
    assert recipe.steps == []
    assert recipe.manual_steps == ["builtin:bash `convert a b`"]


class TestRequirements:
    """build_requirements pins the stacks a recipe uses."""

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


def _write_fake_session(root: Path) -> None:
    """Create a minimal vibe session log dir under *root* (a VIBE_HOME)."""
    sess = root / "logs" / "session" / "session_20260101_000000_abcd1234"
    sess.mkdir(parents=True)
    (sess / "meta.json").write_text(json.dumps({"session_id": SESSION_ID}))
    with (sess / "messages.jsonl").open("w") as f:
        for msg in _MESSAGES:
            f.write(json.dumps(msg) + "\n")


def test_distill_session_end_to_end(tmp_path: Path) -> None:
    """distill_session writes one recipe.yaml, named from the opening request."""
    _write_fake_session(tmp_path)
    workflows_root = tmp_path / "workflows"
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        target = distill.distill_session(SESSION_ID, workflows_root=workflows_root)

    assert target == workflows_root / "skull-strip-and-register-the-t1"
    assert sorted(p.name for p in target.iterdir()) == ["recipe.yaml"]
    recipe = yaml.safe_load((target / "recipe.yaml").read_text())
    assert recipe["name"] == "skull-strip-and-register-the-t1"
    assert recipe["description"] == "Skull strip and register the T1"
    assert [s["tool"] for s in recipe["steps"]] == ["skull_strip", "register_to_template"]
    assert recipe["steps"][1]["arguments"]["input_path"] == "{{step1.brain_path}}"


def test_distill_names_the_workflow_after_the_chat(tmp_path: Path) -> None:
    """A chat title names the workflow; the opening request stays the description."""
    _write_fake_session(tmp_path)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        target = distill.distill_session(
            SESSION_ID,
            name_hint="Brain MRI: strip & register",
            workflows_root=tmp_path / "workflows",
        )
    assert target.name == "brain-mri-strip-register"
    assert distill.load_recipe(target).description == "Skull strip and register the T1"


def test_distill_never_replaces_an_existing_workflow(tmp_path: Path) -> None:
    """Two chats with the same title give two workflows, not one overwriting the other."""
    _write_fake_session(tmp_path)
    root = tmp_path / "workflows"
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        first = distill.distill_session(SESSION_ID, name_hint="Strip", workflows_root=root)
        second = distill.distill_session(SESSION_ID, name_hint="Strip", workflows_root=root)
    assert (first.name, second.name) == ("strip", "strip-2")
    assert distill.load_recipe(second).name == "strip-2"


def test_distill_session_missing_log_raises(tmp_path: Path) -> None:
    """A session with no raw log raises FileNotFoundError."""
    (tmp_path / "logs" / "session").mkdir(parents=True)
    with patch.object(provenance, "VIBE_HOME", tmp_path), pytest.raises(FileNotFoundError):
        distill.distill_session(SESSION_ID)


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
        target = distill.distill_session(SESSION_ID, workflows_root=tmp_path / "workflows")

    recipe = yaml.safe_load((target / "recipe.yaml").read_text())
    assert [s["tool"] for s in recipe["steps"]] == [
        "skull_strip",
        "register_to_template",
        "coregister",
    ]


# ── Rename / delete ──────────────────────────────────────────────────────────


def _make_workflow(tmp_path: Path) -> tuple[Path, Path]:
    """Distill a real workflow; return (workflows_root, workflow_dir)."""
    _write_fake_session(tmp_path)
    workflows_root = tmp_path / "workflows"
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        target = distill.distill_session(SESSION_ID, workflows_root=workflows_root)
    return workflows_root, target


def test_rename_workflow_relocates_and_rewrites(tmp_path: Path) -> None:
    """Rename moves the directory and updates the name in recipe.yaml."""
    workflows_root, wf_dir = _make_workflow(tmp_path)

    new_dir = distill.rename_workflow(wf_dir.name, "Fancy New Name!", workflows_root=workflows_root)

    assert new_dir == workflows_root / "fancy-new-name"
    assert not wf_dir.exists()
    recipe = distill.load_recipe(new_dir)
    assert recipe.name == "fancy-new-name"
    assert recipe.steps  # the pipeline is preserved across the rename


def test_rename_workflow_missing_raises(tmp_path: Path) -> None:
    """Renaming a workflow that doesn't exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        distill.rename_workflow("nope", "x", workflows_root=tmp_path / "workflows")


def test_rename_workflow_never_replaces_another(tmp_path: Path) -> None:
    """Renaming onto a taken name is refused instead of overwriting that workflow."""
    workflows_root, wf_dir = _make_workflow(tmp_path)
    other = workflows_root / "other"
    other.mkdir()
    (other / "recipe.yaml").write_text("name: other\ndescription: keep me\n")

    with pytest.raises(FileExistsError):
        distill.rename_workflow(wf_dir.name, "Other", workflows_root=workflows_root)

    assert distill.load_recipe(other).description == "keep me"
    assert wf_dir.exists()


def test_rename_workflow_to_its_own_name_keeps_it(tmp_path: Path) -> None:
    """A rename that slugifies to the current name changes nothing and does not raise."""
    workflows_root, wf_dir = _make_workflow(tmp_path)
    renamed = distill.rename_workflow(wf_dir.name, wf_dir.name, workflows_root=workflows_root)
    assert renamed == wf_dir
    assert wf_dir.exists()


def test_delete_workflow_removes_dir(tmp_path: Path) -> None:
    """delete_workflow removes the workflow's directory and returns it."""
    workflows_root, wf_dir = _make_workflow(tmp_path)
    removed = distill.delete_workflow(wf_dir.name, workflows_root=workflows_root)
    assert removed == wf_dir
    assert not wf_dir.exists()


def test_delete_workflow_missing_raises(tmp_path: Path) -> None:
    """Deleting a non-existent workflow raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        distill.delete_workflow("nope", workflows_root=tmp_path / "workflows")


def test_legacy_layout_is_folded_in_before_any_lookup(tmp_path: Path) -> None:
    """A workflow saved under the old draft/ dir is found by name after the fold."""
    root = tmp_path / "workflows"
    legacy = root / "draft" / "strip"
    legacy.mkdir(parents=True)
    (legacy / "recipe.yaml").write_text("name: strip\ndescription: d\n")
    (legacy / "SKILL.md").write_text("---\nname: strip\n---\n")

    assert distill.delete_workflow("strip", workflows_root=root) == root / "strip"
    assert not (root / "draft").exists()


# ── derived defaults for container directories ───────────────────────────────


def _pipeline(output_dirs: list[str], second_input: str | None = None) -> list[JsonDict]:
    """A skull-strip → register session whose steps write to *output_dirs*."""
    d = "/data/subj_01"
    strip_args: JsonDict = {"input_path": f"{d}/t1.nii.gz", "output_dir": output_dirs[0]}
    if second_input is not None:
        strip_args["mask_path"] = second_input
    return [
        {"role": "user", "content": "strip and register"},
        _assistant_call("c1", "medmcp-neuro_skull_strip", strip_args),
        _tool_result("c1", f"ok: True\nstructured: {{'brain_path': '{d}/t1_brain.nii.gz'}}"),
        _assistant_call(
            "c2",
            "medmcp-neuro_register_to_template",
            {"input_path": f"{d}/t1_brain.nii.gz", "output_dir": output_dirs[1]},
        ),
        _tool_result("c2", f"ok: True\nstructured: {{'registered_path': '{d}/t1_mni.nii.gz'}}"),
    ]


def _recipe_from(msgs: list[JsonDict]) -> Recipe:
    return distill.build_recipe(msgs, server_names=["medmcp-neuro"], name="n", description="d")


def _defaults(recipe: Recipe) -> dict[str, str]:
    return {i.name: i.default for i in recipe.inputs}


def test_output_dir_in_the_inputs_folder_gets_a_derived_default() -> None:
    """The folder an input already sits in should not have to be retyped.

    It stays a declared input — the session had it, and a caller may want the
    results elsewhere — but defaults to the input's folder, which also makes the
    outputs follow the file being replayed on.
    """
    recipe = _recipe_from(_pipeline(["/data/subj_01", "/data/subj_01"]))

    assert [i.name for i in recipe.inputs] == ["in_1", "in_2"]
    assert _defaults(recipe) == {"in_1": "", "in_2": "{{dir(in_1)}}"}
    # The recipe still reproduces the session: nothing was dropped or rewritten.
    assert recipe.steps[0].arguments["output_dir"] == "{{in_2}}"
    assert recipe.steps[1].arguments["input_path"] == "{{step1.brain_path}}"


def test_output_dir_elsewhere_gets_no_default() -> None:
    """A destination that is not derivable is a real decision, so it is asked for."""
    recipe = _recipe_from(_pipeline(["/data/subj_01", "/exports/run7"]))
    assert _defaults(recipe)["in_3"] == ""


def test_ambiguous_folder_gets_no_default() -> None:
    """Two inputs in one folder give no principled anchor, so none is invented.

    Picking one silently would make the destination follow whichever input
    happened to be lifted first — a wrong answer that looks like a right one.
    """
    recipe = _recipe_from(
        _pipeline(["/data/subj_01", "/data/subj_01"], second_input="/data/subj_01/mask.nii.gz")
    )
    dir_input = next(i for i in recipe.inputs if i.example == "/data/subj_01")
    assert dir_input.default == ""


def test_inputs_are_never_dropped() -> None:
    """Distillation reproduces the session; it does not decide an argument is surplus."""
    recipe = _recipe_from(_pipeline(["/data/subj_01", "/exports/run7"]))
    assert [i.example for i in recipe.inputs] == [
        "/data/subj_01/t1.nii.gz",
        "/data/subj_01",
        "/exports/run7",
    ]


def test_default_survives_a_recipe_round_trip(tmp_path: Path) -> None:
    """A default is only useful if it is still there after save/load."""
    recipe = _recipe_from(_pipeline(["/data/subj_01", "/data/subj_01"]))
    draft = tmp_path / "wf"
    draft.mkdir()
    (draft / "recipe.yaml").write_text(yaml.safe_dump(recipe.to_dict()), encoding="utf-8")
    assert _defaults(distill.load_recipe(draft))["in_2"] == "{{dir(in_1)}}"
