"""Tests for the deterministic workflow replay engine."""

from __future__ import annotations

from typing import Any

import mcp.types as mcp_types
import pytest

from medmcp import replay
from medmcp.workflow import Recipe, RecipeStep, WorkflowInput

# pyright: reportPrivateUsage=false

JsonDict = dict[str, Any]


# ── placeholder resolution ────────────────────────────────────────────────────


class TestResolveValue:
    """resolve_value substitutes {{ref}} placeholders, preserving types."""

    def test_exact_match_preserves_type(self) -> None:
        """A whole-string placeholder yields the bound value with its type."""
        assert replay.resolve_value("{{in_1}}", {"in_1": "/data/x.nii"}) == "/data/x.nii"
        assert replay.resolve_value("{{n}}", {"n": 5}) == 5
        assert replay.resolve_value("{{flag}}", {"flag": True}) is True

    def test_embedded_substitutes_as_text(self) -> None:
        """A placeholder inside a larger string is replaced as text."""
        assert replay.resolve_value("--out {{step1.path}}", {"step1.path": "/o"}) == "--out /o"

    def test_unknown_ref_left_verbatim(self) -> None:
        """An unbound ref is left untouched so callers can detect it."""
        assert replay.resolve_value("{{missing}}", {}) == "{{missing}}"

    def test_recurses_into_dict_and_list(self) -> None:
        """Nested structures are resolved element-by-element."""
        out = replay.resolve_value(
            {"a": "{{in_1}}", "b": ["{{in_2}}", "lit"]}, {"in_1": "X", "in_2": "Y"}
        )
        assert out == {"a": "X", "b": ["Y", "lit"]}

    def test_non_string_passthrough(self) -> None:
        """Non-placeholder scalars pass through unchanged."""
        assert replay.resolve_value(42, {}) == 42


def test_resolve_arguments() -> None:
    """resolve_arguments resolves each value against the bindings."""
    args = {"input_path": "{{in_1}}", "device": "cuda", "n": "{{k}}"}
    assert replay.resolve_arguments(args, {"in_1": "/p", "k": 3}) == {
        "input_path": "/p",
        "device": "cuda",
        "n": 3,
    }


def test_unresolved_refs() -> None:
    """unresolved_refs reports placeholders still present anywhere in a value."""
    assert replay.unresolved_refs({"a": "{{x}}", "b": ["lit", "{{y}}"]}) == {"x", "y"}
    assert replay.unresolved_refs({"a": "done"}) == set()


# ── tool-result interpretation ────────────────────────────────────────────────


def _result(
    *, text: str = "", structured: JsonDict | None = None, is_error: bool = False
) -> mcp_types.CallToolResult:
    content: list[mcp_types.ContentBlock] = []
    if text:
        content.append(mcp_types.TextContent(type="text", text=text))
    return mcp_types.CallToolResult(content=content, structuredContent=structured, isError=is_error)


class TestExtractStructured:
    """extract_structured prefers structuredContent, falls back to text parsing."""

    def test_prefers_structured_content(self) -> None:
        """A protocol structuredContent dict is returned directly."""
        res = _result(structured={"brain_path": "/b.nii"})
        assert replay.extract_structured(res) == {"brain_path": "/b.nii"}

    def test_parses_json_text(self) -> None:
        """A JSON dict text block (FastMCP's default) is parsed."""
        res = _result(text='{"brain_path": "/b.nii", "device": "cuda"}')
        assert replay.extract_structured(res) == {"brain_path": "/b.nii", "device": "cuda"}

    def test_unwraps_result_envelope(self) -> None:
        """A lone {'result': {...}} structuredContent envelope is unwrapped."""
        res = _result(structured={"result": {"brain_path": "/b.nii"}})
        assert replay.extract_structured(res) == {"brain_path": "/b.nii"}

    def test_falls_back_to_text_blob(self) -> None:
        """A 'structured: {...}' text blob is parsed when no JSON/structuredContent."""
        res = _result(text="ok: True\nstructured: {'brain_path': '/b.nii'}")
        assert replay.extract_structured(res) == {"brain_path": "/b.nii"}

    def test_empty_when_nothing_parseable(self) -> None:
        """Plain text with no structured blob yields an empty dict."""
        assert replay.extract_structured(_result(text="done")) == {}


class TestResultFailed:
    """_result_failed flags protocol errors and known failure markers."""

    def test_protocol_error_flag(self) -> None:
        """The MCP isError flag marks a failure."""
        assert replay._result_failed(_result(is_error=True)) is True

    def test_ok_false_marker(self) -> None:
        """An 'ok: False' marker in the text marks a failure."""
        assert replay._result_failed(_result(text="ok: False")) is True

    def test_returncode_marker(self) -> None:
        """A 'returncode: 1' marker in the text marks a failure."""
        assert replay._result_failed(_result(text="returncode: 1")) is True

    def test_success(self) -> None:
        """A clean 'ok: True' result is not a failure."""
        assert replay._result_failed(_result(text="ok: True")) is False


# ── validation ────────────────────────────────────────────────────────────────

_SERVERS: list[JsonDict] = [{"name": "medmcp-neuro", "command": "neuro", "args": []}]


def _recipe(steps: list[RecipeStep], inputs: list[WorkflowInput] | None = None) -> Recipe:
    return Recipe(name="wf", description="d", inputs=inputs or [], steps=steps)


class TestValidate:
    """validate catches anything that would make a replay impossible."""

    def test_ok(self) -> None:
        """A recipe with all inputs and an installed stack validates clean."""
        recipe = _recipe(
            [RecipeStep(server="medmcp-neuro", tool="skull_strip", arguments={"p": "{{in_1}}"})],
            inputs=[WorkflowInput(name="in_1", example="/x.nii")],
        )
        assert replay.validate(recipe, {"in_1": "/y.nii"}, _SERVERS) is None

    def test_missing_input(self) -> None:
        """A declared input with no supplied value is reported."""
        recipe = _recipe(
            [RecipeStep(server="medmcp-neuro", tool="t", arguments={})],
            inputs=[WorkflowInput(name="in_1", example="/x")],
        )
        msg = replay.validate(recipe, {}, _SERVERS)
        assert msg is not None and "in_1" in msg

    def test_builtin_step_rejected(self) -> None:
        """A builtin (non-MCP) step makes the recipe non-replayable."""
        recipe = _recipe([RecipeStep(server="builtin", tool="bash", arguments={})])
        msg = replay.validate(recipe, {}, _SERVERS)
        assert msg is not None and "built-in" in msg

    def test_missing_stack(self) -> None:
        """A step needing an uninstalled stack is reported."""
        recipe = _recipe([RecipeStep(server="medmcp-cardiac", tool="t", arguments={})])
        msg = replay.validate(recipe, {}, _SERVERS)
        assert msg is not None and "medmcp-cardiac" in msg

    def test_no_steps(self) -> None:
        """A recipe with no steps has nothing to replay."""
        msg = replay.validate(_recipe([]), {}, _SERVERS)
        assert msg is not None and "no replayable steps" in msg


# ── execution / chaining ──────────────────────────────────────────────────────


def _two_step_recipe() -> Recipe:
    return _recipe(
        [
            RecipeStep(
                server="medmcp-neuro",
                tool="skull_strip",
                arguments={"input_path": "{{in_1}}", "device": "cuda"},
                produces={"brain_path": "step1.brain_path"},
            ),
            RecipeStep(
                server="medmcp-neuro",
                tool="register_to_template",
                arguments={"input_path": "{{step1.brain_path}}"},
                produces={"registered_path": "step2.registered_path"},
            ),
        ],
        inputs=[WorkflowInput(name="in_1", example="/orig.nii")],
    )


@pytest.mark.asyncio
async def test_replay_chains_outputs_to_later_steps() -> None:
    """A produced output is bound and substituted into a later step's argument."""
    calls: list[tuple[str, str, JsonDict]] = []

    async def caller(server: str, tool: str, args: JsonDict) -> tuple[bool, JsonDict, str | None]:
        calls.append((server, tool, args))
        if tool == "skull_strip":
            return True, {"brain_path": "/new_brain.nii"}, None
        return True, {"registered_path": "/new_mni.nii"}, None

    result = await replay.replay_with_caller(
        _two_step_recipe(), {"in_1": "/new.nii"}, caller=caller
    )

    assert result.ok is True
    assert calls[0][2]["input_path"] == "/new.nii"  # in_1 bound
    assert calls[1][2]["input_path"] == "/new_brain.nii"  # step1 output fed forward
    assert result.steps[1].produced == {"step2.registered_path": "/new_mni.nii"}


@pytest.mark.asyncio
async def test_replay_aborts_on_failed_step() -> None:
    """A failed step stops the run; later steps never execute."""
    calls: list[str] = []

    async def caller(server: str, tool: str, args: JsonDict) -> tuple[bool, JsonDict, str | None]:
        calls.append(tool)
        return False, {}, "boom"

    result = await replay.replay_with_caller(
        _two_step_recipe(), {"in_1": "/new.nii"}, caller=caller
    )

    assert result.ok is False
    assert calls == ["skull_strip"]  # second step never ran
    assert result.steps[0].ok is False
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_replay_aborts_on_unresolved_placeholder() -> None:
    """A step whose output never arrived leaves a later ref unresolved → abort."""

    async def caller(server: str, tool: str, args: JsonDict) -> tuple[bool, JsonDict, str | None]:
        # skull_strip succeeds but produces nothing, so {{step1.brain_path}} stays unbound.
        return True, {}, None

    result = await replay.replay_with_caller(
        _two_step_recipe(), {"in_1": "/new.nii"}, caller=caller
    )

    assert result.ok is False
    assert result.steps[1].ok is False
    assert "unresolved" in (result.steps[1].error or "")


@pytest.mark.asyncio
async def test_on_step_callback_invoked_per_step() -> None:
    """on_step fires once per executed step."""
    seen: list[int] = []

    async def caller(server: str, tool: str, args: JsonDict) -> tuple[bool, JsonDict, str | None]:
        return True, {"brain_path": "/b", "registered_path": "/r"}, None

    async def on_step(sr: replay.StepResult) -> None:
        seen.append(sr.index)

    await replay.replay_with_caller(
        _two_step_recipe(), {"in_1": "/x"}, caller=caller, on_step=on_step
    )
    assert seen == [1, 2]
