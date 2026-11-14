"""Plan first after the first real-model rehearsal (parallel run, 14 Nov).

Four guards, each against something the rehearsal saw:

1. ``confirm_selection`` is never covered by a Go and opens its own dialog with
   the review it would confirm (U1/U3/U7 confirmed with invented notes).
2. A plan is checked against the stacks' real tool schemas before it is shown
   (D6 planned ``select_series`` with an invented ``session_label``).
3. Calls outside a plan are counted; from the third the dialog suggests a
   revised plan and the call's result reminds the agent to propose one.
4. A truncated tool result tells the model to call again, narrower, not to parse
   the saved file with shell (D2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
import pytest

from medmcp import llmshim, plan, plannote, provenance, reviewsummary, server, settings
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool

from ._pool import fake_spec, resolver
from .test_plan import ALLOW, PLAN_TOOL, _Harness, _plan_args, _request
from .test_plan import chat as chat  # the plan-first connection fixture

JsonDict = dict[str, Any]


@pytest.fixture(autouse=True)
def _fresh_core_state(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Each test starts with no pending notes or refused plans."""
    monkeypatch.setattr(server, "_plan_notes", plan.PlanNotes())
    monkeypatch.setattr(server, "_plan_refusals", {})


# ── 1. confirm_selection ─────────────────────────────────────────────────────


def _review_record(folder: Path, *, open_rows: int = 12) -> Path:
    """A review_selection record with *open_rows* rows needing a decision; its CSV path."""
    rows: list[JsonDict] = [
        {
            "collection": ".",
            "subject_id": f"P{i:02d}",
            "session_id": "2020-01-01",
            "role": "T2",
            "status": "ok",
            "reason": "one series",
            "candidates": [{"series_number": "3", "description": "T2 AX"}],
        }
        for i in range(5)
    ]
    rows += [
        {
            "collection": ".",
            "subject_id": f"Q{i:02d}",
            "session_id": "2021-02-02",
            "role": "T2",
            "status": "needs_decision",
            "reason": "two series tie on every rule",
            "candidates": [
                {"series_number": "4", "description": "T2 TSE"},
                {"series_number": "9", "description": "T2 TSE repeat"},
            ],
        }
        for i in range(open_rows)
    ]
    record = {
        "kind": "medmcp-cohort review_selection",
        "counts": {
            "ok": 5,
            "only_derived": 0,
            "needs_decision": open_rows,
            "multiple_stations": 0,
            "missing": 1,
        },
        "rows": rows,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "t2_review.json").write_text(json.dumps(record))
    csv = folder / "t2_review.csv"
    csv.write_text("subject_id\n")
    return csv


class TestReviewSummary:
    """The dialog's view of the review, read from the sibling <name>_review.json."""

    def test_counts_and_the_first_rows_needing_a_decision(self, tmp_path: Path) -> None:
        """Counts per status, up to ten open rows of all of them, the call's note."""
        csv = _review_record(tmp_path / "sel")
        summary = reviewsummary.summarize(
            {"review_path": str(csv), "note": "yes, as shown", "overrides": [{}, {}]}, tmp_path
        )
        assert summary["error"] is None
        assert summary["path"] == str(tmp_path / "sel" / "t2_review.json")
        assert summary["counts"]["needs_decision"] == 12
        assert summary["counts"]["ok"] == 5
        assert summary["rowsTotal"] == 12
        assert len(summary["rows"]) == reviewsummary.MAX_ROWS
        first = summary["rows"][0]
        assert first["subject"] == "Q00"
        assert first["status"] == "needs_decision"
        assert first["candidates"] == ["#4 T2 TSE", "#9 T2 TSE repeat"]
        assert summary["note"] == "yes, as shown"
        assert summary["overrides"] == 2

    @pytest.mark.parametrize("name", ["t2_review.html", "t2_review.json"])
    def test_any_review_file_names_the_record(self, tmp_path: Path, name: str) -> None:
        """The HTML or JSON of the review leads to the same record."""
        _review_record(tmp_path / "sel")
        summary = reviewsummary.summarize({"review_path": str(tmp_path / "sel" / name)}, tmp_path)
        assert summary["error"] is None
        assert summary["rowsTotal"] == 12

    def test_a_missing_review_is_said(self, tmp_path: Path) -> None:
        """Unreadable: the dialog says so instead of showing nothing."""
        summary = reviewsummary.summarize(
            {"review_path": str(tmp_path / "gone_review.csv")}, tmp_path
        )
        assert "not found" in summary["error"]
        assert summary["rows"] == []

    def test_not_a_review_and_no_path_are_said(self, tmp_path: Path) -> None:
        """Another file, another record kind, or no review_path at all."""
        assert (
            "not a review_selection output"
            in reviewsummary.summarize({"review_path": str(tmp_path / "table.csv")}, tmp_path)[
                "error"
            ]
        )
        (tmp_path / "x_review.json").write_text(json.dumps({"kind": "other"}))
        assert (
            "not a review_selection record"
            in reviewsummary.summarize({"review_path": str(tmp_path / "x_review.csv")}, tmp_path)[
                "error"
            ]
        )
        assert "missing" in reviewsummary.summarize({}, tmp_path)["error"]

    def test_a_review_outside_the_workspace_is_not_read(self, tmp_path: Path) -> None:
        """The core reads only inside the workspace."""
        csv = _review_record(tmp_path / "elsewhere")
        ws = tmp_path / "ws"
        ws.mkdir()
        assert (
            "outside the workspace"
            in reviewsummary.summarize({"review_path": str(csv)}, ws)["error"]
        )


def _selection_plan(ws: Path, *tools: str) -> JsonDict:
    steps: list[JsonDict] = [
        {
            "action": f"Run {t}",
            "tool": t,
            "inputs": {"review_path": str(ws / "sel" / "t2_review.csv")},
        }
        for t in tools
    ]
    return _plan_args(ws, steps=steps, output_dir=str(ws / "sel"))


class TestConfirmSelection:
    """Never covered by a Go, never granted for the chat; the dialog shows the review."""

    def test_review_and_confirm_are_never_one_plan(self, tmp_path: Path) -> None:
        """The plan is refused with what to do instead, whatever the tool spelling."""
        args = _selection_plan(tmp_path, "review_selection", "mcp_medmcp_cohort.confirm_selection")
        with pytest.raises(plan.PlanError) as err:
            plan.parse_plan(args)
        assert "cannot be in the same plan" in str(err.value)
        plan.parse_plan(_selection_plan(tmp_path, "review_selection"))
        plan.parse_plan(_selection_plan(tmp_path, "confirm_selection"))

    async def test_a_planned_confirm_still_opens_its_dialog(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Go on a plan that lists confirm_selection does not run it: the person decides."""
        csv = _review_record(tmp_path / "sel")
        await chat.propose(1, _selection_plan(tmp_path, "confirm_selection"), "go")
        args = {"review_path": str(csv), "note": "Automatic confirmation"}
        await chat.ask(2, "medmcp-cohort_confirm_selection", args, "allow_once")
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"]["inPlan"] is True
        assert "own approval" in dialog["plan"]["reason"]
        review = dialog["review"]
        assert review["rowsTotal"] == 12
        assert review["counts"]["needs_decision"] == 12
        assert review["note"] == "Automatic confirmation"
        assert len(review["rows"]) == 10
        assert chat.answer(2) == ALLOW

    async def test_confirm_is_never_granted_for_the_chat(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """No 'for the rest of this chat' option, and an earlier grant covers nothing."""
        csv = _review_record(tmp_path / "sel")
        chat.conn._plan_first = False
        chat.conn._granted_stack_tools.add(("medmcp-cohort", "confirm_selection"))
        await chat.ask(1, "medmcp-cohort_confirm_selection", {"review_path": str(csv)})
        (dialog,) = chat.frames("permission_request")
        assert {o["optionId"] for o in dialog["options"]} == {"allow_once", "reject_once"}
        assert dialog["review"]["rowsTotal"] == 12

    async def test_an_unreadable_review_is_said_in_the_dialog(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """The dialog opens anyway, saying why there is no review to show."""
        await chat.ask(
            1, "medmcp-cohort_confirm_selection", {"review_path": str(tmp_path / "x_review.csv")}
        )
        (dialog,) = chat.frames("permission_request")
        assert "not found" in dialog["review"]["error"]

    def test_the_go_text_says_confirm_is_not_part_of_it(self, tmp_path: Path) -> None:
        """The agent is told so when its plan is approved."""
        text = plan.approval_text(plan.parse_plan(_plan_args(tmp_path)))
        assert "confirm_selection is never part of a Go" in text


# ── 2. plans checked against the stacks' tools ───────────────────────────────

_SCHEMAS: dict[str, dict[str, JsonDict]] = {
    "medmcp-cohort": {
        "select_series": {
            "type": "object",
            "properties": {"index_path": {}, "sequences": {}, "plane": {}},
        },
        "index_dataset": {"type": "object", "properties": {"dataset_dir": {}}},
        "summarize_dataset": {"type": "object", "properties": {"index_path": {}}},
    }
}


def _d6_plan(ws: Path) -> plan.Plan:
    return plan.parse_plan(
        _plan_args(
            ws,
            steps=[
                {
                    "action": "Pre-operative T1c",
                    "tool": "select_series",
                    "inputs": {"index_path": str(ws / "study"), "session_label": "pre-operative"},
                }
            ],
        )
    )


class TestCheckTools:
    """Unknown tools and parameters are refused, with the parameters that exist."""

    def test_an_invented_parameter_is_named_with_the_real_ones(self, tmp_path: Path) -> None:
        """D6: select_series has no session_label."""
        (problem,) = plan.check_tools(_d6_plan(tmp_path), _SCHEMAS, complete=True)
        assert "'session_label'" in problem
        assert "its parameters are: index_path, sequences, plane" in problem

    def test_an_unknown_tool_is_refused_when_every_stack_is_known(self, tmp_path: Path) -> None:
        """With a close match suggested; with an incomplete list, it is not judged."""
        p = plan.parse_plan(_plan_args(tmp_path, steps=[{"action": "x", "tool": "select_serie"}]))
        (problem,) = plan.check_tools(p, _SCHEMAS, complete=True)
        assert "no stack offers a tool 'select_serie'" in problem
        assert "select_series" in problem
        assert plan.check_tools(p, _SCHEMAS, complete=False) == []

    def test_valid_steps_builtins_and_open_schemas_pass(self, tmp_path: Path) -> None:
        """The shipped example plan, a shell step, and a tool taking any argument."""
        assert (
            plan.check_tools(plan.parse_plan(_plan_args(tmp_path)), _SCHEMAS, complete=True) == []
        )
        shell = plan.parse_plan(
            _plan_args(tmp_path, steps=[{"action": "List", "tool": "file_system.bash"}])
        )
        assert plan.check_tools(shell, _SCHEMAS, complete=True) == []
        open_schema = {"medmcp-cohort": {"select_series": {"additionalProperties": True}}}
        assert plan.check_tools(_d6_plan(tmp_path), open_schema, complete=True) == []

    def test_the_tool_hands_the_core_problems_back(self, tmp_path: Path) -> None:
        """A plan the core refused comes back as an error listing the problems."""
        text, is_error = plan.tool_result(_plan_args(tmp_path), ["step 1: no parameter 'x'"])
        assert is_error
        assert "no parameter 'x'" in text
        assert "could not be shown" in text


class _KnownPool:
    """Stands in for the pool: known tool lists, no backends."""

    def __init__(self, known: dict[str, dict[str, JsonDict]]) -> None:
        self.known = known

    def known_tools(self, name: str) -> dict[str, JsonDict] | None:
        return self.known.get(name)

    def output_schema(self, name: str, tool: str) -> JsonDict | None:
        return None


class TestPlanRefusedByTheCore:
    """The core does not show a plan whose steps do not fit; the tool says why."""

    @pytest.fixture
    def with_schemas(self, monkeypatch: pytest.MonkeyPatch, chat: _Harness) -> _Harness:
        """The chat with the broker up and the cohort stack's tool list known."""
        monkeypatch.setattr(server, "_broker_ready", True)
        monkeypatch.setattr(server, "_pool", _KnownPool(_SCHEMAS))
        chat.conn.servers = [{"name": "medmcp-cohort"}]
        return chat

    async def test_no_card_and_the_problems_wait_for_the_tool(
        self, with_schemas: _Harness, tmp_path: Path
    ) -> None:
        """Answered allow without a card; check_plan hands the problems out once."""
        chat = with_schemas
        args = _d6_plan(tmp_path).to_json()
        args.pop("id")
        await chat.conn._handle_permission(_request(1, PLAN_TOOL, args))
        assert chat.frames("plan_request") == []
        assert chat.answer(1) == ALLOW
        assert chat.conn._plan is None
        problems = server._check_plan_op({"key": plan.plan_key(args)})["problems"]
        assert any("session_label" in p for p in problems)
        assert server._check_plan_op({"key": plan.plan_key(args)}) == {"problems": []}

    async def test_a_fitting_plan_is_shown(self, with_schemas: _Harness, tmp_path: Path) -> None:
        """The example plan fits the schemas: card as before."""
        await with_schemas.propose(1, _plan_args(tmp_path), "go")
        assert len(with_schemas.frames("plan_request")) == 1

    async def test_without_a_broker_nothing_is_checked(
        self, with_schemas: _Harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The plan tool could not learn the problems, so the card is shown."""
        monkeypatch.setattr(server, "_broker_ready", False)
        await with_schemas.propose(1, _d6_args(tmp_path), "go")
        assert len(with_schemas.frames("plan_request")) == 1

    async def test_a_stack_without_a_known_list_leaves_tool_names_unjudged(
        self, with_schemas: _Harness, tmp_path: Path
    ) -> None:
        """A tool of a stack whose list is unknown may be real: shown."""
        with_schemas.conn.servers = [{"name": "medmcp-cohort"}, {"name": "medmcp-neuro"}]
        args = _plan_args(tmp_path, steps=[{"action": "Strip", "tool": "skull_strip"}])
        await with_schemas.propose(1, args, "go")
        assert len(with_schemas.frames("plan_request")) == 1


def _d6_args(ws: Path) -> JsonDict:
    args = _d6_plan(ws).to_json()
    args.pop("id")
    return args


async def test_the_plan_tool_asks_the_core_over_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """core_problems reaches check_plan on the broker socket in $VIBE_HOME."""
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    pool = BackendPool(resolve_spec=resolver({}))
    broker = BackendBroker(
        pool, tmp_path / "backend.sock", ops={"check_plan": server._check_plan_op}
    )
    await broker.start()
    try:
        args = _d6_args(tmp_path)
        assert await plan.core_problems(args) == []
        server._refuse_plan(plan.plan_key(args), ["step 1: bad"])
        assert await plan.core_problems(args) == ["step 1: bad"]
    finally:
        await broker.aclose()
        await pool.aclose()
    # No broker: nothing to add, the plan goes through as before.
    assert await plan.core_problems(_d6_args(tmp_path)) == []


async def test_the_pool_keeps_tool_lists_across_eviction_and_restart(tmp_path: Path) -> None:
    """known_tools answers without warming; the cache file survives a new pool."""
    cache = tmp_path / "tool_schemas.json"
    pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}), schema_cache=cache)
    try:
        assert pool.known_tools("fake") is None
        await pool.list_tools("fake")
        await pool.evict("fake")
        known = pool.known_tools("fake")
        assert known is not None
        assert "echo" in known
        assert "warmup" not in known
        assert "text" in known["echo"]["properties"]
    finally:
        await pool.aclose()
    again = BackendPool(resolve_spec=resolver({}), schema_cache=cache)
    assert again.known_tools("fake") == known
    await again.aclose()


# ── 3. calls outside the plan ────────────────────────────────────────────────


class TestOffPlanCalls:
    """Counted per plan; from the third the dialog and the result push for a revised plan."""

    async def _three_off_plan(self, chat: _Harness, tmp_path: Path) -> None:
        await chat.propose(1, _plan_args(tmp_path), "go")
        for n in (2, 3, 4):
            await chat.ask(n, "bash", {"command": f"ls {tmp_path}/study/{n}"}, "allow_once")

    async def test_the_third_dialog_suggests_a_revised_plan(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Calls one and two are flagged; the third also says to ask for a revision."""
        await self._three_off_plan(chat, tmp_path)
        notes = [d["plan"] for d in chat.frames("permission_request")]
        assert [n["offPlanCount"] for n in notes] == [1, 2, 3]
        assert [n["revise"] for n in notes] == [False, False, True]
        assert notes[0]["reason"] == "Not in the plan you approved."
        assert "revise the plan" in notes[2]["reason"]

    async def test_the_allowed_third_call_carries_the_reminder(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Only the nudged call has a note; the hook's op takes it once."""
        await self._three_off_plan(chat, tmp_path)
        assert server._plan_note_op(
            {"tool": "bash", "args": {"command": f"ls {tmp_path}/study/2"}}
        ) == {"note": None}
        took = server._plan_note_op({"tool": "bash", "args": {"command": f"ls {tmp_path}/study/4"}})
        assert "call 3 outside the plan" in took["note"]
        assert "propose_plan" in took["note"]
        assert server._plan_note_op(
            {"tool": "bash", "args": {"command": f"ls {tmp_path}/study/4"}}
        ) == {"note": None}

    async def test_a_denied_call_carries_nothing(self, chat: _Harness, tmp_path: Path) -> None:
        """The reminder rides on a result; a denied call has none."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        for n in (2, 3, 4):
            await chat.ask(n, "bash", {"command": f"ls /x{n}"}, "reject_once")
        assert server._plan_note_op({"tool": "bash", "args": {"command": "ls /x4"}}) == {
            "note": None
        }

    async def test_a_new_plan_starts_the_count_again(self, chat: _Harness, tmp_path: Path) -> None:
        """The count belongs to one plan; asking twice about one call counts once."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "bash", {"command": "ls /a"}, "allow_once")
        await chat.ask(2, "bash", {"command": "ls /a"}, "allow_once")
        await chat.propose(3, _plan_args(tmp_path), "go")
        await chat.ask(4, "bash", {"command": "ls /b"}, "allow_once")
        counts = [d["plan"]["offPlanCount"] for d in chat.frames("permission_request")]
        assert counts == [1, 1, 1]

    async def test_a_stack_call_gets_the_reminder_from_the_broker(
        self, chat: _Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In the text and in the structured result run_typescript hands the program."""
        monkeypatch.setattr(server, "_jobs", None)
        monkeypatch.setattr(server, "_pool", None)
        await chat.propose(1, _plan_args(tmp_path), "go")
        for n in (2, 3, 4):
            args = {"index_path": str(tmp_path / f"s{n}")}
            await chat.ask(n, "medmcp-cohort_select_series", args, "allow_once")
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text='{"n": 1}')],
            structuredContent={"n": 1},
        )
        args = {"index_path": str(tmp_path / "s4")}
        out = await server._after_stack_call("medmcp-cohort", "select_series", args, None, result)
        assert out.structuredContent is not None
        assert "call 3 outside the plan" in out.structuredContent["medmcp_note"]
        last = out.content[-1]
        assert isinstance(last, mcp_types.TextContent)
        assert "propose_plan" in last.text
        again = await server._after_stack_call("medmcp-cohort", "select_series", args, None, result)
        assert again == result

    def test_an_output_schema_that_forbids_extra_keys_keeps_the_structured_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The note then goes in the text only."""

        class _Strict(_KnownPool):
            def output_schema(self, name: str, tool: str) -> JsonDict | None:
                return {"additionalProperties": False}

        monkeypatch.setattr(server, "_pool", _Strict({}))
        result = mcp_types.CallToolResult(content=[], structuredContent={"n": 1})
        out = server._with_note("s", "t", result, "note")
        assert out.structuredContent == {"n": 1}
        assert len(out.content) == 1

    def test_the_hook_prints_the_note_for_a_builtin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """medmcp-plan-note asks the core for vibe's own tools only."""
        asked: list[tuple[str, JsonDict]] = []

        async def _ask(op: str, payload: JsonDict, *, timeout: float = 5.0) -> JsonDict | None:
            asked.append((op, payload))
            return {"ok": True, "note": "[MedMCP] revise"}

        monkeypatch.setattr(plan, "ask_core", _ask)
        payload: JsonDict = {"tool_name": "file_system.bash", "tool_input": {"command": "ls"}}
        assert plannote.note_for(payload) == "[MedMCP] revise"
        assert asked == [("plan_note", {"tool": "bash", "args": {"command": "ls"}})]
        assert plannote.note_for({"tool_name": "run_typescript", "tool_input": {}}) is None
        assert len(asked) == 1

    async def test_provenance_and_the_report_count_them(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Each off-plan call records its number; the report counts per plan and in all."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        for n in (2, 3):
            info: JsonDict = {"toolName": "bash", "status": "completed"}
            chat.conn._tool_calls[f"c{n}"] = info
            await chat.ask(n, "bash", {"command": f"ls /y{n}"}, "allow_once")
            provenance.record_tool_event("sid", f"c{n}", info, [])
        events = [e for e in provenance.read_run_events("sid") if e.get("tool_call_id")]
        assert [e["plan"]["off_plan_n"] for e in events] == [1, 2]
        report = provenance.render_report("sid")
        assert "**Calls outside this plan:** 2" in report
        assert "**Calls outside an approved plan, in all:** 2" in report
        assert "**not in the plan** (call 2 outside it)" in report


def test_plan_notes_expire_and_key_shell_calls_by_command() -> None:
    """A note nobody took lapses; a bash call is known by its command alone."""
    now = [0.0]
    notes = plan.PlanNotes(ttl_sec=10, clock=lambda: now[0])
    notes.add("bash", {"command": "ls"}, "n1")
    assert notes.take("bash", {"command": "ls", "timeout": 30}) == "n1"
    notes.add("bash", {"command": "ls"}, "n2")
    now[0] = 11.0
    assert notes.take("bash", {"command": "ls"}) is None


def test_the_hooks_file_registers_the_plan_note_hook(tmp_path: Path) -> None:
    """enforce_hooks keeps the path guard and the plan note, nothing else."""
    settings._ensure_pathguard_hook(tmp_path)
    text = (tmp_path / "hooks.toml").read_text()
    assert 'command = "medmcp-plan-note"' in text
    assert 'type = "post_tool"' in text


# ── 4. truncated results ─────────────────────────────────────────────────────

_RECEIPT = (
    "The tool output is large. To preserve your context window, it was saved to the file "
    "system under /logs/tool-results/run_typescript-call_1.json. The truncated output below "
    "contains only the first 4000 characters of the tool response.\n"
    '<truncated-output>{"n_subjects": 3774, ...</truncated-output>\n'
    "(Remember: The full content was saved to /logs/x.json\n"
    "Read the JSON's inferred type definition from the <output-type> tag first, then inspect "
    "the saved JSON with `jq` before searching them with `grep` instead of `read_file`.)"
)


class TestTruncatedResults:
    """The shim adds the note to a truncated result and drops the shell advice."""

    def _request(self, content: object) -> JsonDict:
        return {
            "messages": [
                {"role": "user", "content": "How many patients?"},
                {"role": "tool", "tool_call_id": "call_1", "content": content},
            ]
        }

    def test_the_note_replaces_the_jq_advice(self) -> None:
        """Model-facing text: call again narrower, no shell; no jq advice left."""
        out = llmshim.repair_request(self._request(_RECEIPT), {})
        text = out["messages"][1]["content"]
        assert text.endswith(llmshim.TRUNCATION_NOTE)
        assert "Result truncated — call the tool again with narrower arguments" in text
        assert "do not parse files with shell" in text
        assert "with `jq`" not in text
        assert '"n_subjects": 3774' in text

    def test_content_parts_and_repeated_requests(self) -> None:
        """The list form gets it too, and a history replayed again gets it once."""
        out = llmshim.repair_request(self._request([{"type": "text", "text": _RECEIPT}]), {})
        text = out["messages"][1]["content"][0]["text"]
        assert text.count(llmshim.TRUNCATION_NOTE) == 1
        again = llmshim.repair_request(out, {})
        assert again["messages"][1]["content"][0]["text"].count(llmshim.TRUNCATION_NOTE) == 1

    def test_a_whole_result_is_left_alone(self) -> None:
        """No receipt, no note; user text is never touched."""
        out = llmshim.repair_request(self._request('{"n_subjects": 91}'), {})
        assert out["messages"][1]["content"] == '{"n_subjects": 91}'
        req = self._request("x")
        req["messages"][0]["content"] = "<truncated-output>"
        assert llmshim.repair_request(req, {})["messages"][0]["content"] == "<truncated-output>"


# ── the prompt ───────────────────────────────────────────────────────────────

_SHIPPED = Path(__file__).resolve().parents[1] / ".vibe" / "prompts" / "medmcp.md"


class TestPrompt:
    """The rules the model reads, with plan first on and off."""

    def test_the_confirm_rules(self) -> None:
        """Review and confirm apart; confirm only on the user's reply, quoted as user_reply."""
        text = _SHIPPED.read_text()
        assert "`review_selection` and `confirm_selection` are never in the same plan" in text
        assert "**Confirming a selection**" in text
        assert "pass their reply word for word as `user_reply`" in text
        assert "Do not confirm in the same turn" in text

    def test_the_plan_step_and_off_plan_rules(self) -> None:
        """Real parameter names only; a [MedMCP] note means re-plan."""
        text = _SHIPPED.read_text()
        assert "never invent a parameter" in text
        assert "`[MedMCP]` note" in text

    def test_the_truncation_rule(self) -> None:
        """Narrower call, never shell on the saved result."""
        text = _SHIPPED.read_text()
        assert "**Truncated results**" in text
        assert "re-requested with narrower arguments" in text

    def test_plan_first_off_keeps_the_confirm_and_truncation_rules(self) -> None:
        """They are not plan rules: the plan-free prompt keeps them."""
        stripped = settings.without_plan_first(_SHIPPED.read_text())
        assert stripped is not None
        assert "**Confirming a selection**" in stripped
        assert "**Truncated results**" in stripped
        assert "never in the same plan" not in stripped


def test_review_summary_prefers_user_reply(tmp_path: Path) -> None:
    """Cohort records the verbatim reply as ``user_reply``; the dialog shows it."""
    from medmcp import reviewsummary

    out = reviewsummary.summarize(
        {"review_path": str(tmp_path / "missing_review.csv"), "user_reply": "ja, passt"}, tmp_path
    )
    assert out["note"] == "ja, passt"
