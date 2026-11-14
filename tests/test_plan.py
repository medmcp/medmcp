"""Plan first (`medmcp.plan`): the plan, Go/Change, approval scoping, provenance."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from medmcp import distill, pathguard, plan, provenance, server, settings

from .conftest import synced_config

JsonDict = dict[str, Any]

PLAN_TOOL = "medmcp-plan_propose_plan"


def _plan_args(ws: Path, **overrides: object) -> JsonDict:
    """A two-step plan over *ws*: index a study folder, then summarise it."""
    args: JsonDict = {
        "understanding": "Count the sessions per subject in the study folder.",
        "steps": [
            {
                "action": "Index the scans",
                "tool": "index_dataset",
                "inputs": {"dataset_dir": str(ws / "study")},
            },
            {"action": "Summarise sessions", "tool": "medmcp-cohort_summarize_dataset"},
            {"action": "Report the counts"},
        ],
        "output_dir": str(ws / "results" / "sessions"),
        "estimated_time": "about 1 minute",
    }
    args.update(overrides)
    return args


# ── the plan itself ──────────────────────────────────────────────────────────


class TestParsePlan:
    """A plan is validated before it is shown; every problem is reported at once."""

    def test_a_valid_plan_round_trips(self, tmp_path: Path) -> None:
        """The fields the card shows survive validation."""
        p = plan.parse_plan(_plan_args(tmp_path), plan_id="call-1")
        assert p.id == "call-1"
        assert [s.tool for s in p.steps] == ["index_dataset", "medmcp-cohort_summarize_dataset", ""]
        assert p.to_json()["output_dir"] == str(tmp_path / "results" / "sessions")
        assert p.to_json()["steps"][2] == {"action": "Report the counts"}

    def test_the_json_string_form_is_accepted(self, tmp_path: Path) -> None:
        """The JSON-string form vibe sometimes ships is accepted."""
        p = plan.parse_plan(json.dumps(_plan_args(tmp_path)))
        assert len(p.steps) == 3

    def test_a_bare_string_step_is_an_action(self) -> None:
        """A model that writes steps as sentences still gets its plan shown."""
        p = plan.parse_plan({"understanding": "Look it up.", "steps": ["Search the index"]})
        assert p.steps[0].action == "Search the index"
        assert p.steps[0].tool == ""

    def test_a_tool_alone_names_its_action(self) -> None:
        """A step with a tool and no action shows the tool."""
        p = plan.parse_plan({"understanding": "x", "steps": [{"tool": "describe_table"}]})
        assert p.steps[0].action == "describe_table"

    def test_every_problem_is_listed(self) -> None:
        """Missing understanding, empty steps and a relative output folder all come back."""
        with pytest.raises(plan.PlanError) as exc:
            plan.parse_plan({"steps": [], "output_dir": "results"})
        problems = exc.value.problems
        assert "understanding is required" in problems
        assert "steps must be a non-empty list" in problems
        assert "output_dir must be an absolute path" in problems

    def test_a_long_plan_is_refused(self) -> None:
        """A plan is something to read in seconds."""
        steps = [{"action": f"step {n}"} for n in range(plan.MAX_STEPS + 1)]
        with pytest.raises(plan.PlanError, match="too many steps"):
            plan.parse_plan({"understanding": "x", "steps": steps})

    def test_an_empty_call_is_refused(self) -> None:
        """No arguments at all is a plan that cannot be shown."""
        with pytest.raises(plan.PlanError, match="empty"):
            plan.parse_plan({})

    def test_the_tool_result_after_go_lists_the_steps(self, tmp_path: Path) -> None:
        """What the agent reads after Go: carry out exactly these, revise for anything else."""
        text, is_error = plan.tool_result(_plan_args(tmp_path))
        assert not is_error
        assert "approved this plan (Go)" in text
        assert "1. Index the scans [index_dataset]" in text
        assert "revised plan" in text

    def test_the_tool_result_for_a_bad_plan_says_what_to_fix(self) -> None:
        """The core lets a plan it cannot show through; the tool tells the agent why."""
        text, is_error = plan.tool_result({"steps": []})
        assert is_error
        assert "understanding is required" in text


class TestStepNamesTool:
    """The model spells a tool several ways; each names the same call."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "index_dataset",
            "medmcp-cohort_index_dataset",
            "mcp_medmcp_cohort.index_dataset",
            "tools.mcp_medmcp_cohort.index_dataset",
            "medmcp-cohort.index_dataset",
        ],
    )
    def test_spellings_match(self, spelling: str) -> None:
        """Bare, canonical and route spellings all match."""
        assert plan.step_names_tool(spelling, "medmcp-cohort", "index_dataset")

    def test_another_tool_or_stack_does_not(self) -> None:
        """A different tool, or the same tool of another stack, is not a match."""
        assert not plan.step_names_tool("summarize_dataset", "medmcp-cohort", "index_dataset")
        assert not plan.step_names_tool(
            "mcp_medmcp_neuro.index_dataset", "medmcp-cohort", "index_dataset"
        )
        assert not plan.step_names_tool("", "medmcp-cohort", "index_dataset")


class TestMatchCall:
    """Which calls a Go covers: the step's tool, and only the paths the plan shows."""

    @pytest.fixture
    def approved(self, tmp_path: Path) -> plan.Plan:
        """The two-step plan, approved."""
        return plan.parse_plan(_plan_args(tmp_path), plan_id="p1")

    def test_the_listed_call_matches_its_step(self, approved: plan.Plan, tmp_path: Path) -> None:
        """Same tool, same input folder, output in the plan's output folder."""
        args = {
            "dataset_dir": str(tmp_path / "study"),
            "output_dir": str(tmp_path / "results" / "sessions"),
        }
        assert plan.match_call(approved, "medmcp-cohort", "index_dataset", args, tmp_path) == 0

    def test_a_file_inside_a_listed_folder_matches(
        self, approved: plan.Plan, tmp_path: Path
    ) -> None:
        """An input inside the folder the step names is covered by it."""
        args = {"index_path": str(tmp_path / "results" / "sessions" / "index.csv")}
        step = plan.match_call(approved, "medmcp-cohort", "summarize_dataset", args, tmp_path)
        assert step == 1

    def test_another_input_folder_does_not_match(self, approved: plan.Plan, tmp_path: Path) -> None:
        """The step showed one folder; a call on another one is a different call."""
        args = {"dataset_dir": str(tmp_path / "other_study")}
        assert plan.match_call(approved, "medmcp-cohort", "index_dataset", args, tmp_path) is None

    def test_an_output_into_the_input_data_does_not_match(
        self, approved: plan.Plan, tmp_path: Path
    ) -> None:
        """A Go never lets a tool write into the data it was shown reading."""
        args = {
            "dataset_dir": str(tmp_path / "study"),
            "output_dir": str(tmp_path / "study" / "derived"),
        }
        assert plan.match_call(approved, "medmcp-cohort", "index_dataset", args, tmp_path) is None

    def test_an_unlisted_tool_does_not_match(self, approved: plan.Plan, tmp_path: Path) -> None:
        """A tool no step names is outside the plan, whatever its paths."""
        args = {"dataset_dir": str(tmp_path / "study")}
        assert plan.match_call(approved, "medmcp-cohort", "select_series", args, tmp_path) is None

    def test_a_listed_option_must_hold_its_value(self, tmp_path: Path) -> None:
        """An argument the step shows under the same name must be what it shows."""
        approved = plan.parse_plan(
            {
                "understanding": "Segment the FLAIR.",
                "steps": [
                    {
                        "action": "Segment",
                        "tool": "segment",
                        "inputs": {"input_path": str(tmp_path / "flair.nii.gz"), "model": "fast"},
                    }
                ],
                "output_dir": str(tmp_path / "out"),
            }
        )
        same = {"input_path": str(tmp_path / "flair.nii.gz"), "model": "Fast"}
        other = {"input_path": str(tmp_path / "flair.nii.gz"), "model": "accurate"}
        assert plan.match_call(approved, "medmcp-neuro", "segment", same, tmp_path) == 0
        assert plan.match_call(approved, "medmcp-neuro", "segment", other, tmp_path) is None

    def test_the_workspace_root_covers_nothing_beneath_it(self, tmp_path: Path) -> None:
        """A plan naming the whole workspace does not approve every path in it."""
        approved = plan.parse_plan(
            {
                "understanding": "x",
                "steps": [
                    {"action": "a", "tool": "describe_table", "inputs": {"path": str(tmp_path)}}
                ],
            }
        )
        args = {"path": str(tmp_path / "secret" / "table.csv")}
        assert plan.match_call(approved, "medmcp-stats", "describe_table", args, tmp_path) is None

    def test_a_builtin_never_matches_a_step(self, approved: plan.Plan, tmp_path: Path) -> None:
        """A shell call is never a stack step, whatever the step's tool says."""
        assert plan.match_call(approved, "builtin", "bash", {"command": "ls"}, tmp_path) is None


class TestMatchMkdir:
    """Creating the plan's output folder is covered by Go; nothing else in the shell is."""

    @pytest.fixture
    def approved(self, tmp_path: Path) -> plan.Plan:
        """The two-step plan, approved."""
        return plan.parse_plan(_plan_args(tmp_path), plan_id="p1")

    def test_mkdir_of_the_output_folder(self, approved: plan.Plan, tmp_path: Path) -> None:
        """``mkdir -p`` of the folder or one inside it."""
        out = tmp_path / "results" / "sessions"
        assert plan.match_mkdir(approved, f"mkdir -p {out}", tmp_path)
        assert plan.match_mkdir(approved, f"mkdir -p {out}/figures {out}/tables", tmp_path)

    @pytest.mark.parametrize(
        "command",
        [
            "mkdir -p {ws}/elsewhere",
            "mkdir {out}",
            "mkdir -p {out} && rm -rf {ws}/study",
            "mkdir -p {out}; echo hi",
            "mkdir -p $(echo {out})",
            "mkdir -p {out}/../../escape",
            "python3 -c 'import os'",
        ],
    )
    def test_anything_else_is_not(self, approved: plan.Plan, tmp_path: Path, command: str) -> None:
        """Another folder, missing -p, chaining, substitution, an escape, a script."""
        out = tmp_path / "results" / "sessions"
        assert not plan.match_mkdir(approved, command.format(ws=tmp_path, out=out), tmp_path)


# ── the server: Go, Change, scoping, flags ───────────────────────────────────


class _StubWs:
    """Collects frames a _ChatConnection would send to the browser."""

    def __init__(self) -> None:
        self.sent: list[JsonDict] = []

    async def send_json(self, msg: JsonDict) -> None:
        self.sent.append(msg)


_OPTIONS = [
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
]


def _request(req_id: int, tool: str, raw_input: object) -> JsonDict:
    return {
        "id": req_id,
        "method": "session/request_permission",
        "params": {
            "toolCall": {
                "toolCallId": f"c{req_id}",
                "title": tool,
                "toolName": tool,
                "rawInput": raw_input,
            },
            "options": _OPTIONS,
        },
    }


class _Harness:
    """A plan-first chat connection over a stub socket, with what it answered vibe."""

    def __init__(self, conn: server._ChatConnection, responses: list[tuple[int, object]]) -> None:
        self.conn = conn
        self.responses = responses

    @property
    def sent(self) -> list[JsonDict]:
        return cast("_StubWs", self.conn.ws).sent

    def frames(self, kind: str) -> list[JsonDict]:
        return [m for m in self.sent if m.get("type") == kind]

    def answer(self, req_id: int) -> object:
        return next(r for i, r in self.responses if i == req_id)

    async def propose(self, req_id: int, args: object, decision: str, text: str = "") -> None:
        """Propose a plan and answer it with *decision* as the browser would."""
        task = asyncio.create_task(self.conn._handle_permission(_request(req_id, PLAN_TOOL, args)))
        await asyncio.sleep(0.05)
        await self.conn._on_plan_decision(
            {"type": "plan_decision", "requestId": req_id, "decision": decision, "text": text}
        )
        await task

    async def ask(self, req_id: int, tool: str, raw_input: object, option: str = "") -> None:
        """Send a permission request; answer its dialog with *option* if one opens."""
        task = asyncio.create_task(self.conn._handle_permission(_request(req_id, tool, raw_input)))
        await asyncio.sleep(0.05)
        fut = self.conn._pending_perms.get(req_id)
        if fut is not None:
            fut.set_result(option or "reject_once")
        await task


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Harness:
    """A plan-first connection rooted at *tmp_path*; provenance into *tmp_path*/prov."""
    responses: list[tuple[int, object]] = []

    async def _respond(req_id: int, result: object) -> None:
        responses.append((req_id, result))

    monkeypatch.setattr(server._client, "respond", _respond)
    monkeypatch.setattr(settings, "load_provenance_enabled", lambda: True)
    monkeypatch.setattr(settings, "load_explain_enabled", lambda: False)
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(provenance, "VIBE_HOME", tmp_path / "prov")
    monkeypatch.delenv(server.STACK_TOOL_APPROVAL_ENV_VAR, raising=False)
    (tmp_path / "study").mkdir()
    (tmp_path / "results").mkdir()
    queue: asyncio.Queue[JsonDict] = asyncio.Queue()
    conn = server._ChatConnection(
        cast("Any", _StubWs()), "sid", cast("Any", queue), [], plan_first=True
    )
    return _Harness(conn, responses)


ALLOW = {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
REJECT = {"outcome": {"outcome": "selected", "optionId": "reject_once"}}


class TestPlanCard:
    """A proposed plan is a card with Go and Change, not an approval dialog."""

    async def test_the_plan_is_shown_as_a_card(self, chat: _Harness, tmp_path: Path) -> None:
        """A plan_request frame carries the validated plan; no permission dialog opens."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        (card,) = chat.frames("plan_request")
        assert card["toolCallId"] == "c1"
        assert card["plan"]["understanding"].startswith("Count the sessions")
        assert chat.frames("permission_request") == []

    async def test_go_lets_the_plan_tool_run_and_puts_the_plan_in_force(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Go answers the plan call allow_once; the plan now governs approvals."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        assert chat.answer(1) == ALLOW
        assert chat.conn._plan is not None
        assert chat.conn._plan.id == "c1"
        assert chat.frames("plan_status") == [
            {"type": "plan_status", "planId": "c1", "status": "approved"}
        ]

    async def test_change_rejects_the_plan_call(self, chat: _Harness, tmp_path: Path) -> None:
        """Change answers reject_once; no plan is in force."""
        await chat.propose(1, _plan_args(tmp_path), "change", "Use the FLAIR scans only")
        assert chat.answer(1) == REJECT
        assert chat.conn._plan is None
        assert chat.frames("plan_status")[0]["status"] == "changed"

    async def test_a_plan_that_cannot_be_shown_goes_back_to_the_agent(self, chat: _Harness) -> None:
        """No card: the call is let through and the tool tells the agent what to fix."""
        await chat.conn._handle_permission(_request(1, PLAN_TOOL, {"steps": []}))
        assert chat.answer(1) == ALLOW
        assert chat.frames("plan_request") == []
        assert chat.conn._plan is None

    async def test_a_new_plan_withdraws_the_one_in_force(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Proposing again means the agent needs another plan; the old Go lapses."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.propose(2, _plan_args(tmp_path), "change", "no")
        assert chat.conn._plan is None

    async def test_without_plan_first_the_tool_is_an_ordinary_call(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """With the switch off nothing is special about the name (it is not offered)."""
        chat.conn._plan_first = False
        await chat.ask(1, PLAN_TOOL, _plan_args(tmp_path), "reject_once")
        assert chat.frames("plan_request") == []
        assert len(chat.frames("permission_request")) == 1


class TestApprovalScope:
    """Go approves exactly the plan's calls; everything else asks, flagged."""

    async def test_a_planned_call_runs_without_a_dialog(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """The listed stack call is answered allow_once and ticks its step."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "medmcp-cohort_index_dataset", {"dataset_dir": str(tmp_path / "study")})
        assert chat.answer(2) == ALLOW
        assert chat.frames("permission_request") == []
        assert chat.frames("plan_progress") == [
            {
                "type": "plan_progress",
                "planId": "c1",
                "step": 0,
                "toolCallId": "c2",
                "status": "running",
            }
        ]

    async def test_a_planned_call_is_granted_to_the_broker(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """The plan's Go reaches the broker like a click on Allow would."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        args = {"dataset_dir": str(tmp_path / "study")}
        chat.conn._tool_calls["c2"] = {
            "toolName": "medmcp-cohort_index_dataset",
            "rawInput": args,
        }
        await chat.ask(2, "medmcp-cohort_index_dataset", args)
        assert server._approvals.consume("medmcp-cohort", "index_dataset", args)

    async def test_completion_ticks_the_step(self, chat: _Harness, tmp_path: Path) -> None:
        """The checklist moves on when the call settles."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "medmcp-cohort_index_dataset", {"dataset_dir": str(tmp_path / "study")})
        await chat.conn._send_plan_progress("c2", "completed")
        assert chat.frames("plan_progress")[-1]["status"] == "completed"

    async def test_an_off_plan_call_asks_and_is_flagged(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """A tool the plan does not list goes to the dialog, marked not in the plan."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "medmcp-cohort_select_series", {"index_path": str(tmp_path / "study")})
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"] == {
            "inPlan": False,
            "planId": "c1",
            "step": None,
            "reason": "Not in the plan you approved.",
            "offPlanCount": 1,
            "revise": False,
        }
        assert chat.answer(2) == REJECT

    async def test_a_call_with_no_plan_is_flagged_too(self, chat: _Harness, tmp_path: Path) -> None:
        """Before any Go, every call that asks says no plan was approved."""
        await chat.ask(1, "medmcp-cohort_index_dataset", {"dataset_dir": str(tmp_path / "study")})
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"]["inPlan"] is False
        assert dialog["plan"]["reason"] == "No plan was approved for this."

    async def test_the_plans_paths_bound_it(self, chat: _Harness, tmp_path: Path) -> None:
        """The listed tool on another folder is not the listed call."""
        (tmp_path / "other").mkdir()
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "medmcp-cohort_index_dataset", {"dataset_dir": str(tmp_path / "other")})
        assert len(chat.frames("permission_request")) == 1

    async def test_a_script_is_never_covered(self, chat: _Harness, tmp_path: Path) -> None:
        """Code the assistant wrote asks on its own, even when the plan names bash."""
        args = _plan_args(
            tmp_path, steps=[{"action": "Run my script", "tool": "bash"}], output_dir=None
        )
        await chat.propose(1, args, "go")
        await chat.ask(2, "bash", {"command": "python3 -c 'print(1)'"})
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"]["inPlan"] is False

    async def test_mkdir_of_the_output_folder_is_covered(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Creating the plan's output folder needs no second click."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "bash", {"command": f"mkdir -p {tmp_path / 'results' / 'sessions'}"})
        assert chat.answer(2) == ALLOW
        assert chat.frames("permission_request") == []
        # Setup, not a listed step: nothing to tick.
        assert chat.frames("plan_progress") == []

    async def test_a_planned_call_that_overwrites_still_asks(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """An overwrite is news the plan did not show; the dialog says it is in the plan."""
        out = tmp_path / "results" / "sessions"
        out.mkdir()
        (out / "index.csv").write_text("x")
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(
            2,
            "medmcp-cohort_index_dataset",
            {"dataset_dir": str(tmp_path / "study"), "output_file": str(out / "index.csv")},
            "allow_once",
        )
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"]["inPlan"] is True
        assert dialog["plan"]["step"] == 0

    async def test_a_chat_long_grant_does_not_cover_an_off_plan_call(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """While a plan is in force, a call outside it is shown even if its tool was granted."""
        folder = str(tmp_path / "study")
        await chat.ask(1, "medmcp-cohort_select_series", {"index_path": folder}, "")
        chat.conn._granted_stack_tools.add(("medmcp-cohort", "select_series"))
        await chat.propose(2, _plan_args(tmp_path), "go")
        await chat.ask(3, "medmcp-cohort_select_series", {"index_path": folder})
        assert len(chat.frames("permission_request")) == 2


# ── provenance ───────────────────────────────────────────────────────────────


class TestProvenance:
    """The plan, the answer and each call's relation to the plan are recorded."""

    async def test_plan_and_decision_are_in_the_run_log(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """plan_proposed carries the plan; plan_decision the answer and the correction."""
        await chat.propose(1, _plan_args(tmp_path), "change", "Only the baseline sessions")
        events = provenance.read_run_events("sid")
        proposed = [e for e in events if e.get("event") == provenance.PLAN_PROPOSED]
        decided = [e for e in events if e.get("event") == provenance.PLAN_DECISION]
        assert proposed[0]["plan"]["steps"][0]["tool"] == "index_dataset"
        assert decided == [
            {
                "ts": decided[0]["ts"],
                "event": "plan_decision",
                "plan_id": "c1",
                "decision": "change",
                "correction": "Only the baseline sessions",
            }
        ]

    async def test_calls_carry_their_step_or_the_off_plan_mark(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """A planned call names its step; an off-plan one says so."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        planned: JsonDict = {"toolName": "medmcp-cohort_index_dataset", "status": "completed"}
        offplan: JsonDict = {"toolName": "medmcp-cohort_select_series", "status": "completed"}
        chat.conn._tool_calls["c2"] = planned
        chat.conn._tool_calls["c3"] = offplan
        await chat.ask(2, "medmcp-cohort_index_dataset", {"dataset_dir": str(tmp_path / "study")})
        await chat.ask(
            3, "medmcp-cohort_select_series", {"index_path": str(tmp_path)}, "allow_once"
        )
        assert planned["plan"] == {"plan_id": "c1", "step": 1}
        assert offplan["plan"] == {"plan_id": "c1", "off_plan": True, "off_plan_n": 1}
        provenance.record_tool_event("sid", "c2", planned, [])
        provenance.record_tool_event("sid", "c3", offplan, [])
        by_id = {e.get("tool_call_id"): e for e in provenance.read_run_events("sid")}
        assert by_id["c2"]["plan"] == {"plan_id": "c1", "step": 1}
        assert by_id["c3"]["plan"]["off_plan"] is True

    async def test_the_report_shows_plans_and_off_plan_calls(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """The Markdown report lists each plan with its answer and marks the calls."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        provenance.record_tool_event(
            "sid",
            "c3",
            {
                "toolName": "medmcp-cohort_select_series",
                "status": "completed",
                "plan": {"plan_id": "c1", "off_plan": True},
            },
            [],
        )
        report = provenance.render_report("sid")
        assert "## Plans (1)" in report
        assert "### Plan 1 — Go" in report
        assert "1. Index the scans (`index_dataset`)" in report
        assert "**not in the plan**" in report
        assert "## Tool calls (1)" in report


# ── the rest of the core ─────────────────────────────────────────────────────


def test_the_path_guard_lets_a_plan_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan names an output folder that does not exist yet; that is not a bad path."""
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
    invocation: JsonDict = {
        "session_id": "s",
        "cwd": str(tmp_path),
        "tool_name": "mcp_medmcp_plan.propose_plan",
        "tool_input": {"understanding": "x", "steps": [], "output_dir": str(tmp_path / "a/b/c")},
    }
    assert pathguard.decide(invocation) == {"decision": "allow"}


def test_a_plan_is_not_a_workflow_step() -> None:
    """Distilling a chat leaves the plan out: replay runs the steps, not the plan."""
    messages: list[JsonDict] = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "p1", "function": {"name": PLAN_TOOL, "arguments": '{"steps": []}'}},
                {
                    "id": "c1",
                    "function": {
                        "name": "medmcp-neuro_skull_strip",
                        "arguments": '{"input_path": "/ws/t1.nii.gz", "output_dir": "/ws"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "p1", "content": "approved"},
        {"role": "tool", "tool_call_id": "c1", "content": '{"brain_path": "/ws/b.nii.gz"}'},
    ]
    recipe = distill.build_recipe(
        messages, server_names=["medmcp-neuro"], name="n", description="d"
    )
    assert [s.tool for s in recipe.steps] == ["skull_strip"]


# ── settings: the switch, the tool's entry, the prompt ───────────────────────


def _profile(vibe_home: Path) -> None:
    profile = settings.AGENT_PROFILE_PATH
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text('system_prompt_id = "medmcp"\n')
    del vibe_home


def _with_plan_section(vibe_home: Path) -> None:
    prompt = vibe_home / "prompts" / "medmcp.md"
    prompt.write_text(
        prompt.read_text() + f"\n{settings.PLAN_FIRST_HEADING}\n- Plan first.\n\n**Skills**\n- s\n"
    )


class TestSettings:
    """The switch is on by default; off removes the tool and the prompt's plan section."""

    def test_on_by_default(self) -> None:
        """No preference saved means plan-first."""
        assert settings.load_plan_first_enabled() is True

    def test_the_plan_tool_is_configured_while_on(self, vibe_home: Path) -> None:
        """The core's plan server sits beside the stacks, spawned directly (no proxy)."""
        cfg = synced_config(vibe_home, [])
        entries = {e["name"]: e for e in cfg["mcp_servers"]}
        entry = entries[plan.PLAN_SERVER]
        assert Path(entry["command"]).name == plan.PLAN_COMMAND
        assert cfg["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID

    def test_off_drops_the_tool_and_swaps_the_prompt(self, vibe_home: Path) -> None:
        """The plan-free twin carries the todo rules instead of the plan section."""
        _with_plan_section(vibe_home)
        _profile(vibe_home)
        settings.save_plan_first_enabled(False)
        cfg = synced_config(vibe_home, [])
        assert plan.PLAN_SERVER not in {e["name"] for e in cfg.get("mcp_servers", [])}
        prompt_id = f"{settings.BASE_SYSTEM_PROMPT_ID}{settings.NO_PLAN_SUFFIX}"
        assert cfg["system_prompt_id"] == prompt_id
        text = (vibe_home / "prompts" / f"{prompt_id}.md").read_text()
        assert settings.PLAN_FIRST_HEADING not in text
        assert "**Planning multi-step work**" in text
        assert "**Skills**" in text
        assert settings.ONPREM_RULE in text
        assert 'system_prompt_id = "medmcp-noplan"' in settings.AGENT_PROFILE_PATH.read_text()

    def test_off_composes_with_the_external_posture(self, vibe_home: Path) -> None:
        """External servers and plan-first off: both changes land in one prompt."""
        _with_plan_section(vibe_home)
        settings.acknowledge_external_mcp()
        settings.set_external_mcp_enabled(True)
        settings.add_external_server("pubmed", "streamable-http", "https://example.org/mcp")
        settings.save_plan_first_enabled(False)
        cfg = synced_config(vibe_home)
        prompt_id = f"{settings.EXTERNAL_SYSTEM_PROMPT_ID}{settings.NO_PLAN_SUFFIX}"
        assert cfg["system_prompt_id"] == prompt_id
        text = (vibe_home / "prompts" / f"{prompt_id}.md").read_text()
        assert settings.EXTERNAL_RULE in text
        assert settings.PLAN_FIRST_HEADING not in text

    def test_a_prompt_without_the_section_keeps_the_base(self, vibe_home: Path) -> None:
        """No plan-first section to remove: the base prompt stays rather than a lie."""
        settings.save_plan_first_enabled(False)
        assert synced_config(vibe_home, [])["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID

    def test_the_plan_entry_is_never_discovered_as_a_stack(self, vibe_home: Path) -> None:
        """The entry the sync wrote is the core's; discovery must not adopt it."""
        synced_config(vibe_home, [])
        settings.load_mcp_servers.cache_clear()
        assert plan.PLAN_SERVER not in {s["name"] for s in settings.load_mcp_servers()}

    def test_the_shipped_prompt_has_the_section(self) -> None:
        """The plan-free twin is derived from the shipped prompt; guard the anchor."""
        shipped = Path(__file__).resolve().parents[1] / ".vibe" / "prompts" / "medmcp.md"
        text = shipped.read_text()
        assert settings.PLAN_FIRST_HEADING in text
        assert "propose_plan" in text
        stripped = settings.without_plan_first(text)
        assert stripped is not None
        assert "propose_plan" not in stripped
        assert settings.TODO_SECTION in stripped


class TestSettingsApi:
    """GET reports the switch; PUT changes it and restarts the agent."""

    @pytest.fixture
    def restarts(self, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
        """Stub persistence of everything but the switch; count restarts."""
        calls: list[bool] = []

        def _noop(_value: object) -> None:
            return None

        def _names() -> set[str]:
            return set()

        def _servers() -> list[object]:
            return []

        def _save_names(_names: Iterable[str]) -> None:
            return None

        async def _restart() -> None:
            calls.append(True)

        monkeypatch.setattr(settings, "load_active_server_names", _names)
        monkeypatch.setattr(settings, "save_active_server_names", _save_names)
        monkeypatch.setattr(settings, "save_explain_enabled", _noop)
        monkeypatch.setattr(settings, "save_provenance_enabled", _noop)
        monkeypatch.setattr(settings, "sync_servers_to_vibe_config", _noop)
        monkeypatch.setattr(settings, "active_servers", _servers)
        monkeypatch.setattr(server, "_restart_vibe", _restart)
        return calls

    def test_switching_off_restarts_the_agent(
        self, restarts: list[bool], client: TestClient
    ) -> None:
        """The tool and prompt are config: a change needs a fresh agent."""
        body: JsonDict = {"explain_tools": False, "record_provenance": True, "stacks": []}
        resp = client.put("/api/settings", json={**body, "plan_first": False})
        assert resp.json()["restarted"] is True
        assert settings.load_plan_first_enabled() is False
        resp = client.put("/api/settings", json={**body, "plan_first": False})
        assert resp.json()["restarted"] is False
        # An older drawer that does not know the switch leaves it alone.
        resp = client.put("/api/settings", json=body)
        assert resp.json()["restarted"] is False
        assert settings.load_plan_first_enabled() is False
        assert restarts == [True]


async def test_the_plan_server_offers_the_tool_over_stdio(tmp_path: Path) -> None:
    """The MCP server vibe spawns lists propose_plan and answers it from the arguments."""
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-c", "from medmcp.plan import main; main()"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        assert [t.name for t in tools] == [plan.PLAN_TOOL]
        assert tools[0].inputSchema["required"] == ["understanding", "steps"]
        ok = await session.call_tool(plan.PLAN_TOOL, _plan_args(tmp_path))
        assert not ok.isError
        bad = await session.call_tool(plan.PLAN_TOOL, {"steps": []})
        assert bad.isError
