"""Plan first (`medmcp.plan`): the plan, Go/Change, approval scoping, provenance."""

from __future__ import annotations

import asyncio
import json
import re
import time
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
                "tool": "scan_folder",
                "inputs": {"folder": str(ws / "study")},
            },
            {"action": "Summarise sessions", "tool": "medmcp-demo_summarize_folder"},
            {"action": "Report the counts"},
        ],
        "output_dir": str(ws / "results" / "sessions"),
        "estimated_time": "about 1 minute",
    }
    args.update(overrides)
    return args


@pytest.fixture(autouse=True)
def _fresh_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Each test starts with no verdict left over from another."""
    monkeypatch.setattr(server, "_plan_verdicts", {})


# ── the plan itself ──────────────────────────────────────────────────────────


class TestParsePlan:
    """A plan is validated before it is shown; every problem is reported at once."""

    def test_a_valid_plan_round_trips(self, tmp_path: Path) -> None:
        """The fields the card shows survive validation."""
        p = plan.parse_plan(_plan_args(tmp_path), plan_id="call-1")
        assert p.id == "call-1"
        assert [s.tool for s in p.steps] == ["scan_folder", "medmcp-demo_summarize_folder", ""]
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
        text, is_error = plan.tool_result(_plan_args(tmp_path), {"verdict": plan.VERDICT_GO})
        assert not is_error
        assert "approved this plan (Go)" in text
        assert "1. Index the scans [scan_folder]" in text
        assert "revised plan" in text

    def test_the_tool_result_for_a_bad_plan_says_what_to_fix(self) -> None:
        """The core lets a plan it cannot show through; the tool tells the agent why."""
        text, is_error = plan.tool_result({"steps": []}, {"verdict": plan.VERDICT_GO})
        assert is_error
        assert "understanding is required" in text


class TestStepNamesTool:
    """The model spells a tool several ways; each names the same call."""

    @pytest.mark.parametrize(
        "spelling",
        [
            "scan_folder",
            "medmcp-demo_scan_folder",
            "mcp_medmcp_demo.scan_folder",
            "tools.mcp_medmcp_demo.scan_folder",
            "medmcp-demo.scan_folder",
        ],
    )
    def test_spellings_match(self, spelling: str) -> None:
        """Bare, canonical and route spellings all match."""
        assert plan.step_names_tool(spelling, "medmcp-demo", "scan_folder")

    def test_another_tool_or_stack_does_not(self) -> None:
        """A different tool, or the same tool of another stack, is not a match."""
        assert not plan.step_names_tool("summarize_folder", "medmcp-demo", "scan_folder")
        assert not plan.step_names_tool(
            "mcp_medmcp_other.scan_folder", "medmcp-demo", "scan_folder"
        )
        assert not plan.step_names_tool("", "medmcp-demo", "scan_folder")


class TestMatchCall:
    """Which calls a Go covers: the step's tool, and only the paths the plan shows."""

    @pytest.fixture
    def approved(self, tmp_path: Path) -> plan.Plan:
        """The two-step plan, approved."""
        return plan.parse_plan(_plan_args(tmp_path), plan_id="p1")

    def test_the_listed_call_matches_its_step(self, approved: plan.Plan, tmp_path: Path) -> None:
        """Same tool, same input folder, output in the plan's output folder."""
        args = {
            "folder": str(tmp_path / "study"),
            "output_dir": str(tmp_path / "results" / "sessions"),
        }
        assert plan.match_call(approved, "medmcp-demo", "scan_folder", args, tmp_path) == 0

    def test_a_file_inside_a_listed_folder_matches(
        self, approved: plan.Plan, tmp_path: Path
    ) -> None:
        """An input inside the folder the step names is covered by it."""
        args = {"index_path": str(tmp_path / "results" / "sessions" / "index.csv")}
        step = plan.match_call(approved, "medmcp-demo", "summarize_folder", args, tmp_path)
        assert step == 1

    def test_another_input_folder_does_not_match(self, approved: plan.Plan, tmp_path: Path) -> None:
        """The step showed one folder; a call on another one is a different call."""
        args = {"folder": str(tmp_path / "other_study")}
        assert plan.match_call(approved, "medmcp-demo", "scan_folder", args, tmp_path) is None

    def test_an_output_into_the_input_data_does_not_match(
        self, approved: plan.Plan, tmp_path: Path
    ) -> None:
        """A Go never lets a tool write into the data it was shown reading."""
        args = {
            "folder": str(tmp_path / "study"),
            "output_dir": str(tmp_path / "study" / "derived"),
        }
        assert plan.match_call(approved, "medmcp-demo", "scan_folder", args, tmp_path) is None

    def test_an_unlisted_tool_does_not_match(self, approved: plan.Plan, tmp_path: Path) -> None:
        """A tool no step names is outside the plan, whatever its paths."""
        args = {"folder": str(tmp_path / "study")}
        assert plan.match_call(approved, "medmcp-demo", "pick_files", args, tmp_path) is None

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
        assert plan.match_call(approved, "medmcp-other", "segment", same, tmp_path) == 0
        assert plan.match_call(approved, "medmcp-other", "segment", other, tmp_path) is None

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
        assert plan.match_call(approved, "medmcp-other", "describe_table", args, tmp_path) is None

    def test_a_builtin_never_matches_a_step(self, approved: plan.Plan, tmp_path: Path) -> None:
        """A shell call is never a stack step, whatever the step's tool says."""
        assert plan.match_call(approved, "builtin", "bash", {"command": "ls"}, tmp_path) is None


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

    async def propose(
        self, req_id: int, args: object, decision: str, text: str = "", *, run_tool: bool = True
    ) -> JsonDict | None:
        """Propose a plan and answer it with *decision* as the browser would.

        Then, if the call was allowed and *run_tool*, the plan tool runs as vibe
        would run it: it takes the core's verdict (plan_verdict), returned here.
        """
        task = asyncio.create_task(self.conn._handle_permission(_request(req_id, PLAN_TOOL, args)))
        await asyncio.sleep(0.05)
        await self.conn._on_plan_decision(
            {"type": "plan_decision", "requestId": req_id, "decision": decision, "text": text}
        )
        await task
        if run_tool and self.answer(req_id) == ALLOW:
            return server._plan_verdict_op({"key": plan.plan_key(args)})
        return None

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
        await chat.ask(2, "medmcp-demo_scan_folder", {"folder": str(tmp_path / "study")})
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
        args = {"folder": str(tmp_path / "study")}
        chat.conn._tool_calls["c2"] = {
            "toolName": "medmcp-demo_scan_folder",
            "rawInput": args,
        }
        await chat.ask(2, "medmcp-demo_scan_folder", args)
        assert server._approvals.consume("medmcp-demo", "scan_folder", args)

    async def test_completion_ticks_the_step(self, chat: _Harness, tmp_path: Path) -> None:
        """The checklist moves on when the call settles."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "medmcp-demo_scan_folder", {"folder": str(tmp_path / "study")})
        await chat.conn._send_plan_progress("c2", "completed")
        assert chat.frames("plan_progress")[-1]["status"] == "completed"

    async def test_an_off_plan_call_asks_and_is_flagged(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """A tool the plan does not list goes to the dialog, marked not in the plan."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "medmcp-demo_pick_files", {"index_path": str(tmp_path / "study")})
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"] == {
            "inPlan": False,
            "planId": "c1",
            "step": None,
            "reason": "Not in the plan you approved.",
        }
        assert chat.answer(2) == REJECT

    async def test_a_call_with_no_plan_is_flagged_too(self, chat: _Harness, tmp_path: Path) -> None:
        """Before any Go, every call that asks says no plan was approved."""
        await chat.ask(1, "medmcp-demo_scan_folder", {"folder": str(tmp_path / "study")})
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"]["inPlan"] is False
        assert dialog["plan"]["reason"] == "No plan was approved for this."

    async def test_the_plans_paths_bound_it(self, chat: _Harness, tmp_path: Path) -> None:
        """The listed tool on another folder is not the listed call."""
        (tmp_path / "other").mkdir()
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "medmcp-demo_scan_folder", {"folder": str(tmp_path / "other")})
        assert len(chat.frames("permission_request")) == 1

    async def test_a_script_is_never_covered(self, chat: _Harness, tmp_path: Path) -> None:
        """Code the assistant wrote asks on its own, even when the plan names bash."""
        args = _plan_args(
            tmp_path, steps=[{"action": "Run my script", "tool": "bash"}], output_dir=None
        )
        await chat.propose(1, args, "go")
        await chat.ask(2, "bash", {"command": "python3 -c 'print(1)'"})
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"] is None
        assert chat.answer(2) == REJECT

    async def test_a_shell_call_is_never_covered(self, chat: _Harness, tmp_path: Path) -> None:
        """Go approves only the listed stack calls: even mkdir of the output folder asks."""
        await chat.propose(1, _plan_args(tmp_path), "go")
        await chat.ask(2, "bash", {"command": f"mkdir -p {tmp_path / 'results' / 'sessions'}"})
        assert len(chat.frames("permission_request")) == 1
        assert chat.answer(2) == REJECT
        assert chat.frames("plan_progress") == []

    async def test_a_shell_call_is_not_called_off_plan(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """A shell call or file write asks as always, without "Not in the plan".

        A plan cannot cover vibe's own tools, so the off-plan note would be on
        nearly every shell call, including a step the card itself shows.
        """
        args = _plan_args(
            tmp_path,
            steps=[{"action": "Create the output folder", "tool": "bash"}],
        )
        await chat.propose(1, args, "go")
        cached: JsonDict = {"toolName": "bash", "status": "pending"}
        chat.conn._tool_calls["c2"] = cached
        await chat.ask(2, "bash", {"command": f"mkdir -p {tmp_path / 'results'}"})
        await chat.ask(3, "write_file", {"path": str(tmp_path / "results" / "list.txt")})
        first, second = chat.frames("permission_request")
        assert first["plan"] is None
        assert second["plan"] is None
        assert "plan" not in cached

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
            "medmcp-demo_scan_folder",
            {"folder": str(tmp_path / "study"), "output_file": str(out / "index.csv")},
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
        await chat.ask(1, "medmcp-demo_pick_files", {"index_path": folder}, "")
        chat.conn._granted_stack_tools.add(("medmcp-demo", "pick_files"))
        await chat.propose(2, _plan_args(tmp_path), "go")
        await chat.ask(3, "medmcp-demo_pick_files", {"index_path": folder})
        assert len(chat.frames("permission_request")) == 2


class TestApprovedOnlyOnTheCoresGo:
    """The plan tool says "approved" only when the core recorded a Go for that card."""

    async def test_a_go_is_in_force_only_once_the_tool_took_it(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Go records the verdict; the plan governs calls once the tool has taken it."""
        args = _plan_args(tmp_path)
        await chat.propose(1, args, "go", run_tool=False)
        assert chat.answer(1) == ALLOW
        assert chat.conn._plan is None
        verdict = server._plan_verdict_op({"key": plan.plan_key(args)})
        assert verdict["verdict"] == plan.VERDICT_GO
        assert chat.conn._plan is not None
        assert chat.conn._plan.id == "c1"
        # Taken once: the same plan again has no verdict, so it is not approved.
        again = server._plan_verdict_op({"key": plan.plan_key(args)})
        assert again == {"verdict": None, "problems": []}
        assert plan.tool_result(args, again) == (plan.NOT_APPROVED_TEXT, True)

    async def test_a_go_withdrawn_by_a_new_proposal_never_comes_into_force(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """A second proposal before the tool took the first Go withdraws that Go."""
        first = _plan_args(tmp_path)
        await chat.propose(1, first, "go", run_tool=False)
        second = _plan_args(tmp_path, understanding="Something else entirely.")
        await chat.propose(2, second, "change")
        server._plan_verdict_op({"key": plan.plan_key(first)})
        assert chat.conn._plan is None

    async def test_change_leaves_no_verdict(self, chat: _Harness, tmp_path: Path) -> None:
        """A plan the person changed has nothing for the tool to take."""
        args = _plan_args(tmp_path)
        await chat.propose(1, args, "change", "Only the first folder")
        assert server._plan_verdict_op({"key": plan.plan_key(args)})["verdict"] is None

    async def test_a_plan_that_cannot_be_shown_is_refused_with_its_problems(
        self, chat: _Harness
    ) -> None:
        """No card; the tool takes a refusal with the problems, never a Go."""
        args: JsonDict = {"understanding": "x", "steps": [], "output_dir": "rel"}
        await chat.conn._handle_permission(_request(1, PLAN_TOOL, args))
        verdict = server._plan_verdict_op({"key": plan.plan_key(args)})
        assert verdict["verdict"] == plan.VERDICT_REFUSED
        text, is_error = plan.tool_result(args, verdict)
        assert is_error
        assert "approved" not in text

    async def test_a_plan_through_an_ordinary_dialog_is_not_approved(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Allowed in a plain approval dialog (no card), the plan tool still has no Go."""
        chat.conn._plan_first = False  # the card path is not taken
        args = _plan_args(tmp_path)
        await chat.ask(1, PLAN_TOOL, args, "allow_once")
        (dialog,) = chat.frames("permission_request")
        assert server._STACK_TOOL_CHAT_OPTION not in [o["optionId"] for o in dialog["options"]]
        verdict = server._plan_verdict_op({"key": plan.plan_key(args)})
        assert plan.tool_result(args, verdict) == (plan.NOT_APPROVED_TEXT, True)

    async def test_the_plan_tool_is_never_granted_for_the_chat(
        self, chat: _Harness, tmp_path: Path
    ) -> None:
        """Even a client sending the chat-long option gets no grant on the plan tool."""
        chat.conn._plan_first = False
        await chat.ask(1, PLAN_TOOL, _plan_args(tmp_path), server._STACK_TOOL_CHAT_OPTION)
        assert ("medmcp-plan", "propose_plan") not in chat.conn._granted_stack_tools
        assert chat.answer(1) != ALLOW  # not offered: treated as no decision


async def test_an_ops_only_broker_refuses_stack_calls(tmp_path: Path) -> None:
    """Without a stack pool the core's broker answers its ops and nothing else."""
    from medmcp.backend_broker import BackendBroker

    broker = BackendBroker(None, tmp_path / "b.sock", ops=server._core_ops())
    await broker.start()
    try:
        response = await broker._run_op(1, "plan_verdict", "medmcp-plan", {"key": "k"})
        assert response["verdict"] is None
        with pytest.raises(Exception, match="no stack pool"):
            await broker._run_op(2, "call_tool", "medmcp-demo", {"tool": "x", "args": {}})
    finally:
        await broker.aclose()


async def test_lifespan_starts_the_plan_broker_only_when_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No stack pool: off (the default) binds no socket; on binds one for the plan tool."""
    monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
    monkeypatch.setattr(settings, "VIBE_HOME", tmp_path)
    monkeypatch.setattr(server, "_pool", None)
    monkeypatch.setattr(server, "_broker", None)
    sock = tmp_path / "backend.sock"
    async with server._lifespan(server.app):
        assert server._broker is None
        assert not sock.exists()
    settings.save_plan_first_enabled(True)
    async with server._lifespan(server.app):
        assert server._pool is None
        assert server._broker is not None
        assert sock.exists()
    assert server._broker is None
    assert not sock.exists()


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
        assert proposed[0]["plan"]["steps"][0]["tool"] == "scan_folder"
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
        planned: JsonDict = {"toolName": "medmcp-demo_scan_folder", "status": "completed"}
        offplan: JsonDict = {"toolName": "medmcp-demo_pick_files", "status": "completed"}
        chat.conn._tool_calls["c2"] = planned
        chat.conn._tool_calls["c3"] = offplan
        await chat.ask(2, "medmcp-demo_scan_folder", {"folder": str(tmp_path / "study")})
        await chat.ask(3, "medmcp-demo_pick_files", {"index_path": str(tmp_path)}, "allow_once")
        assert planned["plan"] == {"plan_id": "c1", "step": 1}
        assert offplan["plan"] == {"plan_id": "c1", "off_plan": True}
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
                "toolName": "medmcp-demo_pick_files",
                "status": "completed",
                "plan": {"plan_id": "c1", "off_plan": True},
            },
            [],
        )
        report = provenance.render_report("sid")
        assert "## Plans (1)" in report
        assert "### Plan 1 — Go" in report
        assert "1. Index the scans (`scan_folder`)" in report
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
                        "name": "medmcp-other_process_image",
                        "arguments": '{"input_path": "/ws/t1.nii.gz", "output_dir": "/ws"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "p1", "content": "approved"},
        {"role": "tool", "tool_call_id": "c1", "content": '{"brain_path": "/ws/b.nii.gz"}'},
    ]
    recipe = distill.build_recipe(
        messages, server_names=["medmcp-other"], name="n", description="d"
    )
    assert [s.tool for s in recipe.steps] == ["process_image"]


# ── settings: the switch, the tool's entry, the prompt ───────────────────────


def _profile(vibe_home: Path) -> None:
    profile = settings.AGENT_PROFILE_PATH
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text('system_prompt_id = "medmcp"\n')
    del vibe_home


def _with_todo_section(vibe_home: Path) -> None:
    prompt = vibe_home / "prompts" / "medmcp.md"
    prompt.write_text(
        prompt.read_text() + f"\n{settings.TODO_HEADING}\n- Use todo.\n\n**Skills**\n- s\n"
    )


class TestSettings:
    """The switch is off by default and then changes nothing; on adds the tool and prompt."""

    def test_off_by_default(self) -> None:
        """No preference saved means no plan first."""
        assert settings.load_plan_first_enabled() is False

    def test_off_writes_no_plan_tool_and_keeps_the_prompt(self, vibe_home: Path) -> None:
        """Off, the config is what it was before plan first existed."""
        _profile(vibe_home)
        cfg = synced_config(vibe_home, [])
        assert plan.PLAN_SERVER not in {e["name"] for e in cfg.get("mcp_servers", [])}
        assert cfg["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID
        assert not (vibe_home / "prompts" / f"medmcp{settings.PLAN_SUFFIX}.md").exists()

    def test_on_configures_the_tool_and_swaps_the_prompt(self, vibe_home: Path) -> None:
        """On: the plan server sits beside the stacks, the prompt carries the plan rules.

        The server is spawned directly and told the broker socket; the prompt's
        todo-list section becomes the plan-first rules.
        """
        _with_todo_section(vibe_home)
        _profile(vibe_home)
        settings.save_plan_first_enabled(True)
        cfg = synced_config(vibe_home, [])
        entry = {e["name"]: e for e in cfg["mcp_servers"]}[plan.PLAN_SERVER]
        assert Path(entry["command"]).name == plan.PLAN_COMMAND
        assert entry["env"] == {plan.BROKER_SOCK_ENV: str(settings.backend_socket_path())}
        prompt_id = f"{settings.BASE_SYSTEM_PROMPT_ID}{settings.PLAN_SUFFIX}"
        assert cfg["system_prompt_id"] == prompt_id
        text = (vibe_home / "prompts" / f"{prompt_id}.md").read_text()
        assert settings.PLAN_FIRST_SECTION in text
        assert settings.TODO_HEADING not in text
        assert "**Skills**" in text
        assert settings.ONPREM_RULE in text
        assert 'system_prompt_id = "medmcp-plan"' in settings.AGENT_PROFILE_PATH.read_text()

    def test_on_composes_with_the_external_posture(self, vibe_home: Path) -> None:
        """External servers and plan first: both changes land in one prompt."""
        _with_todo_section(vibe_home)
        settings.acknowledge_external_mcp()
        settings.set_external_mcp_enabled(True)
        settings.add_external_server("pubmed", "streamable-http", "https://example.org/mcp")
        settings.save_plan_first_enabled(True)
        cfg = synced_config(vibe_home)
        prompt_id = f"{settings.EXTERNAL_SYSTEM_PROMPT_ID}{settings.PLAN_SUFFIX}"
        assert cfg["system_prompt_id"] == prompt_id
        text = (vibe_home / "prompts" / f"{prompt_id}.md").read_text()
        assert settings.EXTERNAL_RULE in text
        assert settings.PLAN_FIRST_SECTION in text

    def test_a_prompt_without_the_todo_section_keeps_the_base(self, vibe_home: Path) -> None:
        """No todo-list section to swap: the base prompt stays rather than ask for both."""
        settings.save_plan_first_enabled(True)
        assert synced_config(vibe_home, [])["system_prompt_id"] == settings.BASE_SYSTEM_PROMPT_ID

    def test_the_plan_entry_is_never_discovered_as_a_stack(self, vibe_home: Path) -> None:
        """The entry the sync wrote is the core's; discovery must not adopt it."""
        settings.save_plan_first_enabled(True)
        synced_config(vibe_home, [])
        settings.load_mcp_servers.cache_clear()
        assert plan.PLAN_SERVER not in {s["name"] for s in settings.load_mcp_servers()}

    def test_the_shipped_prompt_has_the_anchor(self) -> None:
        """The plan-first twin is derived from the shipped prompt; guard the anchor."""
        shipped = Path(__file__).resolve().parents[1] / ".vibe" / "prompts" / "medmcp.md"
        text = shipped.read_text()
        assert "propose_plan" not in text
        swapped = settings.with_plan_first(text)
        assert swapped is not None
        assert "propose_plan" in swapped
        assert settings.TODO_HEADING not in swapped

    def test_the_plan_rules_name_no_stack_tool(self) -> None:
        """The core is stack-agnostic: the rules name only the core's own tool."""
        words = set(re.findall(r"`([a-z_]+)`", settings.PLAN_FIRST_SECTION))
        assert words <= {
            "propose_plan",
            "run_typescript",
            "understanding",
            "output_dir",
            "questions",
            "todo",
        }


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

    def test_switching_on_restarts_the_agent(
        self, restarts: list[bool], client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tool and prompt are config: a change needs a fresh agent."""
        started: list[bool] = []

        async def _start() -> None:
            started.append(True)

        monkeypatch.setattr(server, "_start_plan_broker", _start)
        monkeypatch.setattr(server, "_pool", None)
        body: JsonDict = {"explain_tools": False, "record_provenance": True, "stacks": []}
        assert client.get("/api/settings").json()["plan_first"] is False
        resp = client.put("/api/settings", json={**body, "plan_first": True})
        assert resp.json()["restarted"] is True
        assert settings.load_plan_first_enabled() is True
        # Without the stack pool, switching on starts the core's own broker.
        assert started
        resp = client.put("/api/settings", json={**body, "plan_first": True})
        assert resp.json()["restarted"] is False
        # An older drawer that does not know the switch leaves it alone.
        resp = client.put("/api/settings", json=body)
        assert resp.json()["restarted"] is False
        assert settings.load_plan_first_enabled() is True
        assert restarts == [True]


async def test_the_plan_server_offers_the_tool_over_stdio(tmp_path: Path) -> None:
    """The MCP server vibe spawns lists propose_plan; with no core to ask, nothing is a Go.

    Spawned the way vibe spawns it — the MCP client's default environment, so no
    ``$VIBE_HOME`` — which is how the rehearsal's plan tool ran (U7, U9): it
    found no broker and used to answer "approved".
    """
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
        unconfirmed = await session.call_tool(plan.PLAN_TOOL, _plan_args(tmp_path))
        assert unconfirmed.isError
        text = cast("Any", unconfirmed.content[0]).text
        assert text == plan.NOT_APPROVED_TEXT
        assert "approved this plan (Go)" not in text
        bad = await session.call_tool(plan.PLAN_TOOL, {"steps": []})
        assert bad.isError


async def test_the_plan_server_says_go_only_on_the_cores_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spawned with exactly the env its config entry gives, the tool reaches the core.

    The core's broker here answers only its own ops (no stack pool). A Go the core
    recorded for this plan is "approved" once, and puts the plan in force; asked
    again, the same plan has no verdict left and is not approved.
    """
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from medmcp.backend_broker import BackendBroker

    monkeypatch.setattr(server, "_plan_verdicts", {})
    sock = tmp_path / "s.sock"
    monkeypatch.setattr(settings, "backend_socket_path", lambda: sock)
    entry = settings.plan_server_entry()
    assert entry["env"] == {plan.BROKER_SOCK_ENV: str(sock)}
    broker = BackendBroker(None, sock, ops=server._core_ops())
    await broker.start()
    taken: list[str] = []
    args = _plan_args(tmp_path)
    server._record_plan_verdict(
        plan.plan_key(args),
        server._PlanVerdict(time.monotonic(), plan.VERDICT_GO, on_take=lambda: taken.append("go")),
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from medmcp.plan import main; main()"],
        env=entry["env"],
    )
    try:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            # vibe may hand the steps as a JSON string: the same plan, the same key.
            as_string = {**args, "steps": json.dumps(args["steps"])}
            ok = await session.call_tool(plan.PLAN_TOOL, as_string)
            assert not ok.isError
            assert "approved this plan (Go)" in cast("Any", ok.content[0]).text
            assert taken == ["go"]
            again = await session.call_tool(plan.PLAN_TOOL, args)
            assert again.isError
            assert cast("Any", again.content[0]).text == plan.NOT_APPROVED_TEXT
    finally:
        await broker.aclose()
