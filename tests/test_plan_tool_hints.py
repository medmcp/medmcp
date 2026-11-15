"""Plan first reads what stacks declare about their tools (`medmcp.plan.ToolHints`).

MCP tool annotations from tools/list (``readOnlyHint``, ``destructiveHint`` …) and
the ``requires_user_reply`` label/manifest field. Stack and tool names are made up.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

import mcp.types as mcp_types
import pytest

from medmcp import backend_pool, plan, provenance, server, settings

JsonDict = dict[str, Any]

PLAN_TOOL = "medmcp-plan_propose_plan"
STACK = "toy-lab"

ALLOW = {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
REJECT = {"outcome": {"outcome": "selected", "optionId": "reject_once"}}
_OPTIONS = [
    {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
    {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
]

# What the made-up stack's tools say about themselves.
ANNOTATIONS: dict[str, dict[str, bool]] = {
    "count_scans": {"readOnlyHint": True},
    "peek_remote": {"readOnlyHint": True, "openWorldHint": True},
    "wipe_folder": {"destructiveHint": True},
    "segment": {"idempotentHint": True},
    "confirm_pick": {"readOnlyHint": True},
}


@pytest.fixture(autouse=True)
def _fresh_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Fresh verdicts."""
    monkeypatch.setattr(server, "_plan_verdicts", {})


class _StubWs:
    """Stub ws."""

    def __init__(self) -> None:
        """Init  ."""
        self.sent: list[JsonDict] = []

    async def send_json(self, msg: JsonDict) -> None:
        """Send json."""
        self.sent.append(msg)


class _FakePool:
    """Only what the approval flow reads: the annotations kept per stack."""

    def __init__(self, known: dict[str, dict[str, dict[str, bool]]]) -> None:
        """Init  ."""
        self.known = known

    def known_annotations(self, name: str) -> dict[str, dict[str, bool]] | None:
        """Known annotations."""
        return self.known.get(name)


def _request(req_id: int, tool: str, raw_input: object) -> JsonDict:
    """Request."""
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


class _Chat:
    """Chat."""

    def __init__(self, conn: server._ChatConnection, responses: list[tuple[int, object]]) -> None:
        """Init  ."""
        self.conn = conn
        self.responses = responses

    def frames(self, kind: str) -> list[JsonDict]:
        """Frames."""
        return [m for m in cast("_StubWs", self.conn.ws).sent if m.get("type") == kind]

    def answer(self, req_id: int) -> object:
        """Answer."""
        return next(r for i, r in self.responses if i == req_id)

    async def go(self, req_id: int, args: JsonDict) -> JsonDict:
        """Propose *args*, answer Go, and let the plan tool take the verdict."""
        task = asyncio.create_task(self.conn._handle_permission(_request(req_id, PLAN_TOOL, args)))
        await asyncio.sleep(0.05)
        await self.conn._on_plan_decision(
            {"type": "plan_decision", "requestId": req_id, "decision": "go"}
        )
        await task
        return server._plan_verdict_op({"key": plan.plan_key(args)})

    async def ask(self, req_id: int, tool: str, raw_input: object, option: str = "") -> None:
        """Ask."""
        task = asyncio.create_task(self.conn._handle_permission(_request(req_id, tool, raw_input)))
        await asyncio.sleep(0.05)
        fut = self.conn._pending_perms.get(req_id)
        if fut is not None:
            fut.set_result(option or "reject_once")
        await task


def _make_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, plan_first: bool = True
) -> _Chat:
    """Make chat."""
    responses: list[tuple[int, object]] = []

    async def _respond(req_id: int, result: object) -> None:
        """Respond."""
        responses.append((req_id, result))

    monkeypatch.setattr(server._client, "respond", _respond)
    monkeypatch.setattr(settings, "load_provenance_enabled", lambda: False)
    monkeypatch.setattr(settings, "load_explain_enabled", lambda: False)
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(provenance, "VIBE_HOME", tmp_path / "prov")
    monkeypatch.setattr(server, "_pool", _FakePool({STACK: ANNOTATIONS}))
    monkeypatch.delenv(server.STACK_TOOL_APPROVAL_ENV_VAR, raising=False)
    (tmp_path / "study").mkdir(exist_ok=True)
    (tmp_path / "results").mkdir(exist_ok=True)
    servers: list[JsonDict] = [
        {"name": STACK, "command": "toy", "requires_user_reply": ["confirm_pick"]}
    ]
    conn = server._ChatConnection(
        cast("Any", _StubWs()), "sid", cast("Any", asyncio.Queue()), servers, plan_first=plan_first
    )
    return _Chat(conn, responses)


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Chat:
    """Chat."""
    return _make_chat(monkeypatch, tmp_path)


def _plan(ws: Path, *steps: JsonDict) -> JsonDict:
    """Plan."""
    return {
        "understanding": "Segment the study scans.",
        "steps": list(steps)
        or [
            {"action": "Segment", "tool": "segment", "inputs": {"folder": str(ws / "study")}},
        ],
        "output_dir": str(ws / "results" / "seg"),
    }


# ── the pure parts ───────────────────────────────────────────────────────────


class TestToolHints:
    """Only explicit hints count; the MCP defaults for absent ones are not applied."""

    def test_an_unannotated_tool_changes_nothing(self) -> None:
        """An unannotated tool changes nothing."""
        hints = plan.ToolHints.from_annotations(None)
        assert not hints.lookup
        assert hints.always_asks == ""

    def test_read_only_is_a_lookup(self) -> None:
        """Read only is a lookup."""
        assert plan.ToolHints.from_annotations({"readOnlyHint": True}).lookup

    def test_read_only_but_destructive_is_not(self) -> None:
        """Read only but destructive is not."""
        hints = plan.ToolHints.from_annotations({"readOnlyHint": True, "destructiveHint": True})
        assert not hints.lookup
        assert hints.always_asks == plan.ASK_DESTRUCTIVE

    def test_open_world_is_not_a_lookup(self) -> None:
        """Open world is not a lookup."""
        assert not plan.ToolHints.from_annotations(
            {"readOnlyHint": True, "openWorldHint": True}
        ).lookup

    def test_user_reply_wins(self) -> None:
        """User reply wins."""
        hints = plan.ToolHints.from_annotations({"readOnlyHint": True}, user_reply=True)
        assert not hints.lookup
        assert hints.always_asks == plan.ASK_USER_REPLY

    def test_non_boolean_hints_are_ignored(self) -> None:
        """Non boolean hints are ignored."""
        hints = plan.ToolHints.from_annotations({"readOnlyHint": "yes", "destructiveHint": 1})
        assert hints.read_only is None
        assert hints.destructive is None


class TestParseRequiresUserReply:
    """Validated like dataset_index tool names; bad entries dropped and logged."""

    def test_valid_names_are_kept(self) -> None:
        """Valid names are kept."""
        assert plan.parse_requires_user_reply("s", ["confirm_pick", "sign_off"]) == (
            "confirm_pick",
            "sign_off",
        )

    def test_absent_is_empty(self) -> None:
        """Absent is empty."""
        assert plan.parse_requires_user_reply("s", None) == ()

    def test_not_a_list_is_dropped_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Not a list is dropped and logged."""
        with caplog.at_level(logging.WARNING, logger="medmcp.plan"):
            assert plan.parse_requires_user_reply("s", "confirm_pick") == ()
        assert "not a list" in caplog.text

    def test_bad_entries_are_dropped_and_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Bad entries are dropped and logged."""
        with caplog.at_level(logging.WARNING, logger="medmcp.plan"):
            got = plan.parse_requires_user_reply("s", ["ok_tool", "rm -rf", 3, "../x", "ok_tool"])
        assert got == ("ok_tool",)
        assert "'rm -rf'" in caplog.text

    def test_too_many_are_capped(self) -> None:
        """Too many are capped."""
        names = [f"t{i}" for i in range(plan.MAX_USER_REPLY_TOOLS + 5)]
        assert len(plan.parse_requires_user_reply("s", names)) == plan.MAX_USER_REPLY_TOOLS

    def test_user_reply_tools_by_server(self) -> None:
        """User reply tools by server."""
        servers = [
            {"name": "a", "requires_user_reply": ["x"]},
            {"name": "b"},
            {"name": "c", "requires_user_reply": "bad"},
        ]
        assert plan.user_reply_tools(servers) == {"a": frozenset({"x"})}


class TestLookupFitsPlan:
    """Test lookup fits plan."""

    def test_inside_a_planned_input_fits(self, tmp_path: Path) -> None:
        """Inside a planned input fits."""
        p = plan.parse_plan(_plan(tmp_path))
        assert plan.lookup_fits_plan(p, {"folder": str(tmp_path / "study" / "sub-01")}, tmp_path)

    def test_inside_the_output_dir_fits(self, tmp_path: Path) -> None:
        """Inside the output dir fits."""
        p = plan.parse_plan(_plan(tmp_path))
        out = tmp_path / "results" / "seg" / "a.nii.gz"
        assert plan.lookup_fits_plan(p, {"path": str(out)}, tmp_path)

    def test_elsewhere_does_not(self, tmp_path: Path) -> None:
        """Elsewhere does not."""
        p = plan.parse_plan(_plan(tmp_path))
        assert not plan.lookup_fits_plan(p, {"folder": str(tmp_path / "other")}, tmp_path)

    def test_the_workspace_root_does_not(self, tmp_path: Path) -> None:
        """The workspace root does not."""
        p = plan.parse_plan(_plan(tmp_path))
        assert not plan.lookup_fits_plan(p, {"folder": str(tmp_path)}, tmp_path)

    def test_no_paths_fits(self, tmp_path: Path) -> None:
        """No paths fits."""
        p = plan.parse_plan(_plan(tmp_path))
        assert plan.lookup_fits_plan(p, {"limit": 5}, tmp_path)


def test_the_approval_text_marks_steps_that_ask_again(tmp_path: Path) -> None:
    """The approval text marks steps that ask again."""
    p = plan.parse_plan(
        _plan(
            tmp_path,
            {"action": "Wipe", "tool": "wipe_folder"},
            {"action": "Confirm", "tool": "confirm_pick"},
            {"action": "Report"},
        )
    )
    text, is_error = plan.tool_result(
        _plan(
            tmp_path,
            {"action": "Wipe", "tool": "wipe_folder"},
            {"action": "Confirm", "tool": "confirm_pick"},
            {"action": "Report"},
        ),
        {"verdict": "go", "asks": {"0": "destructive", "1": "user_reply", "2": "bogus"}},
    )
    assert not is_error
    lines = text.splitlines()
    assert "asked to approve this call again" in lines[1]
    assert "quote their reply" in lines[2]
    assert lines[3] == "3. Report"
    assert len(p.steps) == 3


# ── the server ───────────────────────────────────────────────────────────────


class TestLookups:
    """While a plan is in force, a read-only tool inside its paths needs no dialog."""

    async def test_a_lookup_inside_the_plan_runs(self, chat: _Chat, tmp_path: Path) -> None:
        """A lookup inside the plan runs."""
        await chat.go(1, _plan(tmp_path))
        await chat.ask(2, f"{STACK}_count_scans", {"folder": str(tmp_path / "study")})
        assert chat.answer(2) == ALLOW
        assert chat.frames("permission_request") == []
        # A lookup is not a step: the checklist does not tick.
        assert chat.frames("plan_progress") == []

    async def test_a_lookup_outside_the_plans_paths_asks(self, chat: _Chat, tmp_path: Path) -> None:
        """A lookup outside the plans paths asks."""
        (tmp_path / "other").mkdir()
        await chat.go(1, _plan(tmp_path))
        await chat.ask(2, f"{STACK}_count_scans", {"folder": str(tmp_path / "other")})
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"]["inPlan"] is False

    async def test_no_plan_no_lookup(self, chat: _Chat, tmp_path: Path) -> None:
        """No plan no lookup."""
        await chat.ask(1, f"{STACK}_count_scans", {"folder": str(tmp_path / "study")})
        assert len(chat.frames("permission_request")) == 1

    async def test_an_open_world_read_asks(self, chat: _Chat, tmp_path: Path) -> None:
        """An open world read asks."""
        await chat.go(1, _plan(tmp_path))
        await chat.ask(2, f"{STACK}_peek_remote", {"folder": str(tmp_path / "study")})
        assert len(chat.frames("permission_request")) == 1

    async def test_an_unannotated_tool_still_asks(self, chat: _Chat, tmp_path: Path) -> None:
        """An unannotated tool still asks."""
        await chat.go(1, _plan(tmp_path))
        await chat.ask(2, f"{STACK}_never_listed", {"folder": str(tmp_path / "study")})
        assert len(chat.frames("permission_request")) == 1

    async def test_without_the_pool_there_are_no_lookups(
        self, chat: _Chat, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the pool there are no lookups."""
        monkeypatch.setattr(server, "_pool", None)
        await chat.go(1, _plan(tmp_path))
        await chat.ask(2, f"{STACK}_count_scans", {"folder": str(tmp_path / "study")})
        assert len(chat.frames("permission_request")) == 1

    async def test_a_lookup_that_needs_a_look_asks(self, chat: _Chat, tmp_path: Path) -> None:
        """A path outside the workspace is news the plan did not show."""
        await chat.go(1, _plan(tmp_path, {"action": "x", "tool": "segment", "inputs": {}}))
        await chat.ask(2, f"{STACK}_count_scans", {"folder": "/etc"})
        assert len(chat.frames("permission_request")) == 1


class TestDestructive:
    """Test destructive."""

    async def test_a_listed_destructive_tool_still_asks(self, chat: _Chat, tmp_path: Path) -> None:
        """A listed destructive tool still asks."""
        folder = str(tmp_path / "results" / "seg")
        await chat.go(
            1,
            _plan(
                tmp_path, {"action": "Clear", "tool": "wipe_folder", "inputs": {"folder": folder}}
            ),
        )
        await chat.ask(2, f"{STACK}_wipe_folder", {"folder": folder}, "allow_once")
        (dialog,) = chat.frames("permission_request")
        assert dialog["plan"]["inPlan"] is True
        assert dialog["plan"]["reason"] == plan.ASK_REASONS[plan.ASK_DESTRUCTIVE]
        assert chat.answer(2) == ALLOW

    async def test_the_card_and_the_agent_are_told(self, chat: _Chat, tmp_path: Path) -> None:
        """The card and the agent are told."""
        verdict = await chat.go(
            1,
            _plan(
                tmp_path,
                {"action": "Segment", "tool": "segment"},
                {"action": "Clear", "tool": f"mcp_{STACK.replace('-', '_')}.wipe_folder"},
            ),
        )
        (card,) = chat.frames("plan_request")
        assert card["asks"] == {"1": "destructive"}
        assert verdict["asks"] == {"1": "destructive"}


class TestUserReply:
    """Never covered by a Go nor by "for the rest of this chat"; the dialog says why."""

    async def test_a_listed_user_reply_tool_asks(self, chat: _Chat, tmp_path: Path) -> None:
        """A listed user reply tool asks."""
        await chat.go(1, _plan(tmp_path, {"action": "Confirm", "tool": "confirm_pick"}))
        await chat.ask(2, f"{STACK}_confirm_pick", {}, "allow_once")
        (dialog,) = chat.frames("permission_request")
        assert dialog["userReply"] == plan.USER_REPLY_NOTE
        assert dialog["plan"]["reason"] == plan.ASK_REASONS[plan.ASK_USER_REPLY]
        assert chat.frames("plan_request")[0]["asks"] == {"0": "user_reply"}

    async def test_it_is_never_a_lookup(self, chat: _Chat, tmp_path: Path) -> None:
        """confirm_pick is annotated read-only, but the stack says it needs a reply."""
        await chat.go(1, _plan(tmp_path))
        await chat.ask(2, f"{STACK}_confirm_pick", {})
        assert len(chat.frames("permission_request")) == 1

    async def test_no_chat_long_option(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No chat long option."""
        chat = _make_chat(monkeypatch, tmp_path, plan_first=False)
        await chat.ask(1, f"{STACK}_confirm_pick", {}, "allow_once")
        (dialog,) = chat.frames("permission_request")
        ids = [o["optionId"] for o in dialog["options"]]
        assert server._STACK_TOOL_CHAT_OPTION not in ids
        assert dialog["plan"] is None
        assert dialog["userReply"] == plan.USER_REPLY_NOTE

    async def test_a_grant_never_covers_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A grant never covers it."""
        chat = _make_chat(monkeypatch, tmp_path, plan_first=False)
        chat.conn._granted_stack_tools.add((STACK, "confirm_pick"))
        await chat.ask(1, f"{STACK}_confirm_pick", {}, "allow_once")
        assert len(chat.frames("permission_request")) == 1

    async def test_other_tools_keep_the_chat_long_option(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Other tools keep the chat long option."""
        chat = _make_chat(monkeypatch, tmp_path, plan_first=False)
        await chat.ask(1, f"{STACK}_segment", {"folder": str(tmp_path / "study")}, "allow_once")
        (dialog,) = chat.frames("permission_request")
        assert server._STACK_TOOL_CHAT_OPTION in [o["optionId"] for o in dialog["options"]]
        assert dialog["userReply"] is None


# ── where the declarations come from ─────────────────────────────────────────


class TestPoolKeepsAnnotations:
    """Test pool keeps annotations."""

    def _tools(self) -> list[mcp_types.Tool]:
        """Tools."""
        return [
            mcp_types.Tool(
                name="count_scans",
                inputSchema={"type": "object"},
                annotations=mcp_types.ToolAnnotations(readOnlyHint=True, openWorldHint=False),
            ),
            mcp_types.Tool(name="segment", inputSchema={"type": "object"}),
            mcp_types.Tool(name=backend_pool.WARMUP_TOOL, inputSchema={"type": "object"}),
        ]

    def test_remembered_and_persisted(self, tmp_path: Path) -> None:
        """Remembered and persisted."""
        cache = tmp_path / "state" / "tool_annotations.json"
        pool = backend_pool.BackendPool(resolve_spec=lambda _n: None, annotation_cache=cache)
        assert pool.known_annotations(STACK) is None
        pool._remember(STACK, self._tools())
        expected = {"count_scans": {"readOnlyHint": True, "openWorldHint": False}, "segment": {}}
        assert pool.known_annotations(STACK) == expected
        assert json.loads(cache.read_text()) == {STACK: expected}
        again = backend_pool.BackendPool(resolve_spec=lambda _n: None, annotation_cache=cache)
        assert again.known_annotations(STACK) == expected

    def test_a_corrupt_cache_is_sanitised(self, tmp_path: Path) -> None:
        """A corrupt cache is sanitised."""
        cache = tmp_path / "tool_annotations.json"
        cache.write_text(
            json.dumps(
                {STACK: {"t": {"readOnlyHint": "yes", "destructiveHint": True, "x": 1}}, "b": 3}
            )
        )
        pool = backend_pool.BackendPool(resolve_spec=lambda _n: None, annotation_cache=cache)
        assert pool.known_annotations(STACK) == {"t": {"destructiveHint": True}}
        assert pool.known_annotations("b") is None


class TestDeclarationsAreRead:
    """Test declarations are read."""

    def test_a_manifest_carries_requires_user_reply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest carries requires user reply."""
        stacks_d = tmp_path / "stacks.d"
        stacks_d.mkdir()
        (stacks_d / f"{STACK}.toml").write_text(
            f'name = "{STACK}"\ncommand = "docker"\nargs = ["run", "--rm", "-i", "img"]\n'
            'requires_user_reply = ["confirm_pick", "no good"]\n'
        )
        monkeypatch.setattr(settings, "STACKS_D_PATH", stacks_d)
        (entry,) = settings._load_stack_manifests()
        assert entry["requires_user_reply"] == ["confirm_pick"]

    def test_the_env_var_forces_plan_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The env var forces plan first."""
        monkeypatch.setattr(settings, "PLAN_FIRST_PATH", tmp_path / "plan_first.json")
        monkeypatch.delenv(settings.PLAN_FIRST_ENV_VAR, raising=False)
        assert settings.load_plan_first_enabled() is False
        monkeypatch.setenv(settings.PLAN_FIRST_ENV_VAR, "1")
        assert settings.load_plan_first_enabled() is True
        settings.save_plan_first_enabled(True)
        monkeypatch.setenv(settings.PLAN_FIRST_ENV_VAR, "off")
        assert settings.load_plan_first_enabled() is False
        monkeypatch.setenv(settings.PLAN_FIRST_ENV_VAR, "")
        assert settings.load_plan_first_enabled() is True
