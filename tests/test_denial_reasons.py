"""The path guard's denial reasons reach the model for stack calls (repair 8 of the shim).

A stack tool is called from ``run_typescript``; under vibe 2.26.1 a ``pre_tool``
denial reaches that program -- and so the model -- only as ``tool_skipped: Tool
execution was skipped by Runtime policy.``. The guard leaves its reason with the
core (``record_denial``) and the LLM shim adds it to the next request
(``take_denial``), as ``[MedMCP] <reason>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from starlette.responses import JSONResponse

from medmcp import llmshim, pathguard, plan, server
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool

from ._pool import resolver

JsonDict = dict[str, Any]

# The run_typescript result vibe 2.26.1 records for a denied select_series call.
SKIPPED = (
    'run_typescript failed: {"name":"tool_skipped","message":"Tool execution was skipped '
    'by Runtime policy.","stack":"tool_skipped: Tool execution was skipped by Runtime '
    "policy.\\n    at __deserializeError (<anonymous>:119:24)\\n    at __callTool "
    "(<anonymous>:200:40)\\n    at tools.<computed>.<computed> [as select_series] "
    '(<anonymous>:317:48)\\n    at main (eval at <anonymous>, <anonymous>:5:29)"}'
)


def _skipped(tool: str) -> str:
    return SKIPPED.replace("[as select_series]", f"[as {tool}]")


def _chat(*results: tuple[str, str]) -> JsonDict:
    """A chat ending with an assistant run_typescript call and its *results* (id, text)."""
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": "select the T2"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            *({"role": "tool", "tool_call_id": cid, "content": text} for cid, text in results),
        ],
    }


class _Core:
    """Stands in for take_denial: reasons by tool, and the tools asked about."""

    def __init__(self, reasons: dict[str, list[str]]) -> None:
        self.reasons = reasons
        self.asked: list[str] = []

    async def __call__(self, tool: str) -> str | None:
        self.asked.append(tool)
        queue = self.reasons.get(tool) or []
        return queue.pop(0) if queue else None


class TestDenialNotes:
    """The core's store of recent denials."""

    def test_oldest_first_per_tool_and_taken_once(self) -> None:
        """Denials of a tool are taken oldest first, each once; tools do not mix."""
        notes = pathguard.DenialNotes()
        notes.add("s", "select_series", "first")
        notes.add("s", "skull_strip", "other")
        notes.add("s", "select_series", "second")
        assert notes.take("select_series") == "first"
        assert notes.take("select_series") == "second"
        assert notes.take("select_series") is None
        assert notes.take("skull_strip") == "other"

    def test_expire_and_bounded(self) -> None:
        """A denial expires after the TTL; past the limit the oldest go."""
        now = [0.0]
        notes = pathguard.DenialNotes(ttl_sec=10, limit=2, clock=lambda: now[0])
        notes.add("s", "a", "1")
        now[0] = 11
        assert notes.take("a") is None
        for reason in ("x", "y", "z"):
            notes.add("s", "a", reason)
        assert [notes.take("a"), notes.take("a"), notes.take("a")] == ["y", "z", None]


class TestTheGuardRecords:
    """Every denial of a stack call is handed to the core; nothing else is."""

    @pytest.fixture
    def recorded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[JsonDict]:
        """The requests the guard sends the core."""
        monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
        sent: list[JsonDict] = []

        async def ask(op: str, payload: JsonDict, *, timeout: float = 5.0) -> JsonDict | None:
            sent.append({"op": op, **payload})
            return {"ok": True}

        monkeypatch.setattr(plan, "ask_core", ask)
        return sent

    def _invocation(self, tmp_path: Path, route: str) -> JsonDict:
        return {
            "session_id": "s1",
            "cwd": str(tmp_path),
            "tool_name": route,
            "tool_input": {"input_path": str(tmp_path / "missing" / "t1.nii.gz")},
        }

    def test_a_path_denial_of_a_stack_call(
        self, tmp_path: Path, recorded: list[JsonDict], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason the hook returns is the one the core keeps."""
        monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))
        out = pathguard.decide(self._invocation(tmp_path, "mcp_medmcp_neuro.skull_strip"))
        assert out["decision"] == "deny"
        records = [r for r in recorded if r["op"] == "record_denial"]
        assert records == [
            {"op": "record_denial", "session": "s1", "tool": "skull_strip", "reason": out["reason"]}
        ]

    def test_not_an_allow_nor_a_builtin(
        self, tmp_path: Path, recorded: list[JsonDict], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not for vibe's own tools (called directly, the reason reaches the model) or an allow."""
        monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))
        assert pathguard.decide(self._invocation(tmp_path, "file_system.read_file"))[
            "decision"
        ] == ("deny")
        inv = self._invocation(tmp_path, "mcp_medmcp_neuro.skull_strip")
        inv["tool_input"] = {"threads": 4}
        assert pathguard.decide(inv) == {"decision": "allow"}
        assert [r for r in recorded if r["op"] == "record_denial"] == []

    def test_a_core_that_fails_does_not_change_the_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broker that fails leaves the denial as it was."""
        monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
        monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))

        async def boom(op: str, payload: JsonDict, *, timeout: float = 5.0) -> JsonDict | None:
            raise RuntimeError("broker gone")

        monkeypatch.setattr(plan, "ask_core", boom)
        out = pathguard.decide(self._invocation(tmp_path, "mcp_medmcp_neuro.skull_strip"))
        assert out["decision"] == "deny"


class TestNoteDeniedCalls:
    """The shim adds the reason to the result that only says the call was skipped."""

    async def test_a_fresh_result_gets_the_reason(self) -> None:
        """The result after the last assistant message gets the reason appended."""
        payload = _chat(("c1", SKIPPED))
        core = _Core({"select_series": ["select_series arguments are invalid: ..."]})
        await llmshim.note_denied_calls(payload, {}, core)
        content = payload["messages"][-1]["content"]
        assert content.startswith(SKIPPED)
        assert content.endswith("\n\n[MedMCP] select_series arguments are invalid: ...")
        assert core.asked == ["select_series"]

    async def test_the_note_is_replayed_for_the_rest_of_the_chat(self) -> None:
        """History comes back as vibe recorded it; the memo adds the same note, unasked."""
        memo: dict[str, str] = {}
        await llmshim.note_denied_calls(
            _chat(("c1", SKIPPED)), memo, _Core({"select_series": ["r"]})
        )
        later = _chat(("c1", SKIPPED))
        later["messages"] += [
            {"role": "assistant", "content": "retrying"},
            {"role": "tool", "tool_call_id": "c2", "content": "done"},
        ]
        core = _Core({})
        await llmshim.note_denied_calls(later, memo, core)
        assert later["messages"][2]["content"].endswith("[MedMCP] r")
        assert core.asked == []

    async def test_an_old_result_is_never_asked_about(self) -> None:
        """A skipped call from before the shim started stays as it is."""
        payload = _chat(("c1", SKIPPED))
        payload["messages"].append({"role": "assistant", "content": "hm"})
        core = _Core({"select_series": ["r"]})
        await llmshim.note_denied_calls(payload, {}, core)
        assert payload["messages"][2]["content"] == SKIPPED
        assert core.asked == []

    async def test_no_reason_known_leaves_the_result(self) -> None:
        """Nothing recorded for the tool: the result is unchanged."""
        payload = _chat(("c1", SKIPPED), ("c2", "plain result"))
        await llmshim.note_denied_calls(payload, {}, _Core({}))
        assert [m.get("content") for m in payload["messages"][2:]] == [SKIPPED, "plain result"]

    async def test_each_skipped_call_and_list_content(self) -> None:
        """Two denials in one program's output; a result given as content parts."""
        text = _skipped("select_series") + "\n" + _skipped("check_dataset")
        payload = _chat(("c1", text))
        payload["messages"][-1]["content"] = [{"type": "text", "text": text}]
        core = _Core({"select_series": ["a"], "check_dataset": ["b"]})
        await llmshim.note_denied_calls(payload, {}, core)
        parts = payload["messages"][-1]["content"]
        assert parts[-1] == {"type": "text", "text": "[MedMCP] a\n\n[MedMCP] b"}
        assert core.asked == ["select_series", "check_dataset"]


def test_the_app_sends_the_reason_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the shim's endpoint: the request forwarded to the model carries the note."""
    from starlette.testclient import TestClient

    forwarded: list[JsonDict] = []

    async def complete(
        url: str, payload: JsonDict, *args: object, **kwargs: object
    ) -> JSONResponse:
        forwarded.append(payload)
        return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llmshim, "_complete_with_retry", complete)
    app = llmshim.create_app(
        "http://upstream.invalid", tool_mode="ollama", take_denial=_Core({"select_series": ["r"]})
    )
    resp = TestClient(app).post("/v1/chat/completions", json=_chat(("c1", SKIPPED)))
    assert resp.status_code == 200
    assert forwarded[0]["messages"][-1]["content"].endswith("[MedMCP] r")


async def test_guard_to_shim_over_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_denial from the guard, take_denial from the shim, through the real socket."""
    vibe_home = tmp_path / "vibe"
    vibe_home.mkdir()
    monkeypatch.setenv("VIBE_HOME", str(vibe_home))
    monkeypatch.setattr(server, "_denials", pathguard.DenialNotes())
    pool = BackendPool(resolve_spec=resolver({}))
    broker = BackendBroker(
        pool,
        vibe_home / "backend.sock",
        ops={"record_denial": server._record_denial_op, "take_denial": server._take_denial_op},
    )
    await broker.start()
    try:
        await plan.ask_core(
            "record_denial", {"session": "s", "tool": "select_series", "reason": "bad args"}
        )
        payload = _chat(("c1", SKIPPED))
        await llmshim.note_denied_calls(payload, {}, llmshim.take_denial_from_core)
        assert payload["messages"][-1]["content"].endswith("[MedMCP] bad args")
        assert await llmshim.take_denial_from_core("select_series") is None
    finally:
        await broker.aclose()
        await pool.aclose()
    # No core to ask: nothing to add.
    assert await llmshim.take_denial_from_core("select_series") is None


def test_server_ops_ignore_incomplete_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record without tool or reason is dropped; take answers None, then the reason."""
    monkeypatch.setattr(server, "_denials", pathguard.DenialNotes())
    assert server._record_denial_op({"tool": "select_series"}) == {}
    assert server._take_denial_op({"tool": "select_series"}) == {"reason": None}
    server._record_denial_op({"tool": "select_series", "reason": "r", "session": "s"})
    assert json.dumps(server._take_denial_op({"tool": "select_series"})) == '{"reason": "r"}'
