"""The path guard's denial reasons reach the model for stack calls (repair 7 of the shim).

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

from medmcp import brokerclient, llmshim, pathguard, server
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool

from ._pool import resolver

JsonDict = dict[str, Any]

# The run_typescript result vibe 2.26.1 records for a denied call of a stack tool.
SKIPPED = (
    'run_typescript failed: {"name":"tool_skipped","message":"Tool execution was skipped '
    'by Runtime policy.","stack":"tool_skipped: Tool execution was skipped by Runtime '
    "policy.\\n    at __deserializeError (<anonymous>:119:24)\\n    at __callTool "
    "(<anonymous>:200:40)\\n    at tools.<computed>.<computed> [as inspect] "
    '(<anonymous>:317:48)\\n    at main (eval at <anonymous>, <anonymous>:5:29)"}'
)


def _skipped(tool: str) -> str:
    return SKIPPED.replace("[as inspect]", f"[as {tool}]")


def _chat(*results: tuple[str, str]) -> JsonDict:
    """A chat ending with an assistant run_typescript call and its *results* (id, text)."""
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": "inspect the scan"},
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
        notes.add("inspect", "first")
        notes.add("convert", "other")
        notes.add("inspect", "second")
        assert notes.take("inspect") == "first"
        assert notes.take("inspect") == "second"
        assert notes.take("inspect") is None
        assert notes.take("convert") == "other"

    def test_expire_and_bounded(self) -> None:
        """A denial expires after the TTL; past the limit the oldest go; reasons are capped."""
        now = [0.0]
        notes = pathguard.DenialNotes(ttl_sec=10, limit=2, clock=lambda: now[0])
        notes.add("a", "1")
        now[0] = 11
        assert notes.take("a") is None
        for reason in ("x", "y", "z"):
            notes.add("a", reason)
        assert [notes.take("a"), notes.take("a"), notes.take("a")] == ["y", "z", None]
        notes.add("a", "r" * 10_000)
        assert len(notes.take("a") or "") == pathguard._MAX_REASON


class TestTheGuardRecords:
    """Every denial of a stack call is handed to the core; nothing else is."""

    @pytest.fixture
    def recorded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[JsonDict]:
        """The requests the guard sends the core."""
        monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
        monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))
        sent: list[JsonDict] = []

        def ask(op: str, payload: JsonDict, **_kw: object) -> JsonDict | None:
            if op == "record_denial":  # the guard may ask the core other things too
                sent.append({"op": op, **payload})
            return {"ok": True}

        monkeypatch.setattr(brokerclient, "ask_sync", ask)
        return sent

    def _run(self, monkeypatch: pytest.MonkeyPatch, invocation: JsonDict) -> JsonDict:
        """The hook as vibe runs it: invocation on stdin, decision on stdout."""
        out: list[str] = []
        monkeypatch.setattr("sys.stdin.read", lambda: json.dumps(invocation))
        monkeypatch.setattr("sys.stdout.write", out.append)
        assert pathguard.main() == 0
        return json.loads("".join(out))

    def _invocation(self, tmp_path: Path, route: str) -> JsonDict:
        return {
            "session_id": "s1",
            "cwd": str(tmp_path),
            "tool_name": route,
            "tool_input": {"input_path": str(tmp_path / "missing" / "scan.nii.gz")},
        }

    def test_a_path_denial_of_a_stack_call(
        self, tmp_path: Path, recorded: list[JsonDict], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason the hook returns is the one the core keeps."""
        out = self._run(monkeypatch, self._invocation(tmp_path, "mcp_demo_stack.inspect"))
        assert out["decision"] == "deny"
        assert recorded == [{"op": "record_denial", "tool": "inspect", "reason": out["reason"]}]

    def test_not_an_allow_nor_a_harness_tool(
        self, tmp_path: Path, recorded: list[JsonDict], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not for the harness's own tools (their reason reaches the model) or an allow."""
        out = self._run(monkeypatch, self._invocation(tmp_path, "file_system.read_file"))
        assert out["decision"] == "deny"
        inv = self._invocation(tmp_path, "mcp_demo_stack.inspect")
        inv["tool_input"] = {"threads": 4}
        assert self._run(monkeypatch, inv) == {"decision": "allow"}
        assert recorded == []

    def test_a_core_that_fails_does_not_change_the_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No core, or one that fails: the denial goes out as it was."""
        monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
        monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))
        inv = self._invocation(tmp_path, "mcp_demo_stack.inspect")
        assert self._run(monkeypatch, inv)["decision"] == "deny"

        def boom(op: str, payload: JsonDict, **_kw: object) -> JsonDict | None:
            raise RuntimeError("broker gone")

        monkeypatch.setattr(brokerclient, "ask_sync", boom)
        assert self._run(monkeypatch, inv)["decision"] == "deny"


class TestNoteDeniedCalls:
    """The shim adds the reason to the result that only says the call was skipped."""

    async def test_a_fresh_result_gets_the_reason(self) -> None:
        """The result after the last assistant message gets the reason appended."""
        payload = _chat(("c1", SKIPPED))
        core = _Core({"inspect": ["RETRY THIS CALL WITH A CORRECTED PATH. ..."]})
        await llmshim.note_denied_calls(payload, {}, core)
        content = payload["messages"][-1]["content"]
        assert content.startswith(SKIPPED)
        assert content.endswith("\n\n[MedMCP] RETRY THIS CALL WITH A CORRECTED PATH. ...")
        assert core.asked == ["inspect"]

    async def test_the_note_is_replayed_for_the_rest_of_the_chat(self) -> None:
        """History comes back as vibe recorded it; the memo adds the same note, unasked."""
        memo: dict[str, str] = {}
        await llmshim.note_denied_calls(_chat(("c1", SKIPPED)), memo, _Core({"inspect": ["r"]}))
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
        core = _Core({"inspect": ["r"]})
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
        text = _skipped("inspect") + "\n" + _skipped("convert")
        payload = _chat(("c1", text))
        payload["messages"][-1]["content"] = [{"type": "text", "text": text}]
        core = _Core({"inspect": ["a"], "convert": ["b"]})
        await llmshim.note_denied_calls(payload, {}, core)
        parts = payload["messages"][-1]["content"]
        assert parts[-1] == {"type": "text", "text": "[MedMCP] a\n\n[MedMCP] b"}
        assert core.asked == ["inspect", "convert"]


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
        "http://upstream.invalid", tool_mode="ollama", take_denial=_Core({"inspect": ["r"]})
    )
    resp = TestClient(app).post("/v1/chat/completions", json=_chat(("c1", SKIPPED)))
    assert resp.status_code == 200
    assert forwarded[0]["messages"][-1]["content"].endswith("[MedMCP] r")


def test_a_failing_core_never_fails_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """take_denial raising: the request goes upstream unchanged."""
    from starlette.testclient import TestClient

    forwarded: list[JsonDict] = []

    async def complete(
        url: str, payload: JsonDict, *args: object, **kwargs: object
    ) -> JSONResponse:
        forwarded.append(payload)
        return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

    async def boom(tool: str) -> str | None:
        raise RuntimeError("core gone")

    monkeypatch.setattr(llmshim, "_complete_with_retry", complete)
    app = llmshim.create_app("http://upstream.invalid", tool_mode="ollama", take_denial=boom)
    resp = TestClient(app).post("/v1/chat/completions", json=_chat(("c1", SKIPPED)))
    assert resp.status_code == 200
    assert forwarded[0]["messages"][-1]["content"] == SKIPPED


async def test_guard_to_shim_over_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """record_denial from the guard, take_denial from the shim, through the real socket."""
    sock = tmp_path / "backend.sock"
    monkeypatch.setenv("MEDMCP_BROKER_SOCK", str(sock))
    monkeypatch.setattr(llmshim, "_core_socket", lambda: sock)
    monkeypatch.setattr(server, "_denials", pathguard.DenialNotes())
    pool = BackendPool(resolve_spec=resolver({}))
    broker = BackendBroker(pool, sock, ops=server._core_ops)
    await broker.start()
    try:
        await brokerclient.ask("record_denial", {"tool": "inspect", "reason": "no such file"})
        payload = _chat(("c1", SKIPPED))
        await llmshim.note_denied_calls(payload, {}, llmshim.take_denial_from_core)
        assert payload["messages"][-1]["content"].endswith("[MedMCP] no such file")
        assert await llmshim.take_denial_from_core("inspect") is None
    finally:
        await broker.aclose()
        await pool.aclose()
    # No core to ask: nothing to add.
    assert await llmshim.take_denial_from_core("inspect") is None


def test_server_ops_ignore_incomplete_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record without tool or reason is dropped; take answers None, then the reason."""
    monkeypatch.setattr(server, "_denials", pathguard.DenialNotes())
    assert server._core_ops["record_denial"] is server._record_denial_op
    assert server._core_ops["take_denial"] is server._take_denial_op
    assert server._record_denial_op({"tool": "inspect"}) == {}
    assert server._take_denial_op({"tool": "inspect"}) == {"reason": None}
    server._record_denial_op({"tool": "inspect", "reason": "r"})
    assert server._take_denial_op({"tool": "inspect"}) == {"reason": "r"}
