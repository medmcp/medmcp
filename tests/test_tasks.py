"""Tasks: a long stack call answers with a handle and finishes in the background.

Covers the manager (promotion, the GPU queue, restart, delivery bookkeeping),
the broker path against the fake stack (a short call unchanged, ``get_task``
equal to the synchronous result, cancel stopping the backend), the chat side
(the one get_task approval exception, handing finished tasks to the assistant)
and the records a task leaves (provenance, distillation, replay).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import tomllib
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import jsonschema
import mcp.types as mcp_types
import pytest

from medmcp import distill, provenance, replay, server, settings, tasks
from medmcp.backend_broker import STREAM_LIMIT, BackendBroker
from medmcp.backend_pool import BackendPool
from medmcp.proxy import with_task_handle
from medmcp.workflow import Recipe, RecipeStep

from ._pool import fake_spec, resolver

JsonDict = dict[str, Any]


def _text_result(text: str) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=text)])


def _gated(started: list[str], name: str, gate: asyncio.Event) -> tasks.Work:
    """Work that notes its start, then waits for *gate*."""

    async def work() -> mcp_types.CallToolResult:
        started.append(name)
        await gate.wait()
        return _text_result(f"{name} done")

    return work


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


# ── the manager ───────────────────────────────────────────────────────────────


class TestPromotion:
    """A call becomes a task at the threshold, and only then."""

    async def test_a_short_call_returns_its_own_result(self) -> None:
        """Finished in time: the very result, no task recorded."""
        manager = tasks.TaskManager(after_sec=1.0)
        result = _text_result("quick")

        async def work() -> mcp_types.CallToolResult:
            await asyncio.sleep(0.01)
            return result

        got = await manager.run("cpu", "echo", {}, session="s", work=work)
        assert got is result
        assert manager.list() == []

    async def test_a_slow_call_is_promoted_at_the_threshold_not_before(self) -> None:
        """The handle comes back after about ``after_sec``, the call keeps running."""
        manager = tasks.TaskManager(after_sec=0.3)
        gate = asyncio.Event()
        started: list[str] = []
        t0 = time.monotonic()
        got = await manager.run(
            "cpu", "sleep", {"seconds": 9}, session="s", work=_gated(started, "a", gate)
        )
        waited = time.monotonic() - t0
        assert 0.3 <= waited < 0.6
        task_id = tasks.handle_task_id(got.structuredContent)
        assert task_id is not None
        record = manager.get(task_id)
        assert record is not None and record.status == tasks.WORKING
        gate.set()
        await _until(lambda: record.status == tasks.COMPLETED)
        assert record.result is not None
        assert tasks.result_text(mcp_types.CallToolResult.model_validate(record.result)) == (
            "a done"
        )

    async def test_an_error_before_the_threshold_propagates_as_before(self) -> None:
        """A failing call that ends in time raises exactly what the pool raised."""
        manager = tasks.TaskManager(after_sec=1.0)

        async def work() -> mcp_types.CallToolResult:
            raise RuntimeError("stack 'x' is not installed")

        with pytest.raises(RuntimeError, match="not installed"):
            await manager.run("cpu", "echo", {}, session="s", work=work)

    async def test_the_handle_tells_the_model_and_carries_the_task(self) -> None:
        """One line for the model, in the text and in the structured content."""
        record = tasks.TaskRecord(
            task_id="tsk_0123456789abcdef", stack="s", tool="t", session="c", arguments={},
            created_at=0.0,
        )  # fmt: skip
        handle = tasks.handle_result(record)
        text = cast("mcp_types.TextContent", handle.content[0]).text
        assert "tsk_0123456789abcdef" in text
        assert "in the background" in text and "end your reply" in text and "plan" in text
        assert "cancel_task" in text
        assert "new message" in text
        assert handle.structuredContent == {
            "medmcp_task": {
                "taskId": "tsk_0123456789abcdef",
                "status": "working",
                "pollIntervalMs": tasks.POLL_INTERVAL_MS,
                "message": text,
            }
        }
        assert not handle.isError
        # The relay recognises the handle from either rendering of it.
        assert tasks.handle_task_id(handle.structuredContent) == record.task_id
        assert tasks.handle_task_id(json.dumps(handle.structuredContent)) == record.task_id
        assert tasks.handle_task_id('{"slept": 1}') is None

    def test_the_widened_output_schema_accepts_the_handle_and_the_real_output(self) -> None:
        """The proxy's widened schema admits both shapes (vibe validates structured content)."""
        tool = mcp_types.Tool(
            name="sleep",
            inputSchema={"type": "object"},
            outputSchema={
                "type": "object",
                "properties": {"slept": {"type": "number"}},
                "required": ["slept"],
            },
        )
        widened = with_task_handle(tool).outputSchema
        assert widened is not None
        jsonschema.validate({"slept": 2.0}, widened)
        record = tasks.TaskRecord(
            task_id="tsk_0123456789abcdef", stack="s", tool="t", session="c", arguments={},
            created_at=0.0,
        )  # fmt: skip
        jsonschema.validate(tasks.handle_result(record).structuredContent, widened)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"other": 1}, widened)
        plain = mcp_types.Tool(name="x", inputSchema={"type": "object"})
        assert with_task_handle(plain) is plain

    def test_a_schema_with_local_definitions_still_validates_its_output(self) -> None:
        """``#/$defs/…`` references keep resolving once the schema is wrapped."""
        schema: JsonDict = {
            "$defs": {
                "Match": {
                    "type": "object",
                    "properties": {"structure": {"type": "string"}},
                    "required": ["structure"],
                }
            },
            "type": "object",
            "properties": {"matches": {"type": "array", "items": {"$ref": "#/$defs/Match"}}},
            "required": ["matches"],
        }
        tool = mcp_types.Tool(name="find", inputSchema={"type": "object"}, outputSchema=schema)
        widened = with_task_handle(tool).outputSchema
        assert widened is not None
        jsonschema.validate({"matches": [{"structure": "liver"}]}, widened)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"matches": [{"other": 1}]}, widened)

    def test_a_task_tool_route_is_not_mistaken_for_a_handle(self) -> None:
        """The route ``mcp_medmcp_tasks.get_task`` contains ``medmcp_task`` as a word."""
        text = (
            "Task tsk_0123456789abcdef (s/t) is still working: Running, 3 s so far.\n"
            "mcp_medmcp_tasks.get_task"
        )
        assert tasks.handle_task_id(text) is None


class TestGpuQueue:
    """One GPU call at a time, the rest wait in order; CPU calls never wait."""

    async def test_gpu_calls_queue_in_order_with_how_many_are_ahead(self) -> None:
        """Later GPU calls wait, first come first served, each told how many are ahead."""
        manager = tasks.TaskManager(after_sec=0.05, is_gpu=lambda stack: stack == "gpu")
        gates = {name: asyncio.Event() for name in "abc"}
        started: list[str] = []
        handles: dict[str, mcp_types.CallToolResult] = {}
        for name in "abc":
            handles[name] = await manager.run(
                "gpu", "seg", {"n": name}, session="s", work=_gated(started, name, gates[name])
            )
        ids = {n: tasks.handle_task_id(h.structuredContent) for n, h in handles.items()}
        records = {n: manager.get(str(i)) for n, i in ids.items()}
        assert all(r is not None for r in records.values())
        a, b, c = (cast("tasks.TaskRecord", records[n]) for n in "abc")
        assert started == ["a"]
        assert a.status_message == "Running"
        assert b.status_message == "Waiting for the GPU (1 ahead)"
        assert c.status_message == "Waiting for the GPU (2 ahead)"
        gates["a"].set()
        await _until(lambda: started == ["a", "b"])
        assert b.status_message == "Running"
        assert c.status_message == "Waiting for the GPU (1 ahead)"
        gates["b"].set()
        gates["c"].set()
        await _until(lambda: c.status == tasks.COMPLETED)
        assert started == ["a", "b", "c"]

    async def test_cpu_calls_run_side_by_side(self) -> None:
        """Calls on a CPU stack start at once, whatever else runs."""
        manager = tasks.TaskManager(after_sec=0.05, is_gpu=lambda stack: stack == "gpu")
        gate = asyncio.Event()
        started: list[str] = []
        for name in "ab":
            await manager.run(
                "cpu", "x", {"n": name}, session="s", work=_gated(started, name, gate)
            )
        assert sorted(started) == ["a", "b"]
        assert all(r.status_message == "Running" for r in manager.list())
        gate.set()

    async def test_a_queued_task_cancelled_leaves_the_queue(self) -> None:
        """Stopping a task still waiting for the GPU means it never starts."""
        manager = tasks.TaskManager(after_sec=0.05, is_gpu=lambda _stack: True)
        gates = {name: asyncio.Event() for name in "ab"}
        started: list[str] = []
        ha = await manager.run("gpu", "x", {}, session="s", work=_gated(started, "a", gates["a"]))
        hb = await manager.run("gpu", "x", {}, session="s", work=_gated(started, "b", gates["b"]))
        b_id = str(tasks.handle_task_id(hb.structuredContent))
        record = await manager.cancel(b_id)
        assert record is not None and record.status == tasks.CANCELLED
        gates["a"].set()
        a_record = manager.get(str(tasks.handle_task_id(ha.structuredContent)))
        assert a_record is not None
        await _until(lambda: a_record.status == tasks.COMPLETED)
        assert started == ["a"]


class TestRecords:
    """Restart, TTL and delivery bookkeeping."""

    async def test_a_restart_marks_unfinished_tasks_failed_and_keeps_finished_ones(
        self, tmp_path: Path
    ) -> None:
        """Unfinished: failed, MedMCP restarted, chat still to be told; finished: kept."""
        state = tmp_path / "tasks.json"
        first = tasks.TaskManager(after_sec=0.05, state_path=state)
        gate = asyncio.Event()
        quick = asyncio.Event()
        started: list[str] = []
        h1 = await first.run("cpu", "long", {}, session="chat", work=_gated(started, "a", gate))
        h2 = await first.run("cpu", "done", {}, session="chat", work=_gated(started, "b", quick))
        quick.set()
        done_id = str(tasks.handle_task_id(h2.structuredContent))
        await _until(lambda: (r := first.get(done_id)) is not None and r.finished)
        await first.aclose()

        second = tasks.TaskManager(after_sec=0.05, state_path=state)
        restarted = second.load()
        long_id = str(tasks.handle_task_id(h1.structuredContent))
        assert [r.task_id for r in restarted] == [long_id]
        record = second.get(long_id)
        assert record is not None
        assert record.status == tasks.FAILED
        assert record.status_message == "MedMCP restarted"
        assert not record.delivered
        assert [r.task_id for r in second.undelivered("chat")] == [done_id, long_id]
        kept = second.get(done_id)
        assert kept is not None and kept.status == tasks.COMPLETED and kept.result is not None
        # What get_task gives after the restart is still the original result.
        assert tasks.result_text(second.tool_result(done_id)) == "b done"

    async def test_finished_tasks_expire_after_their_ttl(self) -> None:
        """A finished task is kept for its TTL, then forgotten."""
        now = [1000.0]
        manager = tasks.TaskManager(after_sec=0.05, ttl_ms=60_000, clock=lambda: now[0])
        gate = asyncio.Event()
        gate.set()
        started: list[str] = []

        async def slow() -> mcp_types.CallToolResult:
            await asyncio.sleep(0.1)
            return await _gated(started, "a", gate)()

        handle = await manager.run("cpu", "x", {}, session="s", work=slow)
        task_id = str(tasks.handle_task_id(handle.structuredContent))
        await _until(lambda: (r := manager.get(task_id)) is not None and r.finished)
        now[0] += 59
        assert manager.get(task_id) is not None
        now[0] += 2
        assert manager.get(task_id) is None

    def test_a_task_is_claimed_for_delivery_once(self) -> None:
        """A finished task is handed to its own chat once, never to another."""
        manager = tasks.TaskManager(after_sec=1.0)
        record = _finished_record(manager, "chat")
        assert [r.task_id for r in manager.claim_delivery("chat", [record.task_id])] == [
            record.task_id
        ]
        assert manager.claim_delivery("chat", [record.task_id]) == []
        assert manager.claim_delivery("other", [record.task_id]) == []
        assert manager.undelivered("chat") == []

    def test_the_started_and_finished_notices_name_the_same_call(self) -> None:
        """The chat's two notices for a task read as a pair."""
        record = _finished_record(tasks.TaskManager(after_sec=1.0), "chat")
        call = f"{record.stack.removeprefix('medmcp-')} · {record.tool}"
        assert tasks.started_text(record) == f"Task {call} started in the background"
        assert tasks.notice_text(record) == f"Task {call} finished"


def _finished_record(
    manager: tasks.TaskManager,
    session: str,
    *,
    task_id: str = "tsk_00000000000000aa",
    status: str = tasks.COMPLETED,
) -> tasks.TaskRecord:
    """A finished task put straight into *manager* (no call behind it)."""
    result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text='{"volume_ml": 1520.4}')],
        structuredContent={"volume_ml": 1520.4},
    )
    record = tasks.TaskRecord(
        task_id=task_id,
        stack="medmcp-totalseg",
        tool="segment",
        session=session,
        arguments={"input_path": "/w/ct.nii.gz"},
        created_at=time.time() - 30,
        status=status,
        status_message="Completed",
        promoted_at=time.time() - 10,
        finished_at=time.time(),
        result=result.model_dump(mode="json", by_alias=True, exclude_none=True),
        summary='{"volume_ml": 1520.4}',
    )
    manager._records[task_id] = record  # pyright: ignore[reportPrivateUsage]
    return record


def _working_record(
    manager: tasks.TaskManager, session: str, task_id: str = "tsk_00000000000000bb"
) -> tasks.TaskRecord:
    record = tasks.TaskRecord(
        task_id=task_id, stack="medmcp-totalseg", tool="segment", session=session,
        arguments={}, created_at=time.time(), status_message="Running",
    )  # fmt: skip
    manager._records[task_id] = record  # pyright: ignore[reportPrivateUsage]
    return record


# ── the broker, against the fake stack ────────────────────────────────────────


async def _roundtrip(socket_path: Path, request: JsonDict) -> JsonDict:
    reader, writer = await asyncio.open_unix_connection(str(socket_path), limit=STREAM_LIMIT)
    try:
        writer.write(json.dumps(request).encode() + b"\n")
        await writer.drain()
        line = await reader.readline()
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    return json.loads(line)


class _Stack:
    """A pool on the fake stack, a broker with tasks in front of it, and one without."""

    def __init__(self, pool: BackendPool, manager: tasks.TaskManager, sock: Path) -> None:
        self.pool = pool
        self.manager = manager
        self.sock = sock

    async def call(self, tool: str, args: JsonDict, sock: Path | None = None) -> JsonDict:
        return await _roundtrip(
            sock or self.sock,
            {"op": "call_tool", "stack": "fake", "tool": tool, "args": args, "id": 1},
        )


@pytest.fixture
async def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Stack]:
    """The fake stack behind a pool, a broker with tasks (0.5 s) and one without."""
    pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake", gpu=True)}))
    monkeypatch.setattr(server, "_pool", pool)
    manager = tasks.TaskManager(
        after_sec=0.5, is_gpu=pool.is_gpu, stop_backend=server._stop_task_backend
    )
    monkeypatch.setattr(server, "_tasks", manager)
    with_tasks = BackendBroker(pool, tmp_path / "t.sock", ops=server._core_ops, tasks=manager)
    plain = BackendBroker(pool, tmp_path / "p.sock")
    await with_tasks.start()
    await plain.start()
    try:
        yield _Stack(pool, manager, tmp_path / "t.sock")
    finally:
        await with_tasks.aclose()
        await plain.aclose()
        await manager.aclose()
        await pool.aclose()


def _same_result(a: JsonDict, b: JsonDict) -> None:
    ra, rb = mcp_types.CallToolResult.model_validate(a), mcp_types.CallToolResult.model_validate(b)

    def dump(r: mcp_types.CallToolResult) -> list[JsonDict]:
        return [c.model_dump(mode="json", exclude_none=True) for c in r.content]

    assert dump(ra) == dump(rb)
    assert ra.structuredContent == rb.structuredContent
    assert bool(ra.isError) == bool(rb.isError)


class TestBrokerPath:
    """Calls through the broker socket to the real fake stack."""

    async def test_a_short_call_is_answered_exactly_as_without_tasks(
        self, stack: _Stack, tmp_path: Path
    ) -> None:
        """Below the threshold the broker's answer is byte for byte the old one."""
        with_tasks = await stack.call("echo", {"text": "hi"})
        without = await stack.call("echo", {"text": "hi"}, tmp_path / "p.sock")
        assert with_tasks == without
        assert stack.manager.list() == []

    async def test_get_task_returns_what_the_synchronous_call_returns(
        self, stack: _Stack, tmp_path: Path
    ) -> None:
        """Once completed, get_task gives the result the call would have given in time."""
        sync = await stack.call("sleep", {"seconds": 1.0}, tmp_path / "p.sock")
        handle = await stack.call("sleep", {"seconds": 1.0})
        task_id = tasks.handle_task_id(handle["result"]["structuredContent"])
        assert task_id is not None
        running = await _roundtrip(stack.sock, {"op": "task_get", "args": {"task_id": task_id}})
        assert "still working" in json.dumps(running["result"])
        await _until(lambda: (r := stack.manager.get(task_id)) is not None and r.finished)
        got = await _roundtrip(stack.sock, {"op": "task_get", "args": {"task_id": task_id}})
        _same_result(got["result"], sync["result"])

    async def test_cancel_stops_the_backend_and_marks_the_task_cancelled(
        self, stack: _Stack
    ) -> None:
        """cancel_task (approved) kills the backend running the call and marks it cancelled."""
        handle = await stack.call("sleep", {"seconds": 60})
        task_id = str(tasks.handle_task_id(handle["result"]["structuredContent"]))
        assert stack.pool.warm_names() == ["fake"] and stack.pool.inflight("fake") == 1
        # The op runs only on the person's approval of exactly this cancel_task call.
        refused = await _roundtrip(stack.sock, {"op": "task_cancel", "args": {"task_id": task_id}})
        assert refused["result"]["isError"] is True
        record = stack.manager.get(task_id)
        assert record is not None and record.status == tasks.WORKING
        server._approvals.grant(tasks.TASK_SERVER, tasks.CANCEL_TASK, {"task_id": task_id})
        t0 = time.monotonic()
        done = await _roundtrip(stack.sock, {"op": "task_cancel", "args": {"task_id": task_id}})
        assert time.monotonic() - t0 < 15
        assert "was stopped" in json.dumps(done["result"])
        assert record.status == tasks.CANCELLED
        assert stack.pool.warm_names() == []
        # The model asked for the stop and read the outcome: no automatic turn after.
        assert record.delivered

    async def test_cancel_task_stops_a_background_job_by_its_id(
        self, stack: _Stack, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The agent can stop a running job with cancel_task, on approval as always."""

        class _Job:
            def __init__(self) -> None:
                self.running = True
                self.status = "running"

        job = _Job()
        stopped: list[str] = []

        class _Jobs:
            def get(self, job_id: str) -> _Job:
                if job_id != "job-1":
                    raise server.jobs.JobError(f"no job {job_id!r}")
                return job

            def stop(self, job_id: str) -> _Job:
                stopped.append(job_id)
                job.running, job.status = False, "stopped"
                return job

        monkeypatch.setattr(server, "_jobs", _Jobs())
        args = {"task_id": "job-1"}
        refused = await _roundtrip(stack.sock, {"op": "task_cancel", "args": args})
        assert refused["result"]["isError"] is True and stopped == []
        server._approvals.grant(tasks.TASK_SERVER, tasks.CANCEL_TASK, args)
        done = await _roundtrip(stack.sock, {"op": "task_cancel", "args": args})
        assert "Job job-1 was stopped" in json.dumps(done["result"])
        assert stopped == ["job-1"]


# ── the chat side ─────────────────────────────────────────────────────────────


class _StubWs:
    def __init__(self) -> None:
        self.sent: list[JsonDict] = []

    async def send_json(self, msg: JsonDict) -> None:
        self.sent.append(msg)


class _FakeVibe:
    """vibe-acp as far as a chat connection sees it: prompts, answers, cancels."""

    def __init__(self) -> None:
        self.prompts: list[JsonDict] = []
        self.answers: list[JsonDict] = []
        self.turns: list[asyncio.Future[JsonDict]] = []

    async def request(self, method: str, params: JsonDict) -> JsonDict:
        if method != "session/prompt":
            return {"result": {}}
        self.prompts.append(params)
        turn: asyncio.Future[JsonDict] = asyncio.get_running_loop().create_future()
        self.turns.append(turn)
        return await turn

    def end_turn(self) -> None:
        for turn in self.turns:
            if not turn.done():
                turn.set_result({"result": {"stopReason": "end_turn"}})

    async def notify(self, method: str, params: JsonDict) -> None:
        if method == "session/cancel":
            self.end_turn()

    async def respond(self, req_id: int, result: JsonDict) -> None:
        self.answers.append({"id": req_id, **result})

    def unregister_session(self, session_id: str) -> None:
        del session_id

    def prompt_texts(self) -> list[str]:
        return [str(cast("list[JsonDict]", p["prompt"])[0]["text"]) for p in self.prompts]


@pytest.fixture
def vibe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[_FakeVibe]:
    """vibe-acp faked, no live sockets, provenance in *tmp_path* (off unless a test says)."""
    fake = _FakeVibe()
    monkeypatch.setattr(server, "_client", fake)
    monkeypatch.setattr(server, "_connections", set[server._ChatConnection]())
    monkeypatch.setattr(settings, "load_provenance_enabled", lambda: False)
    monkeypatch.setattr(settings, "load_explain_enabled", lambda: False)
    monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        yield fake


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> tasks.TaskManager:
    """The core's task manager, empty."""
    found = tasks.TaskManager(after_sec=20.0)
    monkeypatch.setattr(server, "_tasks", found)
    return found


@pytest.fixture
def audit() -> Iterator[list[str]]:
    """The lines written to the medmcp.audit log during the test."""
    lines: list[str] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    handler = _Keep(level=logging.INFO)
    logger = logging.getLogger("medmcp.audit")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    yield lines
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def _conn(session: str = "chat") -> server._ChatConnection:
    queue: asyncio.Queue[JsonDict] = asyncio.Queue()
    conn = server._ChatConnection(cast("Any", _StubWs()), session, queue, [])
    server._connections.add(conn)
    return conn


def _sent(conn: server._ChatConnection, kind: str) -> list[JsonDict]:
    return [m for m in cast("_StubWs", conn.ws).sent if m.get("type") == kind]


def _permission(req_id: int, tool: str, **arguments: str) -> JsonDict:
    return {
        "id": req_id,
        "method": "session/request_permission",
        "params": {
            "toolCall": {
                "toolCallId": f"c{req_id}",
                "title": tool,
                "toolName": tool,
                "rawInput": arguments,
            },
            "options": [
                {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
            ],
        },
    }


class TestGetTaskApproval:
    """get_task on this chat's own task: no dialog, audit-logged. Everything else asks."""

    async def _ask(self, conn: server._ChatConnection, req: JsonDict) -> bool:
        """Run the request; whether a dialog was shown (then answered allow_once)."""
        task = asyncio.create_task(conn._handle_permission(req))
        await asyncio.sleep(0.05)
        shown = bool(_sent(conn, "permission_request"))
        if shown:
            conn._pending_perms[cast("int", req["id"])].set_result("allow_once")
        await task
        return shown

    async def test_own_chat_task_is_read_without_a_dialog(
        self, vibe: _FakeVibe, manager: tasks.TaskManager, audit: list[str]
    ) -> None:
        """get_task of this chat's own task: allowed without a dialog, audit-logged."""
        record = _working_record(manager, "chat")
        conn = _conn("chat")
        req = _permission(1, "medmcp-tasks_get_task", task_id=record.task_id)
        assert await self._ask(conn, req) is False
        assert vibe.answers == [
            {"id": 1, "outcome": {"outcome": "selected", "optionId": "allow_once"}}
        ]
        assert f"task status read without dialog: {record.task_id}" in audit

    async def test_another_chats_task_asks(
        self, vibe: _FakeVibe, manager: tasks.TaskManager, audit: list[str]
    ) -> None:
        """get_task of another chat's task goes to the dialog like any call."""
        record = _working_record(manager, "other chat")
        conn = _conn("chat")
        req = _permission(1, "medmcp-tasks_get_task", task_id=record.task_id)
        assert await self._ask(conn, req) is True
        assert not any("without dialog" in line for line in audit)

    async def test_an_unknown_task_asks(self, vibe: _FakeVibe, manager: tasks.TaskManager) -> None:
        """get_task of an id nobody knows goes to the dialog."""
        del manager
        conn = _conn("chat")
        req = _permission(1, "medmcp-tasks_get_task", task_id="tsk_ffffffffffffffff")
        assert await self._ask(conn, req) is True

    async def test_cancel_task_always_asks_and_offers_no_chat_long_option(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """cancel_task asks every time, and its approval is what the broker op spends."""
        record = _working_record(manager, "chat")
        conn = _conn("chat")
        req = _permission(1, "medmcp-tasks_cancel_task", task_id=record.task_id)
        # What the relay cached from the call's tool_call frame.
        conn._tool_calls["c1"] = {
            "toolName": "medmcp-tasks_cancel_task",
            "rawInput": {"task_id": record.task_id},
        }
        assert await self._ask(conn, req) is True
        options = cast("list[JsonDict]", _sent(conn, "permission_request")[0]["options"])
        assert {o["optionId"] for o in options} == {"allow_once", "reject_once"}
        # The approval is granted to exactly that cancel, for the broker op to spend.
        assert server._approvals.consume(
            tasks.TASK_SERVER, tasks.CANCEL_TASK, {"task_id": record.task_id}
        )


def _is_delivery(params: JsonDict) -> bool:
    block = cast("list[JsonDict]", params["prompt"])[0]
    return cast("JsonDict", block.get("_meta") or {}).get("automatic") is True


async def _user_turn(conn: server._ChatConnection, text: str) -> None:
    async with conn._turn_lock:
        conn._start_turn(conn._run_prompt(text))


class TestDelivery:
    """A finished task starts a turn in its chat, at the right moment, once."""

    async def test_delivered_once_the_running_turn_ends(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """A task finishing mid-turn is handed over after that turn, as an automatic turn."""
        conn = _conn("chat")
        await _user_turn(conn, "segment the liver")
        await _until(lambda: len(vibe.prompts) == 1)
        record = _finished_record(manager, "chat")
        await conn.task_finished(record)
        await asyncio.sleep(0.05)
        assert len(vibe.prompts) == 1  # not while the turn runs
        assert _sent(conn, "task_update")[0]["status"] == "completed"
        vibe.end_turn()
        await _until(lambda: len(vibe.prompts) == 2)
        delivery = vibe.prompts[1]
        assert _is_delivery(delivery)
        text = vibe.prompt_texts()[1]
        assert text.startswith(f"[task {record.task_id} (medmcp-totalseg/segment) finished:")
        assert "1520.4" in text
        display = cast("JsonDict", delivery["_meta"])["user_display_content"]
        assert display["content"][0]["text"] == tasks.notice_line(record)
        notice = _sent(conn, "task_notice")
        assert notice and notice[0]["live"] is True and "segment" in notice[0]["text"]
        # The turn opens with the marker the page uses to drop an interrupted
        # answer's last pieces.
        assert _sent(conn, "turn_start")
        assert not _sent(conn, "user")  # never a user bubble
        vibe.end_turn()

    async def test_deferred_while_an_approval_is_open(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """Nothing starts while an approval waits; the turn's end hands the task over."""
        conn = _conn("chat")
        await _user_turn(conn, "go")
        await _until(lambda: len(vibe.prompts) == 1)
        await conn.queue.put(_permission(7, "medmcp-other_tool", path="x"))
        await _until(lambda: bool(conn._pending_perms))
        record = _finished_record(manager, "chat")
        conn.offer_tasks([record.task_id])
        await asyncio.sleep(0.1)
        assert len(vibe.prompts) == 1
        conn._pending_perms[7].set_result("allow_once")
        await asyncio.sleep(0.05)
        assert len(vibe.prompts) == 1  # the approval was answered; the turn goes on
        vibe.end_turn()
        await _until(lambda: len(vibe.prompts) == 2)
        assert _is_delivery(vibe.prompts[1])
        vibe.end_turn()

    async def test_a_user_prompt_meanwhile_goes_first(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """A user prompt that takes the turn first runs first; the task follows."""
        conn = _conn("chat")
        # The user's prompt holds the turn lock when the task finishes.
        async with conn._turn_lock:
            record = _finished_record(manager, "chat")
            conn.offer_tasks([record.task_id])
            await asyncio.sleep(0.05)
            conn._start_turn(conn._run_prompt("my question"))
        await _until(lambda: len(vibe.prompts) == 1)
        await asyncio.sleep(0.05)
        assert vibe.prompt_texts() == ["my question"]
        vibe.end_turn()
        await _until(lambda: len(vibe.prompts) == 2)
        assert _is_delivery(vibe.prompts[1])
        vibe.end_turn()

    async def test_delivered_exactly_once(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """Offers, finishes and a later socket never hand the same task over twice."""
        conn = _conn("chat")
        record = _finished_record(manager, "chat")
        conn.offer_tasks([record.task_id])
        conn.offer_tasks([record.task_id])
        await conn.task_finished(record)
        await _until(lambda: len(vibe.prompts) == 1)
        vibe.end_turn()
        await asyncio.sleep(0.1)
        # A later socket of the same chat finds nothing left to tell.
        again = _conn("chat")
        again.offer_tasks([r.task_id for r in manager.undelivered("chat")])
        again.offer_tasks([record.task_id])
        await asyncio.sleep(0.1)
        assert len(vibe.prompts) == 1

    async def test_kept_until_the_chat_is_opened_again(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """With no socket on the chat the task waits, and is handed over on open."""
        record = _finished_record(manager, "chat")
        await server._on_task_finished(record)  # no socket holds the chat
        assert vibe.prompts == []
        assert [r.task_id for r in manager.undelivered("chat")] == [record.task_id]
        conn = _conn("chat")  # the chat is opened (ws_chat offers what is undelivered)
        conn.offer_tasks([r.task_id for r in manager.undelivered("chat")])
        await _until(lambda: len(vibe.prompts) == 1)
        assert _is_delivery(vibe.prompts[0])
        assert manager.undelivered("chat") == []
        vibe.end_turn()

    async def test_the_replayed_trigger_is_a_notice_not_a_user_turn(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """On resume the automatic turn's trigger comes back as a notice."""
        del vibe
        record = _finished_record(manager, "chat")
        conn = _conn("chat")
        update = {
            "sessionUpdate": "user_message_chunk",
            "content": {"type": "text", "text": "ignored"},
            "_meta": {
                "user_display_content": {
                    "version": "1",
                    "content": [{"type": "text", "text": tasks.notice_line(record)}],
                }
            },
        }
        frame = {"method": "session/update", "params": {"update": update}}
        await conn._forward_frame(frame, replay=True)
        assert _sent(conn, "user") == []
        assert _sent(conn, "task_notice")[0]["live"] is False


# ── what a task leaves behind ─────────────────────────────────────────────────


class TestRecordsLeftBehind:
    """Provenance, distillation and replay of a call that became a task."""

    async def test_provenance_carries_the_final_result_not_the_handle(
        self, vibe: _FakeVibe, manager: tasks.TaskManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call's run-log event waits for the task and carries its real output."""
        del vibe
        monkeypatch.setattr(settings, "load_provenance_enabled", lambda: True)
        record = _working_record(manager, "chat")
        conn = _conn("chat")
        handle = tasks.handle_result(record)
        envelope = {
            "type": "success",
            "content": [{"type": "text", "text": json.dumps(handle.structuredContent)}],
        }
        await conn._forward_frame(_frame("tool_call", "c1", status="in_progress"))
        await conn._forward_frame(
            _frame("tool_call_update", "c1", status="completed", rawOutput=envelope)
        )
        update = _sent(conn, "tool_call_update")[0]
        assert cast("JsonDict", update["task"])["taskId"] == record.task_id
        assert provenance.read_run_events("chat") == []  # held back: it is only the handle
        record.status = tasks.COMPLETED
        record.finished_at = time.time()
        record.result = _text_payload({"liver_ml": 1520.4})
        await server._on_task_finished(record)
        events = provenance.read_run_events("chat")
        call = next(e for e in events if e.get("tool_call_id") == "c1")
        assert call["output"] == {"liver_ml": 1520.4}
        assert call["status"] == "completed"
        assert call["task"] == {"id": record.task_id, "status": "completed"}
        finished = next(e for e in events if e.get("event") == provenance.TASK_FINISHED)
        assert finished["task_id"] == record.task_id and "1520.4" in finished["summary"]
        assert not any("medmcp_task" in json.dumps(e.get("output")) for e in events)

    async def test_distill_folds_the_final_result_into_the_call(
        self, vibe: _FakeVibe, manager: tasks.TaskManager
    ) -> None:
        """The step gets the task's result; get_task is not a step."""
        del vibe, manager
        task_id = "tsk_0000000000000abc"
        handle = json.dumps(
            {"medmcp_task": {"taskId": task_id, "status": "working", "message": "…"}}
        )
        final = json.dumps({"mask_path": "/w/out/liver.nii.gz", "liver_ml": 1520.4})
        messages: list[JsonDict] = [
            {"role": "user", "content": "segment the liver"},
            _call("c1", "medmcp-totalseg_segment", {"input_path": "/w/ct.nii.gz"}),
            {"role": "tool", "tool_call_id": "c1", "content": handle, "status": "completed"},
            _call("c2", "medmcp-tasks_get_task", {"task_id": task_id}),
            {"role": "tool", "tool_call_id": "c2", "content": "still working"},
            provenance.task_result_record(task_id, "completed", final),
            _call("c3", "medmcp-totalseg_stats", {"mask": "/w/out/liver.nii.gz"}),
            {"role": "tool", "tool_call_id": "c3", "content": '{"ok": true}'},
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-totalseg"], name="x", description="d"
        )
        assert [s.tool for s in recipe.steps] == ["segment", "stats"]
        assert recipe.steps[0].produces == {"mask_path": "step1.mask_path"}
        assert recipe.steps[1].arguments == {"mask": "{{step1.mask_path}}"}

    async def test_an_unfinished_task_is_not_a_step(self) -> None:
        """A handle with no result behind it is skipped like an interrupted call."""
        handle = json.dumps({"medmcp_task": {"taskId": "tsk_0000000000000abd"}})
        messages: list[JsonDict] = [
            _call("c1", "medmcp-totalseg_segment", {"input_path": "/w/ct.nii.gz"}),
            {"role": "tool", "tool_call_id": "c1", "content": handle},
        ]
        recipe = distill.build_recipe(
            messages, server_names=["medmcp-totalseg"], name="x", description="d"
        )
        assert recipe.steps == []

    async def test_replay_runs_a_long_tool_synchronously(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replay calls stacks itself, below the broker: no threshold, no handle."""
        monkeypatch.setenv(tasks.AFTER_ENV, "0.1")
        spec = fake_spec("fake")
        servers: list[JsonDict] = [
            {"name": "fake", "command": spec.command, "args": spec.args, "env": {}}
        ]
        recipe = Recipe(
            name="r",
            description="d",
            steps=[RecipeStep(server="fake", tool="sleep", arguments={"seconds": 0.6})],
        )
        report = await replay.run(recipe, {}, servers=servers)
        assert report.ok
        assert report.steps[0].output == {"slept": 0.6}


def _call(call_id: str, name: str, arguments: JsonDict) -> JsonDict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


def _text_payload(value: JsonDict) -> JsonDict:
    result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(value))],
        structuredContent=value,
    )
    return result.model_dump(mode="json", by_alias=True, exclude_none=True)


def _frame(kind: str, tc_id: str, **fields: object) -> JsonDict:
    update: JsonDict = {
        "sessionUpdate": kind,
        "toolCallId": tc_id,
        "_meta": {"tool_name": "mcp_medmcp_totalseg.segment"},
        "rawInput": {"input_path": "/w/ct.nii.gz"},
        **fields,
    }
    if kind == "tool_call":
        update["title"] = "mcp_medmcp_totalseg.segment"
    return {"method": "session/update", "params": {"update": update}}


# ── the task tools in vibe's config ───────────────────────────────────────────


class TestTaskServerEntry:
    """medmcp-tasks is offered exactly when long calls can become tasks."""

    @staticmethod
    def _entries(vibe_home: Path) -> dict[str, JsonDict]:
        stack: JsonDict = {"name": "medmcp-x", "command": "/abs/x", "args": [], "env": {}}
        settings.sync_servers_to_vibe_config([stack])
        with (vibe_home / "config.toml").open("rb") as fh:
            cfg = tomllib.load(fh)
        return {str(e["name"]): e for e in cast("list[JsonDict]", cfg["mcp_servers"])}

    def test_written_with_the_pool_on(
        self, monkeypatch: pytest.MonkeyPatch, vibe_home: Path
    ) -> None:
        """Pool on: vibe spawns medmcp-tasks, pointed at the broker socket."""
        monkeypatch.setenv("MEDMCP_STACK_POOL", "1")
        monkeypatch.delenv(tasks.AFTER_ENV, raising=False)
        entry = self._entries(vibe_home)[tasks.TASK_SERVER]
        assert Path(str(entry["command"])).name == tasks.TASK_COMMAND
        assert entry["env"] == {"MEDMCP_BROKER_SOCK": str(vibe_home / "backend.sock")}
        assert entry["sampling_enabled"] is False

    def test_absent_with_the_pool_off_or_tasks_off(
        self, monkeypatch: pytest.MonkeyPatch, vibe_home: Path
    ) -> None:
        """Pool off (or MEDMCP_TASK_AFTER_SEC=0): no task tools at all."""
        monkeypatch.delenv("MEDMCP_STACK_POOL", raising=False)
        assert tasks.TASK_SERVER not in self._entries(vibe_home)
        monkeypatch.setenv("MEDMCP_STACK_POOL", "1")
        monkeypatch.setenv(tasks.AFTER_ENV, "0")
        assert tasks.TASK_SERVER not in self._entries(vibe_home)

    def test_never_adopted_as_a_stack(
        self, monkeypatch: pytest.MonkeyPatch, vibe_home: Path
    ) -> None:
        """Discovery never reads the synced entry back as a manual stack."""
        monkeypatch.setenv("MEDMCP_STACK_POOL", "1")
        self._entries(vibe_home)
        settings.load_mcp_servers.cache_clear()
        assert tasks.TASK_SERVER not in {s["name"] for s in settings.load_mcp_servers()}
        settings.load_mcp_servers.cache_clear()
