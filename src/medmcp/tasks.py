"""Tasks: a long stack tool call that answers at once and finishes in the background.

The shape follows the MCP Tasks extension (``io.modelcontextprotocol/tasks``,
spec 2026-07-28): a task has a ``taskId``, a ``status`` (``working``,
``completed``, ``failed`` or ``cancelled``; ``input_required`` is never used
here), a ``statusMessage``, the original ``CallToolResult`` once completed or an
``error`` once failed, a ``ttlMs`` and a ``pollIntervalMs``. Neither vibe nor the
MCP Python SDK speaks the extension yet, so it is implemented in the core, below
both, and the stacks stay unchanged:

1. The broker forwards a stack call as always and waits up to a threshold
   (:func:`after_sec`, ``MEDMCP_TASK_AFTER_SEC``, default 20 s, ``0`` = off). A
   call that finishes in time is answered exactly as before.
2. A call that does not is *promoted*: the broker answers vibe at once with a
   task handle (:func:`handle_result`) and the call keeps running in the pool —
   the backend stays busy and is not cancelled when vibe's program ends.
3. The model reads a task with ``get_task`` and stops one with ``cancel_task``,
   two tools the core serves (``medmcp-tasks``, :mod:`medmcp.tasktool`, over the
   broker ops ``task_get`` / ``task_cancel``).
4. When a task finishes, the core starts a turn in the chat that owns it with an
   automatic message (:func:`delivery_prompt`), so the assistant continues by
   itself (see ``server.py``).

At most one call on a GPU stack runs at a time; a further one waits in a FIFO
queue rather than being refused, and is promoted after the threshold like any
other call ("Waiting for the GPU (1 ahead)"). Calls on CPU stacks are not queued.

Tasks live in this process. Their records — not result bodies over
:data:`RESULT_CAP_BYTES` — are kept in ``.vibe/state/tasks.json`` so that after a
restart an unfinished task is reported as failed ("MedMCP restarted") to the chat
that owns it, and finished results stay readable until their TTL. Work that takes
hours belongs to background jobs (:mod:`medmcp.jobs`), which outlive the core.

With the stack pool off (the host-native default) there is no broker, so nothing
here runs: every stack call stays synchronous, exactly as before.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import mcp.types as mcp_types

from medmcp.statefile import atomic_write_json, read_json

log: logging.Logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

# The core's own MCP server for the task tools, and its command (see tasktool).
TASK_SERVER: Final = "medmcp-tasks"
TASK_COMMAND: Final = "medmcp-task-tool"
GET_TASK: Final = "get_task"
CANCEL_TASK: Final = "cancel_task"
TASK_TOOLS: Final = frozenset({GET_TASK, CANCEL_TASK})

AFTER_ENV: Final = "MEDMCP_TASK_AFTER_SEC"
DEFAULT_AFTER_SEC: Final = 20.0
DEFAULT_TTL_MS: Final = 3_600_000
POLL_INTERVAL_MS: Final = 5_000
# A result body larger than this (as JSON) is kept in memory only, not in the
# state file; after a restart such a task reports a summary instead.
RESULT_CAP_BYTES: Final = 64 * 1024
# How much of a result the automatic "task finished" message carries.
DELIVERY_CLIP_CHARS: Final = 4000

WORKING: Final = "working"
COMPLETED: Final = "completed"
FAILED: Final = "failed"
CANCELLED: Final = "cancelled"
FINISHED: Final = frozenset({COMPLETED, FAILED, CANCELLED})

# The key a task handle carries in its structured content.
HANDLE_KEY: Final = "medmcp_task"
RESTARTED_MESSAGE: Final = "MedMCP restarted"

_ID_PATTERN = re.compile(r"\btsk_[0-9a-f]{16}\b")
# The first line of the automatic message, also what the chat shows of it.
_NOTICE_PATTERN = re.compile(r"^\[task (tsk_[0-9a-f]{16}) \(([^)]*)\) finished: (\w+)\]")

# The schema a task handle satisfies; the proxy adds it to every tool's output
# schema as an alternative, since vibe validates structured content against it.
HANDLE_SCHEMA: Final[JsonDict] = {
    "type": "object",
    "properties": {
        HANDLE_KEY: {
            "type": "object",
            "properties": {
                "taskId": {"type": "string"},
                "status": {"type": "string"},
                "pollIntervalMs": {"type": "integer"},
                "message": {"type": "string"},
            },
            "required": ["taskId", "status"],
        }
    },
    "required": [HANDLE_KEY],
}


def task_tool(name: str) -> str | None:
    """``get_task`` / ``cancel_task`` when *name* is one of the core's task tools.

    *name* is MedMCP's canonical name (``medmcp-tasks_get_task``) or the harness
    route (``mcp_medmcp_tasks.get_task``), in either spelling of the server.
    """
    for prefix in (f"{TASK_SERVER}_", "medmcp_tasks_", "mcp_medmcp_tasks."):
        if name.startswith(prefix) and name[len(prefix) :] in TASK_TOOLS:
            return name[len(prefix) :]
    return None


def is_task_tool(name: str) -> bool:
    """Whether *name* is one of the core's task tools (see :func:`task_tool`)."""
    return task_tool(name) is not None


def after_sec() -> float:
    """Seconds a stack call may take before it becomes a task (``0``: never)."""
    raw = os.environ.get(AFTER_ENV, "").strip()
    if not raw:
        return DEFAULT_AFTER_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_AFTER_SEC


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None


def new_task_id() -> str:
    """A fresh task id (``tsk_`` and 16 hex digits)."""
    return f"tsk_{secrets.token_hex(8)}"


# A background job (medmcp.jobs) can adopt this record later: its container id
# becomes ``task_id``, ``running``/``finished``/``stopped``/``failed`` map onto
# working/completed/cancelled/failed, its progress line is ``status_message``, and
# ``result`` stays empty (a job's outcome is files in the workspace). The jobs
# list and stop endpoint would then serve TaskRecord.public() like tasks do.
@dataclass
class TaskRecord:
    """One task, in the fields of an MCP Tasks ``Task`` plus what MedMCP needs.

    Attributes:
        task_id: ``tsk_…``.
        stack: The stack the call went to.
        tool: The tool called.
        session: The chat (canonical session id) whose approval the call spent.
        arguments: The call's arguments.
        created_at: When the call started (epoch seconds).
        status: ``working``, ``completed``, ``failed`` or ``cancelled``.
        status_message: A one-line state for people and the model.
        gpu: Whether the call held (or waited for) the GPU slot.
        promoted_at: When the call became a task.
        finished_at: When it ended, if it has.
        result: The original ``CallToolResult`` (dumped), once completed.
        error: Why it failed or was cancelled.
        summary: The result's text, clipped; kept when the body is too large.
        ttl_ms: How long a finished task is kept.
        poll_interval_ms: How often a client may sensibly poll.
        tool_call_id: The chat's tool call that returned the handle, once seen.
        delivered: Whether the owning chat has been told that it finished.
        call_event: The provenance event of that tool call, waiting for the
            final result (written when the task finishes).
    """

    task_id: str
    stack: str
    tool: str
    session: str
    arguments: JsonDict
    created_at: float
    status: str = WORKING
    status_message: str = ""
    gpu: bool = False
    promoted_at: float | None = None
    finished_at: float | None = None
    result: JsonDict | None = None
    error: str | None = None
    summary: str = ""
    ttl_ms: int = DEFAULT_TTL_MS
    poll_interval_ms: int = POLL_INTERVAL_MS
    tool_call_id: str = ""
    delivered: bool = False
    call_event: JsonDict | None = None

    @property
    def finished(self) -> bool:
        """Whether the task has reached a terminal status."""
        return self.status in FINISHED

    def elapsed_sec(self, now: float) -> float:
        """Seconds since the call started, up to its end if it has ended."""
        end = self.finished_at if self.finished_at is not None else now
        return max(0.0, end - self.created_at)

    def public(self, now: float) -> JsonDict:
        """The task as the API and the model see it (no result body)."""
        out: JsonDict = {
            "taskId": self.task_id,
            "status": self.status,
            "statusMessage": self.status_message,
            "createdAt": _iso(self.created_at),
            "lastUpdatedAt": _iso(self.finished_at or self.promoted_at or self.created_at),
            "ttlMs": self.ttl_ms,
            "pollIntervalMs": self.poll_interval_ms,
            "stack": self.stack,
            "tool": self.tool,
            "session": self.session,
            "elapsedSec": round(self.elapsed_sec(now), 1),
            "toolCallId": self.tool_call_id,
            "arguments": self.arguments,
        }
        if self.error is not None:
            out["error"] = self.error
        return out

    def to_state(self) -> JsonDict:
        """The record as stored in the state file (a large result body left out)."""
        data = asdict(self)
        if self.result is not None and len(json.dumps(self.result)) > RESULT_CAP_BYTES:
            data["result"] = None
        return data

    @classmethod
    def from_state(cls, data: JsonDict) -> TaskRecord | None:
        """A record read back from the state file; ``None`` when it is not one."""
        try:
            record = cls(
                task_id=str(data["task_id"]),
                stack=str(data["stack"]),
                tool=str(data["tool"]),
                session=str(data.get("session") or ""),
                arguments=cast("JsonDict", data.get("arguments") or {}),
                created_at=float(data["created_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
        record.status = str(data.get("status") or WORKING)
        record.status_message = str(data.get("status_message") or "")
        record.gpu = bool(data.get("gpu"))
        for name in ("promoted_at", "finished_at"):
            value = data.get(name)
            setattr(record, name, float(value) if isinstance(value, (int, float)) else None)
        result = data.get("result")
        record.result = cast("JsonDict", result) if isinstance(result, dict) else None
        error = data.get("error")
        record.error = str(error) if error is not None else None
        record.summary = str(data.get("summary") or "")
        record.ttl_ms = int(data.get("ttl_ms") or DEFAULT_TTL_MS)
        record.poll_interval_ms = int(data.get("poll_interval_ms") or POLL_INTERVAL_MS)
        record.tool_call_id = str(data.get("tool_call_id") or "")
        record.delivered = bool(data.get("delivered"))
        event = data.get("call_event")
        record.call_event = cast("JsonDict", event) if isinstance(event, dict) else None
        if not _ID_PATTERN.fullmatch(record.task_id):
            return None
        return record


# ── what the model and the chat see ──────────────────────────────────────────


def result_text(result: mcp_types.CallToolResult) -> str:
    """A result as vibe shows it to the model: the structured content, else the text."""
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent)
    return "\n".join(b.text for b in result.content if isinstance(b, mcp_types.TextContent))


def result_value(result: mcp_types.CallToolResult) -> object:
    """A result's value as provenance records it: the structured content, else the text."""
    if result.structuredContent is not None:
        return result.structuredContent
    return result_text(result)


def clip(text: str, limit: int) -> str:
    """*text* cut to about *limit* characters at a line or word boundary."""
    if len(text) <= limit:
        return text
    cut = max(text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
    if cut < limit // 2:
        cut = limit
    return text[:cut].rstrip() + " … [clipped]"


def handle_message(task_id: str) -> str:
    """The handle's line for the model."""
    return (
        f"This call continues in the background as task {task_id}. Do not wait for it, "
        f"check on it or make a plan for it: say in one short sentence that it is running "
        f"and end your reply. When it ends, its result comes to you in a new message and "
        f"you continue from there. To stop it, call cancel_task with this task id."
    )


def handle_result(record: TaskRecord) -> mcp_types.CallToolResult:
    """The tool result that stands in for a promoted call.

    vibe shows the model only the structured content when there is some, so the
    line for the model rides inside it too (``message``).
    """
    message = handle_message(record.task_id)
    handle: JsonDict = {
        "taskId": record.task_id,
        "status": WORKING,
        "pollIntervalMs": record.poll_interval_ms,
        "message": message,
    }
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=message)],
        structuredContent={HANDLE_KEY: handle},
    )


def handle_task_id(output: object) -> str | None:
    """The task id when *output* (a tool result's text or value) is a task handle."""
    if isinstance(output, dict):
        handle = cast("JsonDict", output).get(HANDLE_KEY)
        if isinstance(handle, dict):
            task_id = cast("JsonDict", handle).get("taskId")
            if isinstance(task_id, str) and _ID_PATTERN.fullmatch(task_id):
                return task_id
        return None
    if not isinstance(output, str):
        return None
    # The quoted key, not the bare word: a route such as mcp_medmcp_tasks.get_task
    # contains "medmcp_task" too.
    if f'"{HANDLE_KEY}"' not in output and "in the background as task" not in output:
        return None
    match = _ID_PATTERN.search(output)
    return match.group(0) if match else None


def with_handle_schema(output_schema: JsonDict | None) -> JsonDict | None:
    """*output_schema* widened so a task handle also satisfies it (``None`` stays).

    Local references (``#/$defs/X``) resolve against the root, so the original's
    ``$defs`` / ``definitions`` move up to the new root with it; left inside the
    ``anyOf`` branch, every result of a tool that uses them would fail validation.
    """
    if output_schema is None:
        return None
    original = dict(output_schema)
    widened: JsonDict = {"type": "object"}
    for key in ("$defs", "definitions"):
        if key in original:
            widened[key] = original.pop(key)
    widened["anyOf"] = [original, HANDLE_SCHEMA]
    return widened


def notice_line(record: TaskRecord) -> str:
    """The first line of the automatic message about one finished task."""
    return f"[task {record.task_id} ({record.stack}/{record.tool}) finished: {record.status}]"


def parse_notice(text: str) -> tuple[str, str, str] | None:
    """``(task_id, "stack/tool", status)`` when *text* starts with a notice line."""
    match = _NOTICE_PATTERN.match(text.strip())
    return (match.group(1), match.group(2), match.group(3)) if match else None


def _body(record: TaskRecord) -> str:
    if record.status == COMPLETED:
        if record.result is not None:
            result = mcp_types.CallToolResult.model_validate(record.result)
            text = result_text(result)
            prefix = "The tool reported an error:\n" if result.isError else ""
            return prefix + clip(text, DELIVERY_CLIP_CHARS)
        return clip(record.summary, DELIVERY_CLIP_CHARS) or "(no result kept)"
    return f"Error: {record.error or record.status_message or record.status}"


def delivery_prompt(records: Iterable[TaskRecord]) -> tuple[str, str]:
    """The automatic message for finished tasks: what the model reads, what the chat shows.

    The model gets each notice line followed by the result (clipped) or the
    error; the chat shows the notice lines only.
    """
    blocks: list[str] = []
    lines: list[str] = []
    for record in records:
        line = notice_line(record)
        lines.append(line)
        blocks.append(f"{line}\n{_body(record)}")
    return "\n\n".join(blocks), "\n".join(lines)


def started_text(record: TaskRecord) -> str:
    """The one line the chat shows when a call became a task ("Task … started …")."""
    return f"Task {record.stack.removeprefix('medmcp-')} · {record.tool} started in the background"


def notice_text(record_or_line: TaskRecord | str) -> str:
    """The one line the chat shows for a finished task ("Task … finished")."""
    if isinstance(record_or_line, TaskRecord):
        parsed: tuple[str, str, str] | None = (
            record_or_line.task_id,
            f"{record_or_line.stack}/{record_or_line.tool}",
            record_or_line.status,
        )
    else:
        parsed = parse_notice(record_or_line)
    if parsed is None:
        return str(record_or_line)
    _task_id, call, status = parsed
    stack, _, tool = call.partition("/")
    word = {COMPLETED: "finished", FAILED: "failed", CANCELLED: "was stopped"}.get(status, status)
    return f"Task {stack.removeprefix('medmcp-')} · {tool} {word}"


def tool_result_for(record: TaskRecord, now: float) -> mcp_types.CallToolResult:
    """What ``get_task`` returns for *record*.

    A task that ran to the end returns the original result unchanged — what the
    call would have returned had it finished in time (an error result included).
    Otherwise one line with the status, and the task's fields as structured
    content; a task that failed or was stopped is reported, not raised: reading
    it worked (vibe turns an error result into an exception whose text the
    model never sees).
    """
    if record.result is not None and record.status in (COMPLETED, FAILED):
        return mcp_types.CallToolResult.model_validate(record.result)
    public = record.public(now)
    # vibe shows a structured result carrying "error" as a failed call; the call
    # did not fail (it read a task that did), so the reason goes in statusMessage.
    error = public.pop("error", None)
    if error:
        public["statusMessage"] = error
    elapsed = int(record.elapsed_sec(now))
    if record.status == WORKING:
        text = (
            f"Task {record.task_id} ({record.stack}/{record.tool}) is still working: "
            f"{record.status_message}, {elapsed} s so far. Do not call get_task again: finish "
            f"your reply now; the result comes to you in a new message when it ends."
        )
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=text)], structuredContent=public
        )
    if record.status == COMPLETED:
        text = f"Task {record.task_id} completed. Result (summary): {record.summary}"
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=text)], structuredContent=public
        )
    verb = "was stopped" if record.status == CANCELLED else "failed"
    text = f"Task {record.task_id} ({record.stack}/{record.tool}) {verb}: {record.error or ''}"
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text.rstrip(": "))],
        structuredContent=public,
    )


def unknown_task_result(task_id: str) -> mcp_types.CallToolResult:
    """``get_task`` for an id the core does not know (said, not raised: see above)."""
    text = f"No task {task_id!r} is known (finished tasks are kept for an hour)."
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        structuredContent={"taskId": task_id, "status": "unknown", "statusMessage": text},
    )


# ── the manager ───────────────────────────────────────────────────────────────

Work = Callable[[], Awaitable[mcp_types.CallToolResult]]
TaskCallback = Callable[[TaskRecord], Awaitable[None]]


@dataclass(eq=False)
class _Call:
    """One stack call in flight under the manager (promoted or not)."""

    stack: str
    tool: str
    args: JsonDict
    session: str
    gpu: bool
    created_at: float
    started_at: float | None = None
    job: asyncio.Task[mcp_types.CallToolResult] | None = None
    record: TaskRecord | None = None
    cancel_requested: bool = False
    watcher: asyncio.Task[None] | None = field(default=None, repr=False)


class _GpuQueue:
    """First come, first served: one GPU call runs, the rest wait in order."""

    def __init__(self) -> None:
        self.holder: _Call | None = None
        self._waiters: list[tuple[_Call, asyncio.Future[None]]] = []

    def ahead(self, call: _Call) -> int:
        """How many calls run or wait before *call* (0 once it holds the slot)."""
        for index, (waiting, _) in enumerate(self._waiters):
            if waiting is call:
                return index + (1 if self.holder is not None else 0)
        return 0

    async def acquire(self, call: _Call, on_change: Callable[[], None]) -> None:
        if self.holder is None and not self._waiters:
            self.holder = call
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append((call, future))
        on_change()
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release(call, on_change)  # handed the slot as it was cancelled
            else:
                self._waiters = [(c, f) for c, f in self._waiters if c is not call]
                on_change()
            raise

    def release(self, call: _Call, on_change: Callable[[], None]) -> None:
        if self.holder is not call:
            return
        self.holder = None
        while self._waiters:
            nxt, future = self._waiters.pop(0)
            if not future.done():
                self.holder = nxt
                future.set_result(None)
                break
        on_change()


class TaskManager:
    """Runs stack calls, promotes the slow ones to tasks, and keeps the records."""

    def __init__(
        self,
        *,
        after_sec: float,
        state_path: Path | None = None,
        is_gpu: Callable[[str], bool] = lambda _stack: False,
        stop_backend: Callable[[str], Awaitable[None]] | None = None,
        on_promoted: TaskCallback | None = None,
        on_finished: TaskCallback | None = None,
        ttl_ms: int = DEFAULT_TTL_MS,
        poll_interval_ms: int = POLL_INTERVAL_MS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Create a manager; :meth:`load` reads back the previous process's records.

        Args:
            after_sec: How long a call may take before it is promoted (> 0).
            state_path: The state file (``.vibe/state/tasks.json``), or ``None``.
            is_gpu: Whether a stack holds the GPU (its calls are queued).
            stop_backend: Stops a stack's backend (the pool's evict path), used
                when a running task is cancelled.
            on_promoted: Awaited when a call becomes a task.
            on_finished: Awaited once when a task reaches a terminal status.
            ttl_ms: How long a finished task is kept.
            poll_interval_ms: The polling interval a handle suggests.
            clock: Epoch seconds (injected by tests).
        """
        self.after_sec = after_sec
        self._state_path = state_path
        self._is_gpu = is_gpu
        self._stop_backend = stop_backend
        self._on_promoted = on_promoted
        self._on_finished = on_finished
        self._ttl_ms = ttl_ms
        self._poll_ms = poll_interval_ms
        self._clock = clock
        self._records: dict[str, TaskRecord] = {}
        self._calls: dict[str, _Call] = {}
        self._gpu = _GpuQueue()
        self._waiting: list[_Call] = []  # GPU calls queued, promoted or not

    # ── running calls ────────────────────────────────────────────────────────

    async def run(
        self, stack: str, tool: str, args: JsonDict, *, session: str, work: Work
    ) -> mcp_types.CallToolResult:
        """Run *work* (the stack call); its result, or a task handle after the threshold.

        A call that ends in time returns (or raises) exactly what *work* does.
        One that does not is registered as a task and keeps running; the
        handle is returned instead. Cancelling this coroutine before the
        promotion cancels the call too, as a plain call would be.
        """
        call = _Call(
            stack=stack,
            tool=tool,
            args=args,
            session=session,
            gpu=self._is_gpu(stack),
            created_at=self._clock(),
        )
        job = asyncio.create_task(self._execute(call, work), name=f"task:{stack}.{tool}")
        call.job = job
        try:
            done, _ = await asyncio.wait({job}, timeout=self.after_sec)
        except asyncio.CancelledError:
            job.cancel()
            raise
        if job in done:
            return job.result()
        record = self._promote(call)
        return handle_result(record)

    async def _execute(self, call: _Call, work: Work) -> mcp_types.CallToolResult:
        if call.gpu:
            await self._gpu.acquire(call, self._refresh_messages)
        try:
            call.started_at = self._clock()
            self._refresh_messages()
            return await work()
        finally:
            if call.gpu:
                self._gpu.release(call, self._refresh_messages)

    def _message(self, call: _Call) -> str:
        if call.started_at is None and call.gpu:
            return f"Waiting for the GPU ({self._gpu.ahead(call)} ahead)"
        return "Running"

    def _refresh_messages(self) -> None:
        for call in self._calls.values():
            if call.record is not None and call.record.status == WORKING:
                call.record.status_message = self._message(call)

    def _promote(self, call: _Call) -> TaskRecord:
        record = TaskRecord(
            task_id=new_task_id(),
            stack=call.stack,
            tool=call.tool,
            session=call.session,
            arguments=call.args,
            created_at=call.created_at,
            gpu=call.gpu,
            promoted_at=self._clock(),
            ttl_ms=self._ttl_ms,
            poll_interval_ms=self._poll_ms,
        )
        call.record = record
        record.status_message = self._message(call)
        self._records[record.task_id] = record
        self._calls[record.task_id] = call
        self._save()
        log.info("call %s.%s became task %s", call.stack, call.tool, record.task_id)
        call.watcher = asyncio.create_task(self._watch(call), name=f"watch:{record.task_id}")
        if self._on_promoted is not None:
            callback = self._on_promoted
            asyncio.get_running_loop().create_task(self._guarded(callback, record))
        return record

    async def _guarded(self, callback: TaskCallback, record: TaskRecord) -> None:
        try:
            await callback(record)
        except Exception:
            log.exception("task callback failed for %s", record.task_id)

    async def _watch(self, call: _Call) -> None:
        job, record = call.job, call.record
        assert job is not None and record is not None
        await asyncio.wait({job})
        if record.finished:
            return
        if job.cancelled() or call.cancel_requested:
            record.status = CANCELLED
            record.status_message = "Stopped"
            record.error = "Stopped before it finished."
        elif (exc := job.exception()) is not None:
            record.status = FAILED
            record.status_message = "Failed"
            record.error = str(exc) or type(exc).__name__
        else:
            result = job.result()
            record.result = result.model_dump(mode="json", by_alias=True, exclude_none=True)
            record.summary = clip(result_text(result), 2000)
            record.status = FAILED if result.isError else COMPLETED
            record.status_message = "Failed" if result.isError else "Completed"
            if result.isError:
                record.error = clip(result_text(result), 2000)
        record.finished_at = self._clock()
        self._calls.pop(record.task_id, None)
        self._save()
        log.info("task %s %s", record.task_id, record.status)
        if self._on_finished is not None:
            await self._guarded(self._on_finished, record)

    # ── reading and changing tasks ───────────────────────────────────────────

    def get(self, task_id: str) -> TaskRecord | None:
        """The task *task_id*, unless unknown or past its TTL."""
        self._expire()
        return self._records.get(task_id)

    def list(self) -> list[TaskRecord]:
        """Every task still kept, newest first."""
        self._expire()
        return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def public(self, record: TaskRecord) -> JsonDict:
        """*record* as the API shows it."""
        return record.public(self._clock())

    def tool_result(self, task_id: str) -> mcp_types.CallToolResult:
        """What ``get_task`` returns for *task_id*."""
        record = self.get(task_id)
        if record is None:
            return unknown_task_result(task_id)
        return tool_result_for(record, self._clock())

    async def cancel(self, task_id: str, *, timeout: float = 10.0) -> TaskRecord | None:
        """Stop a working task: cancel its call and stop the backend it was running on.

        A task still waiting for the GPU just leaves the queue. A finished task
        is returned unchanged; an unknown one is ``None``.
        """
        record = self.get(task_id)
        call = self._calls.get(task_id)
        if record is None or record.finished or call is None or call.job is None:
            return record
        running = call.started_at is not None
        call.cancel_requested = True
        call.job.cancel()
        if call.watcher is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(call.watcher), timeout)
        if running and self._stop_backend is not None:
            try:
                await self._stop_backend(record.stack)
            except Exception:
                log.exception("could not stop the backend of %s", task_id)
        return record

    def bind_call(
        self, task_id: str, tool_call_id: str, event: JsonDict | None
    ) -> TaskRecord | None:
        """Note the chat's tool call that got *task_id*'s handle, and its provenance event."""
        record = self.get(task_id)
        if record is None:
            return None
        if not record.tool_call_id:
            record.tool_call_id = tool_call_id
        if event is not None and record.call_event is None:
            record.call_event = event
        self._save()
        return record

    def take_call_event(self, task_id: str) -> JsonDict | None:
        """The pending provenance event of *task_id*'s call, removed (written once)."""
        record = self._records.get(task_id)
        if record is None or record.call_event is None:
            return None
        event, record.call_event = record.call_event, None
        self._save()
        return event

    def undelivered(self, session: str) -> list[TaskRecord]:
        """Finished tasks of chat *session* it has not been told about, oldest first."""
        self._expire()
        found = [
            r
            for r in self._records.values()
            if r.session == session and r.finished and not r.delivered
        ]
        return sorted(found, key=lambda r: r.finished_at or 0.0)

    def claim_delivery(self, session: str, task_ids: Iterable[str]) -> list[TaskRecord]:
        """Mark these finished tasks of *session* as told and return them (never twice)."""
        claimed: list[TaskRecord] = []
        for task_id in task_ids:
            record = self.get(task_id)
            if record is None or record.session != session:
                continue
            if record.finished and not record.delivered:
                record.delivered = True
                claimed.append(record)
        if claimed:
            self._save()
        return claimed

    def belongs_to(self, task_id: str, session: str) -> bool:
        """Whether *task_id* is a task started in chat *session*."""
        record = self.get(task_id)
        return record is not None and bool(session) and record.session == session

    # ── persistence ──────────────────────────────────────────────────────────

    def load(self) -> list[TaskRecord]:
        """Read the previous process's records; its unfinished tasks are marked failed.

        Returns the tasks this marked failed, so the caller can record and
        report them.
        """
        data = read_json(self._state_path) if self._state_path is not None else None
        raw = (data or {}).get("tasks")
        restarted: list[TaskRecord] = []
        now = self._clock()
        for item in cast("list[object]", raw) if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            record = TaskRecord.from_state(cast("JsonDict", item))
            if record is None:
                continue
            if not record.finished:
                record.status = FAILED
                record.status_message = RESTARTED_MESSAGE
                record.error = "MedMCP restarted while the task was running."
                record.finished_at = now
                record.delivered = False
                restarted.append(record)
            self._records[record.task_id] = record
        self._expire()
        if restarted:
            self._save()
        return restarted

    def _expire(self) -> None:
        now = self._clock()
        gone = [
            task_id
            for task_id, r in self._records.items()
            if r.finished and r.finished_at is not None and now > r.finished_at + r.ttl_ms / 1000
        ]
        for task_id in gone:
            del self._records[task_id]
        if gone:
            self._save()

    def _save(self) -> None:
        if self._state_path is None:
            return
        payload = {"version": 1, "tasks": [r.to_state() for r in self._records.values()]}
        try:
            atomic_write_json(self._state_path, payload)
        except OSError as exc:
            log.warning("could not write the task state: %s", exc)

    async def aclose(self) -> None:
        """Stop watching (the process is ending); the state file keeps the records.

        Unfinished tasks stay ``working`` on disk, so the next start reports them
        as failed ("MedMCP restarted").
        """
        for call in list(self._calls.values()):
            for task in (call.watcher, call.job):
                if task is not None and not task.done():
                    task.cancel()
        self._calls.clear()
