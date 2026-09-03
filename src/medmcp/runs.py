"""Replay runs that outlive the browser: a record on disk, a task in the server.

Before this module a replay lived and died with its WebSocket: the socket *was*
the run, so a page reload, a laptop lid, or a flaky proxy aborted a two-hour
cohort roll-out at whatever step it was on, and the only trace afterwards was
whatever the browser still had in memory. That is the wrong shape for imaging
pipelines, where a single step can take ten minutes and a batch an afternoon.

A run is now a first-class thing with an id:

- :class:`RunRecord` is the durable form — one JSON file per run under
  ``.vibe/runs/`` (the same atomic-write discipline as the other state files),
  updated after every step so a crash mid-run leaves an honest partial record.
- :class:`RunManager` owns the live ones. It starts a run as a background task,
  fans its progress frames out to any number of attached sockets, and keeps
  running when the last one goes away. Stopping is an explicit request, never a
  side effect of a disconnect.

The record's frames are reconstructible from the record itself
(:meth:`RunRecord.frames`), so attaching to a finished run and attaching to a
live one look identical to the client.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from medmcp import replay
from medmcp.acp import VIBE_HOME
from medmcp.workflow import Recipe

JsonDict = dict[str, Any]

log = logging.getLogger(__name__)

RUNS_DIR: Path = VIBE_HOME / "runs"

RunStatus = str  # "running" | "done" | "failed" | "cancelled"

# How many output files an item reports at most (a prefix can fan out).
OUTPUT_FILES_LIMIT: int = 24


def _now_iso() -> str:
    # Full precision: a step's finish time is compared against file mtimes, and a
    # second-resolution stamp would call an input written in the same second
    # "newer than the step" and throw away a perfectly good cache entry.
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _iso_to_ts(value: str) -> float:
    """Epoch seconds for an ISO timestamp we wrote; 0 for anything unparsable."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def new_run_id() -> str:
    """A run id that sorts chronologically and cannot collide within a second."""
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


# ── The durable record ───────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """One executed step of one item."""

    index: int
    server: str
    tool: str
    arguments: JsonDict
    ok: bool
    error: str | None = None
    produced: dict[str, str] = field(default_factory=dict[str, str])
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> JsonDict:
        """Plain-dict form for the JSON record."""
        return {
            "index": self.index,
            "server": self.server,
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "error": self.error,
            "produced": self.produced,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> StepRecord:
        """Rebuild from the JSON record; missing fields take their defaults."""
        produced = cast("JsonDict", data.get("produced") or {})
        return cls(
            index=int(data.get("index", 0)),
            server=str(data.get("server", "")),
            tool=str(data.get("tool", "")),
            arguments=cast("JsonDict", data.get("arguments") or {}),
            ok=bool(data.get("ok", False)),
            error=cast("str | None", data.get("error")),
            produced={str(k): str(v) for k, v in produced.items()},
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
        )


@dataclass
class ItemRecord:
    """One input binding of a run and what happened to it."""

    inputs: dict[str, str]
    ok: bool | None = None
    """``None`` until the item has finished (still running, or never reached)."""
    error: str | None = None
    steps: list[StepRecord] = field(default_factory=list[StepRecord])

    @property
    def outputs(self) -> list[str]:
        """Every value the item's steps produced, in step order (raw, for chaining)."""
        return [v for s in self.steps for v in s.produced.values()]

    def files(self, *, limit: int = OUTPUT_FILES_LIMIT) -> list[str]:
        """The output *files* the item left behind — what a person wants to open.

        A tool's structured result names more than its outputs: it echoes its
        inputs, returns a template it read, or a *prefix* several files hang
        off. All of those are worth recording for chaining, none are files to
        open, so this keeps only produced values that are not one of the step's
        own arguments and that exist on disk as a file — and expands a prefix
        to the files it names, so a registration's transforms show up under it.
        """
        out: list[str] = []
        seen: set[str] = set()
        for step in self.steps:
            inputs = set(_string_values(step.arguments))
            for value in step.produced.values():
                if value in inputs or not value:
                    continue
                for path in _files_for(value):
                    if path not in seen:
                        seen.add(path)
                        out.append(path)
                    if len(out) >= limit:
                        return out
        return out

    def to_dict(self) -> JsonDict:
        """Plain-dict form for the JSON record."""
        return {
            "inputs": self.inputs,
            "ok": self.ok,
            "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> ItemRecord:
        """Rebuild from the JSON record."""
        inputs = cast("JsonDict", data.get("inputs") or {})
        steps = cast("list[JsonDict]", data.get("steps") or [])
        return cls(
            inputs={str(k): str(v) for k, v in inputs.items()},
            ok=cast("bool | None", data.get("ok")),
            error=cast("str | None", data.get("error")),
            steps=[StepRecord.from_dict(s) for s in steps],
        )


@dataclass
class RunRecord:
    """A replay run: which workflow, on which inputs, and how far it got."""

    id: str
    workflow: str
    steps_per_item: int
    items: list[ItemRecord]
    status: RunStatus = "running"
    started_at: str = field(default_factory=_now_iso)
    finished_at: str = ""
    error: str | None = None

    @property
    def finished(self) -> bool:
        """Whether the run has reached a terminal status."""
        return self.status != "running"

    def summary(self) -> JsonDict:
        """The list-view shape: counts and times, no per-step detail."""
        done = [i for i in self.items if i.ok is not None]
        return {
            "id": self.id,
            "workflow": self.workflow,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "total": len(self.items),
            "succeeded": sum(1 for i in done if i.ok),
            "failed": sum(1 for i in done if not i.ok),
            "steps_per_item": self.steps_per_item,
        }

    def to_dict(self) -> JsonDict:
        """The full JSON record: the summary plus every item and step."""
        return {
            **self.summary(),
            "items": [i.to_dict() for i in self.items],
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> RunRecord:
        """Rebuild from the JSON record."""
        items = cast("list[JsonDict]", data.get("items") or [])
        return cls(
            id=str(data.get("id", "")),
            workflow=str(data.get("workflow", "")),
            steps_per_item=int(data.get("steps_per_item", 0)),
            items=[ItemRecord.from_dict(i) for i in items],
            status=str(data.get("status", "running")),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            error=cast("str | None", data.get("error")),
        )

    def frames(self) -> list[JsonDict]:
        """The streaming frames a client attached from the start would have seen.

        Derived from the record rather than logged separately, so there is one
        source of truth and a finished run can be "attached" exactly like a live
        one.
        """
        out: list[JsonDict] = [
            {
                "type": "started",
                "run_id": self.id,
                "workflow": self.workflow,
                "total": len(self.items),
                "steps_per_item": self.steps_per_item,
                "started_at": self.started_at,
                "runs": [i.inputs for i in self.items],
            }
        ]
        for index, item in enumerate(self.items):
            for step in item.steps:
                out.append(step_frame(index, step))
            if item.ok is not None:
                out.append(item_frame(index, item))
        if self.finished:
            out.append(self.result_frame())
        return out

    def result_frame(self) -> JsonDict:
        """The terminal frame (``ok`` only for a run where every item succeeded)."""
        return {
            "type": "result",
            "status": self.status,
            "ok": self.status == "done",
            "error": self.error,
            "outputs": [v for i in self.items for v in i.files()],
            "finished_at": self.finished_at,
        }


def _string_values(value: object) -> list[str]:
    """Every string anywhere inside a (possibly nested) argument value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in cast("JsonDict", value).values() for s in _string_values(v)]
    if isinstance(value, list):
        return [s for v in cast("list[Any]", value) for s in _string_values(v)]
    return []


def _files_for(value: str) -> list[str]:
    """The files a produced value stands for: itself, or what a prefix expands to."""
    try:
        path = Path(value)
        if path.is_file():
            return [value]
        if path.is_dir() or not path.parent.is_dir():
            return []
        matches = sorted(p for p in path.parent.glob(f"{path.name}*") if p.is_file())
        return [str(p) for p in matches]
    except OSError:
        return []


def step_frame(item: int, step: StepRecord) -> JsonDict:
    """The ``step`` frame for one finished step of *item*."""
    return {
        "type": "step",
        "item": item,
        "index": step.index,
        "server": step.server,
        "tool": step.tool,
        "ok": step.ok,
        "error": step.error,
        "produced": step.produced,
        "started_at": step.started_at,
        "finished_at": step.finished_at,
    }


def item_frame(index: int, item: ItemRecord) -> JsonDict:
    """The ``item_result`` frame for a finished item."""
    return {
        "type": "item_result",
        "item": index,
        "ok": bool(item.ok),
        "error": item.error,
        "outputs": item.files(),
    }


# ── Storage ──────────────────────────────────────────────────────────────────


def _runs_root(root: Path | None) -> Path:
    return root if root is not None else RUNS_DIR


def run_path(run_id: str, *, root: Path | None = None) -> Path:
    """Where *run_id*'s record lives."""
    return _runs_root(root) / f"{run_id}.json"


def save_run(record: RunRecord, *, root: Path | None = None) -> None:
    """Write the record atomically (a reader never sees a half-written file)."""
    path = run_path(record.id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def load_run(run_id: str, *, root: Path | None = None) -> RunRecord | None:
    """Read one record; ``None`` when absent or unreadable."""
    path = run_path(run_id, root=root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return RunRecord.from_dict(cast("JsonDict", data)) if isinstance(data, dict) else None


def list_runs(
    *, workflow: str | None = None, limit: int = 50, root: Path | None = None
) -> list[RunRecord]:
    """Past and present runs, newest first; unreadable files are skipped."""
    base = _runs_root(root)
    if not base.is_dir():
        return []
    out: list[RunRecord] = []
    for path in sorted(base.glob("*.json"), reverse=True):
        record = load_run(path.stem, root=root)
        if record is None or (workflow is not None and record.workflow != workflow):
            continue
        out.append(record)
        if len(out) >= limit:
            break
    return out


def delete_run(run_id: str, *, root: Path | None = None) -> bool:
    """Remove a record; ``False`` when there was none."""
    try:
        run_path(run_id, root=root).unlink()
    except FileNotFoundError:
        return False
    return True


def reconcile_interrupted(*, root: Path | None = None) -> int:
    """Mark runs left ``running`` by a previous server process as failed.

    A record is only ``running`` while a task in *this* process drives it; on
    start-up there is none, so any such record was cut off by a crash or a
    restart. Left alone it would look live forever. Returns how many were fixed.
    """
    fixed = 0
    for record in list_runs(limit=10_000, root=root):
        if record.status != "running":
            continue
        record.status = "failed"
        record.error = "interrupted: the workspace server stopped while it was running"
        record.finished_at = record.finished_at or _now_iso()
        for item in record.items:
            if item.ok is None and item.steps:
                item.ok = False
                item.error = record.error
        save_run(record, root=root)
        fixed += 1
    return fixed


# ── Live runs ────────────────────────────────────────────────────────────────

Frame = JsonDict
_END: Frame = {}  # sentinel put on a subscriber queue when the run is over


@dataclass
class _LiveRun:
    record: RunRecord
    task: asyncio.Task[None]
    frames: list[Frame] = field(default_factory=list[Frame])
    subscribers: set[asyncio.Queue[Frame]] = field(default_factory=set["asyncio.Queue[Frame]"])
    # The step currently being executed, if any: (item, index, server, tool,
    # started_at) — the time travels with it so a late attach shows the real
    # elapsed time of the step, not the time since attaching.
    current: tuple[int, int, str, str, str] | None = None


class RunManager:
    """Owns the runs in flight in this process and fans their frames out.

    The manager is deliberately thin over :func:`medmcp.replay.run_batch`: it
    adds identity (a run id), durability (the record is saved after every step)
    and detachment (sockets subscribe and unsubscribe; the task neither knows nor
    cares). Callers must still do the confirmation the engine's docstring asks
    for — nothing here adds a permission prompt.
    """

    def __init__(self, *, root: Path | None = None) -> None:
        """Create an empty manager; *root* overrides the records directory (tests)."""
        self._root = root
        self._live: dict[str, _LiveRun] = {}

    # ── lifecycle ──

    def start(
        self,
        *,
        recipe: Recipe,
        runs: list[dict[str, str]],
        servers: list[JsonDict],
        cwd: str | None,
        tool_timeout_sec: float = replay.DEFAULT_TOOL_TIMEOUT_SEC,
        on_finished: Callable[[RunRecord], None] | None = None,
    ) -> RunRecord:
        """Begin a run as a background task and return its (already saved) record."""
        record = RunRecord(
            id=new_run_id(),
            workflow=recipe.name,
            steps_per_item=len(recipe.steps),
            items=[ItemRecord(inputs=dict(r)) for r in runs],
        )
        save_run(record, root=self._root)
        live = _LiveRun(record=record, task=cast("asyncio.Task[None]", None))
        live.frames.append(record.frames()[0])  # the "started" frame
        self._live[record.id] = live
        live.task = asyncio.create_task(
            self._execute(live, recipe, runs, servers, cwd, tool_timeout_sec, on_finished),
            name=f"replay-run-{record.id}",
        )
        return record

    async def _execute(
        self,
        live: _LiveRun,
        recipe: Recipe,
        runs: list[dict[str, str]],
        servers: list[JsonDict],
        cwd: str | None,
        tool_timeout_sec: float,
        on_finished: Callable[[RunRecord], None] | None,
    ) -> None:
        record = live.record

        async def _on_step_start(item: int, index: int, server: str, tool: str) -> None:
            live.current = (item, index, server, tool, _now_iso())
            self._broadcast(live, _step_started_frame(*live.current))

        async def _on_step(item: int, sr: replay.StepResult) -> None:
            started = live.current[4] if live.current is not None else ""
            live.current = None
            step = StepRecord(
                index=sr.index,
                server=sr.server,
                tool=sr.tool,
                arguments=sr.arguments,
                ok=sr.ok,
                error=sr.error,
                produced=dict(sr.produced),
                started_at=started,
                finished_at=_now_iso(),
            )
            record.items[item].steps.append(step)
            self._save(record)
            self._broadcast(live, step_frame(item, step))

        async def _on_item(item: int, res: replay.ReplayResult) -> None:
            entry = record.items[item]
            entry.ok = res.ok
            entry.error = res.error
            self._save(record)
            self._broadcast(live, item_frame(item, entry))

        try:
            results = await replay.run_batch(
                recipe,
                runs,
                servers=servers,
                cwd=cwd,
                tool_timeout_sec=tool_timeout_sec,
                on_step=_on_step,
                on_item=_on_item,
                on_step_start=_on_step_start,
            )
            failed = sum(1 for r in results if not r.ok)
            record.status = "done" if failed == 0 else "failed"
            record.error = None if failed == 0 else f"{failed} of {len(runs)} item(s) failed"
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.error = "stopped"
            for entry in record.items:
                if entry.ok is None and entry.steps:
                    entry.ok = False
                    entry.error = "stopped"
            raise
        except Exception as exc:  # an engine bug must still leave an honest record
            log.exception("replay run %s crashed", record.id)
            record.status = "failed"
            record.error = f"engine error: {exc}"
        finally:
            live.current = None
            record.finished_at = _now_iso()
            self._save(record)
            self._broadcast(live, record.result_frame())
            for queue in list(live.subscribers):
                queue.put_nowait(_END)
            self._live.pop(record.id, None)
            if on_finished is not None:
                with contextlib.suppress(Exception):
                    on_finished(record)

    def _save(self, record: RunRecord) -> None:
        try:
            save_run(record, root=self._root)
        except OSError:  # a full disk must not kill a run that is otherwise fine
            log.warning("could not persist run %s", record.id, exc_info=True)

    def _broadcast(self, live: _LiveRun, frame: Frame) -> None:
        live.frames.append(frame)
        for queue in list(live.subscribers):
            queue.put_nowait(frame)

    # ── observation ──

    def is_live(self, run_id: str) -> bool:
        """Whether *run_id* is being driven by this process right now."""
        return run_id in self._live

    def live_ids(self) -> list[str]:
        """The ids of every run in flight."""
        return list(self._live)

    def attach(self, run_id: str) -> tuple[list[Frame], asyncio.Queue[Frame] | None] | None:
        """Everything that happened so far, plus a queue for what follows.

        The queue is ``None`` for a finished run (its frames are complete). A
        finished run's frames are read from disk, so a run from a previous server
        process can be opened the same way. ``None`` altogether means no such run.
        """
        live = self._live.get(run_id)
        if live is not None:
            queue: asyncio.Queue[Frame] = asyncio.Queue()
            live.subscribers.add(queue)
            frames = list(live.frames)
            if live.current is not None:
                frames.append(_step_started_frame(*live.current))
            return frames, queue
        record = load_run(run_id, root=self._root)
        if record is None:
            return None
        return record.frames(), None

    def detach(self, run_id: str, queue: asyncio.Queue[Frame]) -> None:
        """Stop feeding *queue*; the run itself is unaffected."""
        live = self._live.get(run_id)
        if live is not None:
            live.subscribers.discard(queue)

    async def cancel(self, run_id: str) -> bool:
        """Stop a live run; the in-flight tool call is killed with its stack."""
        live = self._live.get(run_id)
        if live is None:
            return False
        live.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await live.task
        return True

    async def shutdown(self) -> None:
        """Cancel every live run (server shutdown)."""
        for run_id in list(self._live):
            await self.cancel(run_id)


def _step_started_frame(item: int, index: int, server: str, tool: str, started_at: str) -> Frame:
    return {
        "type": "step_started",
        "item": item,
        "index": index,
        "server": server,
        "tool": tool,
        "started_at": started_at,
    }


def is_end(frame: Frame) -> bool:
    """Whether a queued frame is the end-of-run sentinel rather than a real frame."""
    return frame is _END


def elapsed_seconds(record: RunRecord) -> float:
    """Wall-clock seconds the run took (or has taken so far)."""
    start = _iso_to_ts(record.started_at)
    end = _iso_to_ts(record.finished_at) if record.finished_at else time.time()
    return max(0.0, end - start) if start else 0.0
