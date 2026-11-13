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

A run is also evidence. Beside the index entry ``.vibe/runs/<id>.json`` (the
progress, small, rewritten after every step) each run has a folder
``.vibe/runs/<id>/`` that stands on its own:

- ``run.json`` — the full record: environment (core build, each stack's image
  and digest), who confirmed it, and per item its label, bound inputs, every
  step's arguments, timing, error and **structured result**, and the SHA-256 of
  each input and output file. The per-item section is that subject's provenance.
- ``recipe.yaml`` — the workflow exactly as it ran (it can be edited or deleted
  afterwards; the run keeps its own copy).
- ``manifest.csv`` — the cohort plan or selection the items came from, as given.

See :mod:`medmcp.runrecord` for how the environment and checksums are gathered.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import secrets
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from medmcp import pathcheck, replay, runrecord, workflow
from medmcp.acp import VIBE_HOME
from medmcp.statefile import atomic_write_json, read_json
from medmcp.workflow import Recipe

JsonDict = dict[str, Any]

log = logging.getLogger(__name__)

RUNS_DIR: Path = VIBE_HOME / "runs"

RunStatus = str  # "running" | "done" | "failed" | "cancelled"

# How many output files an item reports at most (a prefix can fan out).
OUTPUT_FILES_LIMIT: int = 24

# Version of the run-folder record (``run.json``); bump on a breaking change.
RECORD_VERSION: int = 1

RUN_FILE = "run.json"
RECIPE_FILE = "recipe.yaml"
MANIFEST_FILE = "manifest.csv"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _seconds_between(started: str, finished: str) -> float | None:
    """Elapsed seconds between two ISO timestamps, ``None`` when either is missing."""
    try:
        delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    except ValueError:
        return None
    return round(delta.total_seconds(), 3)


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
    result: JsonDict | None = None
    """The tool's structured result, as it answered. Kept in the run folder only:
    a measuring tool's whole point is this answer, and the index entry that is
    rewritten after every step should stay small."""

    def to_dict(self, *, full: bool = False) -> JsonDict:
        """Plain-dict form: the index entry, or with ``full`` the run-folder record."""
        out: JsonDict = {
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
        if full:
            out["duration_sec"] = _seconds_between(self.started_at, self.finished_at)
            out["result"] = self.result
        return out

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
            result=cast("JsonDict | None", data.get("result")),
        )


@dataclass
class ItemRecord:
    """One input binding of a run and what happened to it."""

    inputs: dict[str, str]
    ok: bool | None = None
    """``None`` until the item has finished (still running, or never reached)."""
    error: str | None = None
    steps: list[StepRecord] = field(default_factory=list[StepRecord])
    label: str = ""
    """Who the item is — ``subject[/session]`` from the manifest, "" for hand-picked files."""
    input_files: list[JsonDict] = field(default_factory=list[JsonDict])
    """Checksums of the files the item read (see :func:`medmcp.runrecord.checksum`)."""
    output_files: list[JsonDict] = field(default_factory=list[JsonDict])
    """Checksums of the files the item wrote."""

    @property
    def outputs(self) -> list[str]:
        """Every value the item's steps produced, in step order (raw, for chaining)."""
        return [v for s in self.steps for v in s.produced.values()]

    def files(self, *, limit: int = OUTPUT_FILES_LIMIT) -> list[str]:
        """The output *files* the item left behind — what a person wants to open.

        A tool's structured result names more than its outputs: it echoes its
        inputs, returns a template it read, or a *prefix* several files hang
        off. So this keeps what a step produced unless it is one of the step's
        own arguments — except an argument the tool **writes to**: an
        ``output_path`` the caller chose is the output, whether or not the tool
        echoes it. Only files that exist count; a prefix expands to the files it
        names, and an output folder to the files that appeared in it during the
        step.
        """
        out: list[str] = []
        seen: set[str] = set()

        def _add(paths: list[str]) -> bool:
            for path in paths:
                if path not in seen:
                    seen.add(path)
                    out.append(path)
                if len(out) >= limit:
                    return True
            return False

        for step in self.steps:
            written: dict[str, workflow.OutputKind] = {}
            read: set[str] = set()
            for key, value in step.arguments.items():
                strings = _string_values(value)
                if pathcheck.classify_role(key) == "output":
                    written.update(dict.fromkeys(strings, workflow.output_kind(key)))
                else:
                    read.update(strings)
            candidates = [v for v in step.produced.values() if v and v not in read]
            candidates += [v for v in written if v not in candidates]
            for value in candidates:
                if written.get(value) == "dir":
                    found = _files_written_in(value, since=step.started_at)
                else:
                    found = _files_for(value)
                if _add(found):
                    return out
        return out

    def read_files(self) -> list[str]:
        """The files and folders the item read: its bound inputs and constant paths.

        Anything a step of this item produced is an intermediate, not an input.
        """
        made = {v for s in self.steps for v in s.produced.values()}
        out: list[str] = []
        for step in self.steps:
            for key, value in step.arguments.items():
                if pathcheck.classify_role(key) == "output":
                    continue
                for text in _string_values(value):
                    if text in made or text in out or not pathcheck.looks_like_path(text):
                        continue
                    with contextlib.suppress(OSError):
                        if Path(text).exists():
                            out.append(text)
        return out

    def to_dict(self, *, full: bool = False) -> JsonDict:
        """Plain-dict form: the index entry, or with ``full`` the run-folder record."""
        out: JsonDict = {
            "inputs": self.inputs,
            "ok": self.ok,
            "error": self.error,
            "steps": [s.to_dict(full=full) for s in self.steps],
        }
        if self.label:
            out["label"] = self.label
        if full:
            out["outputs"] = self.files()
            out["input_files"] = self.input_files
            out["output_files"] = self.output_files
        return out

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
            label=str(data.get("label", "")),
            input_files=cast("list[JsonDict]", data.get("input_files") or []),
            output_files=cast("list[JsonDict]", data.get("output_files") or []),
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
    recipe: JsonDict = field(default_factory=dict[str, Any])
    """The workflow as it ran (``Recipe.to_dict()``)."""
    environment: JsonDict = field(default_factory=dict[str, Any])
    """Core build and stack images (:func:`medmcp.runrecord.environment`)."""
    manifest: JsonDict = field(default_factory=dict[str, Any])
    """The plan or selection the items came from: its path, checksum and copy."""
    confirmation: JsonDict = field(default_factory=dict[str, Any])
    """Who started the run and whether these exact items had been previewed."""
    output_root: str = ""

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

    def to_dict(self, *, full: bool = False) -> JsonDict:
        """The index entry (summary, items, steps), or with ``full`` everything.

        The full form is what the run folder holds and what an export hands out:
        it adds the recipe, the environment, the manifest reference, the
        confirmation, and per item the checksums and each step's result.
        """
        out: JsonDict = {
            **self.summary(),
            "items": [i.to_dict(full=full) for i in self.items],
        }
        if full:
            out["record_version"] = RECORD_VERSION
            out["recipe"] = self.recipe
            out["environment"] = self.environment
            out["manifest"] = self.manifest
            out["confirmation"] = self.confirmation
            out["output_root"] = self.output_root
        return out

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
            recipe=cast("JsonDict", data.get("recipe") or {}),
            environment=cast("JsonDict", data.get("environment") or {}),
            manifest=cast("JsonDict", data.get("manifest") or {}),
            confirmation=cast("JsonDict", data.get("confirmation") or {}),
            output_root=str(data.get("output_root", "")),
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
                "labels": [i.label for i in self.items],
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


def _files_written_in(folder: str, *, since: str) -> list[str]:
    """Files directly in an output *folder* that were written at or after *since*.

    An output folder may be shared (results beside the scans), so everything in
    it is not this step's output; what changed while the step ran is. Without a
    usable start time nothing is claimed.
    """
    try:
        started = datetime.fromisoformat(since).timestamp() - 1.0
        path = Path(folder)
        if not path.is_dir():
            return []
        return [
            str(p)
            for p in sorted(path.iterdir())
            if p.is_file() and not p.is_symlink() and p.stat().st_mtime >= started
        ]
    except (OSError, ValueError):
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
    atomic_write_json(run_path(record.id, root=root), record.to_dict())


def load_run(run_id: str, *, root: Path | None = None) -> RunRecord | None:
    """Read one record; ``None`` when absent or unreadable."""
    data = read_json(run_path(run_id, root=root))
    return RunRecord.from_dict(data) if data is not None else None


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


def run_folder(run_id: str, *, root: Path | None = None) -> Path:
    """The folder holding *run_id*'s self-contained record (not created)."""
    return _runs_root(root) / run_id


def save_run_folder(record: RunRecord, *, root: Path | None = None) -> None:
    """Write the full record into the run's folder, atomically."""
    folder = run_folder(record.id, root=root)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_json(folder / RUN_FILE, record.to_dict(full=True))


def load_full(run_id: str, *, root: Path | None = None) -> RunRecord | None:
    """One run's full record; for a run from before folders, what the index knows.

    The index entry is written after every step and the folder after every item,
    so after a crash the index can be a few steps ahead. Status and timing are
    therefore the index's, and an item the index knows more about is taken from
    it (without step results, which only the folder had time to record).
    """
    index = load_run(run_id, root=root)
    if index is None:
        return None
    data = read_json(run_folder(run_id, root=root) / RUN_FILE)
    if data is None:
        return index
    full = RunRecord.from_dict(data)
    full.status, full.finished_at, full.error = index.status, index.finished_at, index.error
    items: list[ItemRecord] = []
    for position, current in enumerate(index.items):
        stored = full.items[position] if position < len(full.items) else None
        if (
            stored is not None
            and stored.ok == current.ok
            and len(stored.steps) == len(current.steps)
        ):
            items.append(stored)
        else:
            items.append(current)
    full.items = items
    return full


def delete_run(run_id: str, *, root: Path | None = None) -> bool:
    """Remove a record and its folder; ``False`` when there was none."""
    shutil.rmtree(run_folder(run_id, root=root), ignore_errors=True)
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
        if run_folder(record.id, root=root).is_dir():
            full = load_full(record.id, root=root)
            if full is not None:
                save_run_folder(full, root=root)
        fixed += 1
    return fixed


# ── Live runs ────────────────────────────────────────────────────────────────

# Blocking record-keeping (docker inspect, file hashing) goes through this one
# name so a test can run it inline: a test client's event loop does not outlive
# the socket the way the server's does.
_to_thread = asyncio.to_thread

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
        # Runs whose result is out but whose record is still being completed.
        self._closing: dict[str, asyncio.Task[None]] = {}

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
        confine_to: Path | None = None,
        labels: list[str] | None = None,
        manifest_path: Path | None = None,
        confirmation: JsonDict | None = None,
        output_root: str = "",
        image_identity: runrecord.ImageIdentity | None = None,
    ) -> RunRecord:
        """Begin a run as a background task and return its (already saved) record.

        *confine_to* is handed to :func:`medmcp.replay.run_batch`: a step naming a
        path outside it fails instead of running. *labels* name the items
        (``subject[/session]``), *manifest_path* is the plan or selection they
        came from (copied into the run folder), *confirmation* says who started
        the run, and *image_identity* resolves each stack's image digest for the
        record's environment.
        """
        names = labels or []
        record = RunRecord(
            id=new_run_id(),
            workflow=recipe.name,
            steps_per_item=len(recipe.steps),
            items=[
                ItemRecord(inputs=dict(r), label=names[i] if i < len(names) else "")
                for i, r in enumerate(runs)
            ],
            recipe=recipe.to_dict(),
            confirmation=dict(confirmation or {}),
            output_root=output_root,
        )
        save_run(record, root=self._root)
        live = _LiveRun(record=record, task=cast("asyncio.Task[None]", None))
        live.frames.append(record.frames()[0])  # the "started" frame
        self._live[record.id] = live
        live.task = asyncio.create_task(
            self._execute(
                live,
                recipe,
                runs,
                servers,
                cwd,
                tool_timeout_sec,
                on_finished,
                confine_to,
                manifest_path,
                image_identity,
            ),
            name=f"replay-run-{record.id}",
        )
        self._closing[record.id] = live.task
        return record

    def _open_folder(
        self,
        record: RunRecord,
        recipe: Recipe,
        servers: list[JsonDict],
        manifest_path: Path | None,
        image_identity: runrecord.ImageIdentity | None,
    ) -> None:
        """Write what is known before the first step: recipe, manifest, environment.

        Blocking (file copies, one ``docker image inspect`` per stack), so the
        run calls it in a thread. Each part is best-effort: a record with a gap
        that says so beats a run that did not start.
        """
        folder = run_folder(record.id, root=self._root)
        folder.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            (folder / RECIPE_FILE).write_text(
                yaml.safe_dump(recipe.to_dict(), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        if manifest_path is not None:
            record.manifest = {"path": str(manifest_path), "sha256": None, "copy": None}
            try:
                data = manifest_path.read_bytes()
                (folder / MANIFEST_FILE).write_bytes(data)
                record.manifest["sha256"] = hashlib.sha256(data).hexdigest()
                record.manifest["copy"] = MANIFEST_FILE
            except OSError as exc:
                record.manifest["error"] = f"could not be copied: {exc.strerror or exc}"
        try:
            record.environment = runrecord.environment(
                recipe, servers, image_identity=image_identity
            )
        except Exception as exc:  # a docker hiccup must not stop the run
            log.warning("could not record the environment of run %s", record.id, exc_info=True)
            record.environment = {"error": str(exc)}

    async def _checksum_item(self, record: RunRecord, index: int) -> None:
        """Hash what item *index* read and wrote, off the event loop."""
        entry = record.items[index]

        def _work() -> tuple[list[JsonDict], list[JsonDict]]:
            return (
                [runrecord.checksum(p) for p in entry.read_files()],
                [runrecord.checksum(p) for p in entry.files()],
            )

        try:
            entry.input_files, entry.output_files = await _to_thread(_work)
        except Exception:  # checksums are evidence, not a reason to lose the run
            log.warning("could not checksum item %d of run %s", index, record.id, exc_info=True)

    async def _execute(
        self,
        live: _LiveRun,
        recipe: Recipe,
        runs: list[dict[str, str]],
        servers: list[JsonDict],
        cwd: str | None,
        tool_timeout_sec: float,
        on_finished: Callable[[RunRecord], None] | None,
        confine_to: Path | None,
        manifest_path: Path | None,
        image_identity: runrecord.ImageIdentity | None,
    ) -> None:
        record = live.record
        # Checksums run beside the items that follow rather than between them;
        # the record is complete once these have been awaited at the end.
        hashing: list[asyncio.Task[None]] = []

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
                result=sr.output,
            )
            record.items[item].steps.append(step)
            self._save(record)
            self._broadcast(live, step_frame(item, step))

        async def _on_item(item: int, res: replay.ReplayResult) -> None:
            entry = record.items[item]
            entry.ok = res.ok
            entry.error = res.error
            self._save(record)
            self._save_folder(record)
            self._broadcast(live, item_frame(item, entry))
            if entry.steps:
                hashing.append(asyncio.create_task(self._checksum_item(record, item)))

        try:
            await _to_thread(
                lambda: self._open_folder(record, recipe, servers, manifest_path, image_identity)
            )
            self._save_folder(record)
            results = await replay.run_batch(
                recipe,
                runs,
                servers=servers,
                cwd=cwd,
                tool_timeout_sec=tool_timeout_sec,
                on_step=_on_step,
                on_item=_on_item,
                on_step_start=_on_step_start,
                confine_to=confine_to,
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
            self._save_folder(record)
            self._broadcast(live, record.result_frame())
            for queue in list(live.subscribers):
                queue.put_nowait(_END)
            self._live.pop(record.id, None)
            # The run is over for anyone watching; its checksums may still be
            # in flight. Wait for them, then write the record a last time.
            try:
                if hashing:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(asyncio.gather(*hashing, return_exceptions=True))
                    self._save_folder(record)
            finally:
                self._closing.pop(record.id, None)
            if on_finished is not None:
                with contextlib.suppress(Exception):
                    on_finished(record)

    def _save_folder(self, record: RunRecord) -> None:
        try:
            save_run_folder(record, root=self._root)
        except OSError:
            log.warning("could not write the folder of run %s", record.id, exc_info=True)

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

    def full(self, run_id: str) -> RunRecord | None:
        """The full record of *run_id*: the live one while it runs, else from disk."""
        live = self._live.get(run_id)
        if live is not None:
            return live.record
        return load_full(run_id, root=self._root)

    async def wait(self, run_id: str) -> None:
        """Return once *run_id* is over and its record is complete (no-op if it is not live)."""
        task = self._closing.get(run_id)
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

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
