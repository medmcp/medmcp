"""Tests for persisted replay runs and the detached run manager."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from medmcp import replay, runs
from medmcp.workflow import Recipe, RecipeStep, WorkflowInput

# pyright: reportPrivateUsage=false

JsonDict = dict[str, Any]


def _recipe() -> Recipe:
    return Recipe(
        name="strip-register",
        description="",
        inputs=[WorkflowInput(name="in_1", example="/data/t1.nii.gz")],
        steps=[
            RecipeStep(
                server="medmcp-neuro",
                tool="skull_strip",
                arguments={"input_path": "{{in_1}}"},
                produces={"brain_path": "step1.brain_path"},
            ),
            RecipeStep(
                server="medmcp-neuro",
                tool="register_to_template",
                arguments={"input_path": "{{step1.brain_path}}"},
                produces={"registered_path": "step2.registered_path"},
            ),
        ],
    )


def _servers() -> list[JsonDict]:
    return [{"name": "medmcp-neuro", "command": "x", "args": []}]


def _fake_caller(calls: list[tuple[str, str, JsonDict]], outputs: dict[str, str]):  # noqa: ANN202
    @contextlib.asynccontextmanager
    async def _cm(*_args: object, **_kwargs: object) -> AsyncGenerator[replay.ToolCaller]:
        async def _call(
            server: str, tool: str, args: JsonDict
        ) -> tuple[bool, JsonDict, str | None]:
            calls.append((server, tool, args))
            if tool == "skull_strip":
                return True, {"brain_path": outputs["brain"]}, None
            return True, {"registered_path": outputs["registered"]}, None

        yield _call

    return _cm


# ── the record ────────────────────────────────────────────────────────────────


def test_record_round_trips_and_derives_frames(tmp_path: Path) -> None:
    """A saved record reloads identically and reconstructs the frames a client saw."""
    out = tmp_path / "o1.nii.gz"
    out.write_bytes(b"x")
    record = runs.RunRecord(
        id="20260101-000000-abc123",
        workflow="w",
        steps_per_item=2,
        items=[runs.ItemRecord(inputs={"in_1": "/a"}), runs.ItemRecord(inputs={"in_1": "/b"})],
    )
    record.items[0].steps.append(
        runs.StepRecord(1, "s", "t", {"x": "/a"}, True, produced={"step1.p": str(out)})
    )
    record.items[0].steps.append(runs.StepRecord(2, "s", "u", {"x": "/o1"}, True, produced={}))
    record.items[0].ok = True
    runs.save_run(record, root=tmp_path)

    loaded = runs.load_run(record.id, root=tmp_path)
    assert loaded is not None
    assert loaded.to_dict() == record.to_dict()

    kinds = [f["type"] for f in loaded.frames()]
    assert kinds == ["started", "step", "step", "item_result"]  # not finished: no result frame
    loaded.status = "done"
    assert loaded.frames()[-1]["type"] == "result"
    assert loaded.frames()[-1]["outputs"] == [str(out)]
    assert loaded.summary()["succeeded"] == 1


def test_item_files_keeps_only_output_files(tmp_path: Path) -> None:
    """Echoed inputs, absent paths and directories are not outputs; a prefix expands."""
    scan = tmp_path / "t1.nii.gz"
    scan.write_bytes(b"x")
    brain = tmp_path / "t1_brain.nii.gz"
    brain.write_bytes(b"y")
    for suffix in ("0GenericAffine.mat", "1Warp.nii.gz"):
        (tmp_path / f"t1_to_MNI_{suffix}").write_bytes(b"z")
    item = runs.ItemRecord(inputs={"in_1": str(scan)})
    item.steps.append(
        runs.StepRecord(
            1,
            "neuro",
            "skull_strip",
            {"input_path": str(scan)},
            True,
            produced={
                "step1.input_path": str(scan),  # the tool echoing its input
                "step1.brain_path": str(brain),
                "step1.output_dir": str(tmp_path),  # a directory
                "step1.template_path": "/root/.cache/template.nii.gz",  # not on this disk
                "step1.transform_prefix": str(tmp_path / "t1_to_MNI_"),  # a prefix
            },
        )
    )
    assert item.files() == [
        str(brain),
        str(tmp_path / "t1_to_MNI_0GenericAffine.mat"),
        str(tmp_path / "t1_to_MNI_1Warp.nii.gz"),
    ]
    assert len(item.outputs) == 5  # the raw record keeps everything for chaining


def test_list_runs_newest_first_and_filtered(tmp_path: Path) -> None:
    """Listing sorts by id (chronological) descending and filters by workflow."""
    for rid, wf in [("20260101-000000-aaa", "w1"), ("20260102-000000-bbb", "w2")]:
        runs.save_run(
            runs.RunRecord(id=rid, workflow=wf, steps_per_item=1, items=[]), root=tmp_path
        )
    assert [r.id for r in runs.list_runs(root=tmp_path)] == [
        "20260102-000000-bbb",
        "20260101-000000-aaa",
    ]
    assert [r.workflow for r in runs.list_runs(workflow="w1", root=tmp_path)] == ["w1"]
    assert runs.delete_run("20260101-000000-aaa", root=tmp_path) is True
    assert runs.delete_run("20260101-000000-aaa", root=tmp_path) is False


def test_reconcile_marks_orphaned_running_records_failed(tmp_path: Path) -> None:
    """A record left 'running' by a dead process is closed out, not shown as live forever."""
    record = runs.RunRecord(id="20260101-000000-abc", workflow="w", steps_per_item=1, items=[])
    record.items.append(runs.ItemRecord(inputs={}, steps=[runs.StepRecord(1, "s", "t", {}, True)]))
    runs.save_run(record, root=tmp_path)
    assert runs.reconcile_interrupted(root=tmp_path) == 1
    fixed = runs.load_run(record.id, root=tmp_path)
    assert fixed is not None
    assert fixed.status == "failed"
    assert fixed.items[0].ok is False
    assert runs.reconcile_interrupted(root=tmp_path) == 0  # idempotent


# ── the run manager ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manager_runs_detached_and_records_every_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run survives its subscriber leaving; the record on disk tells the story."""
    calls: list[tuple[str, str, JsonDict]] = []
    monkeypatch.setattr(
        replay, "mcp_caller", _fake_caller(calls, {"brain": "/b", "registered": "/r"})
    )
    manager = runs.RunManager(root=tmp_path)

    record = manager.start(
        recipe=_recipe(),
        runs=[{"in_1": "/a"}, {"in_1": "/b"}],
        servers=_servers(),
        cwd=None,
    )
    assert manager.is_live(record.id)
    attached = manager.attach(record.id)
    assert attached is not None
    past, queue = attached
    assert past[0]["type"] == "started" and past[0]["run_id"] == record.id
    assert queue is not None
    manager.detach(record.id, queue)  # the browser goes away…

    for _ in range(100):  # …and the run finishes anyway
        if not manager.is_live(record.id):
            break
        await asyncio.sleep(0.01)
    assert not manager.is_live(record.id)
    assert len(calls) == 4

    saved = runs.load_run(record.id, root=tmp_path)
    assert saved is not None
    assert saved.status == "done"
    assert [i.ok for i in saved.items] == [True, True]
    assert saved.items[1].outputs == ["/b", "/r"]  # raw produced values, files or not
    # A late attach to the finished run replays the full story from disk.
    late = manager.attach(record.id)
    assert late is not None
    frames, late_queue = late
    assert late_queue is None
    assert [f["type"] for f in frames] == [
        "started",
        "step",
        "step",
        "item_result",
        "step",
        "step",
        "item_result",
        "result",
    ]


@pytest.mark.asyncio
async def test_manager_cancel_stops_the_run_and_marks_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel kills the in-flight step; the record says 'cancelled', not 'running'."""
    started = asyncio.Event()

    @contextlib.asynccontextmanager
    async def slow_caller(*_a: object, **_k: object) -> AsyncGenerator[replay.ToolCaller]:
        async def _call(
            server: str, tool: str, args: JsonDict
        ) -> tuple[bool, JsonDict, str | None]:
            started.set()
            await asyncio.sleep(60)
            return True, {}, None

        yield _call

    monkeypatch.setattr(replay, "mcp_caller", slow_caller)
    manager = runs.RunManager(root=tmp_path)
    record = manager.start(recipe=_recipe(), runs=[{"in_1": "/a"}], servers=_servers(), cwd=None)
    attached = manager.attach(record.id)
    assert attached is not None
    _past, queue = attached
    assert queue is not None
    await asyncio.wait_for(started.wait(), 2)
    assert (await queue.get())["type"] == "step_started"

    assert await manager.cancel(record.id) is True
    assert not manager.is_live(record.id)
    final = await queue.get()
    assert final["type"] == "result" and final["status"] == "cancelled"
    assert runs.is_end(await queue.get())
    saved = runs.load_run(record.id, root=tmp_path)
    assert saved is not None and saved.status == "cancelled"
    assert await manager.cancel(record.id) is False
