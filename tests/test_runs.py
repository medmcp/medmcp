"""Tests for persisted replay runs and the detached run manager."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from medmcp import replay, runs

from ._recipes import McpCallerFactory, neuro_servers, two_step_recipe

JsonDict = dict[str, Any]


def _fake_caller(
    calls: list[tuple[str, str, JsonDict]], outputs: dict[str, str]
) -> McpCallerFactory:
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
        recipe=two_step_recipe(),
        runs=[{"in_1": "/a"}, {"in_1": "/b"}],
        servers=neuro_servers(),
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
    record = manager.start(
        recipe=two_step_recipe(), runs=[{"in_1": "/a"}], servers=neuro_servers(), cwd=None
    )
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


# ── the run folder ────────────────────────────────────────────────────────────


def test_a_chosen_output_path_counts_as_an_output(tmp_path: Path) -> None:
    """A file the caller told the tool to write is listed even though it is an argument."""
    scan = tmp_path / "scan.nii.gz"
    scan.write_bytes(b"x")
    out = tmp_path / "scan_RAS.nii.gz"
    out.write_bytes(b"y")
    item = runs.ItemRecord(inputs={"in_1": str(scan), "in_2": str(out)})
    item.steps.append(
        runs.StepRecord(
            1,
            "s",
            "reorient",
            {"input_path": str(scan), "output_path": str(out)},
            True,
            produced={"step1.output_path": str(out)},
        )
    )
    assert item.files() == [str(out)]
    assert item.read_files() == [str(scan)]


def test_an_output_folder_lists_what_the_step_wrote(tmp_path: Path) -> None:
    """Only files that appeared during the step are claimed from a shared folder."""
    import os

    folder = tmp_path / "out"
    folder.mkdir()
    old = folder / "old.nii.gz"
    old.write_bytes(b"o")
    os.utime(old, (1_000_000_000, 1_000_000_000))
    new = folder / "mask.nii.gz"
    new.write_bytes(b"n")
    item = runs.ItemRecord(inputs={})
    item.steps.append(
        runs.StepRecord(
            1,
            "s",
            "segment",
            {"output_dir": str(folder)},
            True,
            started_at="2020-01-01T00:00:00+00:00",
        )
    )
    assert item.files() == [str(new)]


async def test_run_folder_is_a_self_contained_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recipe, manifest copy, environment, labels, results and checksums are all on disk."""
    import hashlib

    import yaml

    scan = tmp_path / "scan.nii.gz"
    scan.write_bytes(b"scan-bytes")
    brain = tmp_path / "brain.nii.gz"
    registered = tmp_path / "registered.nii.gz"
    brain.write_bytes(b"brain-bytes")
    registered.write_bytes(b"registered-bytes")
    manifest = tmp_path / "selection.csv"
    manifest.write_text("subject_id,path\nP1,scan.nii.gz\n")
    calls: list[tuple[str, str, JsonDict]] = []
    monkeypatch.setattr(
        replay,
        "mcp_caller",
        _fake_caller(calls, {"brain": str(brain), "registered": str(registered)}),
    )

    def identity(image: str) -> JsonDict:
        return {"digest": "sha256:feed", "id": "sha256:beef"}

    servers: list[JsonDict] = [
        {"name": "medmcp-neuro", "command": "docker", "args": ["run", "-i", "neuro:main"]}
    ]
    root = tmp_path / "runs"
    manager = runs.RunManager(root=root)
    record = manager.start(
        recipe=two_step_recipe(),
        runs=[{"in_1": str(scan)}],
        servers=servers,
        cwd=None,
        labels=["P1/ses-1"],
        manifest_path=manifest,
        confirmation={"by": "workspace user", "previewed": True},
        output_root="/ws/results/wf",
        image_identity=identity,
    )
    await manager.wait(record.id)

    folder = runs.run_folder(record.id, root=root)
    assert sorted(p.name for p in folder.iterdir()) == ["manifest.csv", "recipe.yaml", "run.json"]
    assert yaml.safe_load((folder / "recipe.yaml").read_text())["steps"][0]["tool"] == "skull_strip"
    assert (folder / "manifest.csv").read_text() == manifest.read_text()

    full = runs.load_full(record.id, root=root)
    assert full is not None
    data = full.to_dict(full=True)
    assert data["status"] == "done" and data["record_version"] == runs.RECORD_VERSION
    assert data["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert data["confirmation"]["previewed"] is True
    assert data["output_root"] == "/ws/results/wf"
    stack = data["environment"]["stacks"][0]
    assert stack["image"] == "neuro:main" and stack["digest"] == "sha256:feed"
    item = data["items"][0]
    assert item["label"] == "P1/ses-1"
    assert item["steps"][0]["result"] == {"brain_path": str(brain)}
    assert item["steps"][0]["duration_sec"] is not None
    assert item["input_files"] == [
        {
            "path": str(scan),
            "kind": "file",
            "bytes": 10,
            "sha256": hashlib.sha256(b"scan-bytes").hexdigest(),
        }
    ]
    assert {e["path"]: e["sha256"] for e in item["output_files"]} == {
        str(brain): hashlib.sha256(b"brain-bytes").hexdigest(),
        str(registered): hashlib.sha256(b"registered-bytes").hexdigest(),
    }
    # The index entry stays small: no results, no checksums.
    index = runs.load_run(record.id, root=root)
    assert index is not None
    assert "result" not in index.to_dict()["items"][0]["steps"][0]
    assert index.items[0].label == "P1/ses-1"

    assert runs.delete_run(record.id, root=root)
    assert not folder.exists()


def test_a_confirmation_that_no_longer_matches_is_written_down(tmp_path: Path) -> None:
    """The run is under way when the copies are made: a difference is recorded, not hidden."""
    import hashlib

    folder = tmp_path / "run"
    folder.mkdir()
    record_file = tmp_path / "sel_confirmed.json"
    record_file.write_text('{"kind": "medmcp-cohort confirm_selection"}')
    summary: JsonDict = {
        "path": str(record_file),
        "sha256": hashlib.sha256(record_file.read_bytes()).hexdigest(),
        "confirmed_csv_sha256": "a" * 64,
    }
    same = runs.RunManager._copy_confirmation(folder, summary, "a" * 64)
    assert same["copy"] == runs.CONFIRMATION_FILE and "error" not in same
    assert (folder / runs.CONFIRMATION_FILE).read_bytes() == record_file.read_bytes()

    record_file.write_text("{}")
    changed = runs.RunManager._copy_confirmation(folder, summary, "b" * 64)
    assert "does not have the confirmed SHA-256" in changed["error"]
    assert "changed after the run was checked" in changed["error"]

    record_file.unlink()
    gone = runs.RunManager._copy_confirmation(folder, summary, "a" * 64)
    assert gone["copy"] is None and "could not be copied" in gone["error"]


def test_load_full_falls_back_to_the_index_for_an_old_run(tmp_path: Path) -> None:
    """A run recorded before folders existed still loads, with what the index knows."""
    record = runs.RunRecord(
        id="20260101-000000-abc123",
        workflow="w",
        steps_per_item=1,
        items=[runs.ItemRecord(inputs={"in_1": "/a"}, ok=True)],
        status="done",
    )
    runs.save_run(record, root=tmp_path)
    full = runs.load_full(record.id, root=tmp_path)
    assert full is not None and full.to_dict(full=True)["environment"] == {}
    assert runs.load_full("nope", root=tmp_path) is None
