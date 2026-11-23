"""Tests for what a run records beyond progress: checksums, environment, report."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from medmcp import provenance, runrecord, settings
from medmcp.workflow import Recipe, RecipeStep, StackRequirement

JsonDict = dict[str, Any]


def test_checksum_of_a_file(tmp_path: Path) -> None:
    """A file is hashed as it is, with its size."""
    scan = tmp_path / "scan.nii.gz"
    scan.write_bytes(b"voxels")
    entry = runrecord.checksum(scan)
    assert entry["sha256"] == hashlib.sha256(b"voxels").hexdigest()
    assert entry["bytes"] == 6 and entry["kind"] == "file"
    assert "skipped" not in entry


def test_checksum_of_a_folder_changes_with_any_file(tmp_path: Path) -> None:
    """A series folder hashes over names and contents, in a stable order."""
    series = tmp_path / "series"
    series.mkdir()
    (series / "b.dcm").write_bytes(b"2")
    (series / "a.dcm").write_bytes(b"1")
    first = runrecord.checksum(series)
    assert first["kind"] == "dir" and first["files"] == 2 and first["bytes"] == 2
    assert first["sha256"] == runrecord.checksum(series)["sha256"]
    (series / "a.dcm").rename(series / "c.dcm")
    assert runrecord.checksum(series)["sha256"] != first["sha256"]


def test_checksum_says_why_it_did_not_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Over the cap, switched off, or absent: the size and the reason are recorded."""
    big = tmp_path / "big.nii"
    big.write_bytes(b"x" * 2048)
    capped = runrecord.checksum(big, max_bytes=1024)
    assert capped["sha256"] is None and capped["bytes"] == 2048 and "cap" in capped["skipped"]
    monkeypatch.setenv("MEDMCP_RUN_HASH_MAX_MB", "0")
    assert "turned off" in runrecord.checksum(big)["skipped"]
    monkeypatch.setenv("MEDMCP_RUN_HASH_MAX_MB", "not a number")
    assert runrecord.hash_max_bytes() == 2048 * 1024 * 1024
    assert runrecord.checksum(tmp_path / "gone.nii")["skipped"] == "not found"


def _recipe() -> Recipe:
    return Recipe(
        name="wf",
        description="d",
        steps=[RecipeStep(server="medmcp-tptbox", tool="describe_image", arguments={})],
        requires=[StackRequirement(stack="medmcp-tptbox", image="img:main", digest="sha256:old")],
        source={"session_id": "s1", "model": "muse-medmcp"},
    )


def test_environment_records_image_digest_and_the_pin() -> None:
    """Each used stack carries its image, the installed digest and whether it matches."""
    servers: list[JsonDict] = [
        {"name": "medmcp-tptbox", "command": "docker", "args": ["run", "-i", "img:main"]},
        {"name": "medmcp-unused", "command": "docker", "args": ["run", "-i", "other:main"]},
    ]

    def identity(image: str) -> JsonDict:
        return {"digest": "sha256:new", "id": "sha256:id"}

    env = runrecord.environment(_recipe(), servers, image_identity=identity)
    assert [s["name"] for s in env["stacks"]] == ["medmcp-tptbox"]
    stack = env["stacks"][0]
    assert stack["image"] == "img:main" and stack["digest"] == "sha256:new"
    assert stack["image_id"] == "sha256:id"
    assert stack["pinned_digest"] == "sha256:old" and stack["matches_pin"] is False
    assert env["model"] is None and env["recorded_model"] == "muse-medmcp"
    assert env["medmcp"]["version"]


def test_environment_names_a_stack_that_is_not_installed() -> None:
    """A missing stack is listed as such rather than left out."""
    env = runrecord.environment(_recipe(), [])
    assert env["stacks"] == [
        {"name": "medmcp-tptbox", "installed": False, "pinned_digest": "sha256:old"}
    ]


def test_core_identity_falls_back_to_what_the_image_baked_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No git checkout in an image: the commit comes from the build arguments."""

    def no_git(args: list[str]) -> str | None:
        return None

    monkeypatch.setattr(provenance, "_git", no_git)
    monkeypatch.setenv("MEDMCP_BUILD", "v0.8.0")
    monkeypatch.setenv("MEDMCP_REVISION", "abc123")
    identity = provenance.core_identity()
    assert identity["build"] == "v0.8.0" and identity["git_commit"] == "abc123"
    monkeypatch.delenv("MEDMCP_REVISION")
    assert provenance.core_identity()["git_commit"] is None


def test_chat_manifest_carries_stack_digests() -> None:
    """A chat's manifest pins each container stack by digest, not only by tag."""
    servers: list[JsonDict] = [
        {"name": "medmcp-tptbox", "command": "docker", "args": ["run", "-i", "img:main"]}
    ]

    def identity(image: str) -> JsonDict:
        return {"digest": "sha256:d", "id": "sha256:i"}

    manifest = provenance.build_manifest(
        "s1", servers=servers, model_name="m", image_identity=identity
    )
    assert manifest["stacks"][0]["digest"] == "sha256:d"
    assert manifest["medmcp"]["version"]


def test_image_identity_reads_digest_and_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """One inspect call yields the matching registry digest and the image id."""
    import subprocess

    def fake_docker(args: list[str], *, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        out = '["ghcr.io/other/x@sha256:zzz","ghcr.io/medmcp/tptbox@sha256:abc"] sha256:idid\n'
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr(settings, "_run_docker", fake_docker)
    assert settings.image_identity("ghcr.io/medmcp/tptbox:main") == {
        "digest": "sha256:abc",
        "id": "sha256:idid",
    }

    def local_only(args: list[str], *, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="[] sha256:local\n", stderr="")

    monkeypatch.setattr(settings, "_run_docker", local_only)
    assert settings.image_identity("medmcp-spine:dev") == {"digest": None, "id": "sha256:local"}

    def absent(args: list[str], *, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("No such image")

    monkeypatch.setattr(settings, "_run_docker", absent)
    assert settings.image_identity("nope:1") == {"digest": None, "id": None}


def test_report_names_items_results_and_checksums() -> None:
    """The Markdown report carries what the JSON does, readably."""
    record: JsonDict = {
        "id": "r1",
        "workflow": "wf",
        "status": "failed",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:05+00:00",
        "total": 1,
        "succeeded": 0,
        "failed": 1,
        "confirmation": {"previewed": True, "by": "workspace user", "confirmed_at": "t"},
        "environment": {
            "medmcp": {"version": "0.7.4", "build": "v0.7.4", "git_commit": "abc"},
            "stacks": [{"name": "medmcp-tptbox", "image": "img:main", "digest": "sha256:d"}],
            "platform": {"platform": "Linux", "python": "3.12"},
        },
        "items": [
            {
                "label": "P1/ses-1",
                "ok": False,
                "error": "boom\ntraceback",
                "input_files": [{"path": "/a", "bytes": 3, "sha256": "aa"}],
                "output_files": [{"path": "/b", "bytes": 9, "sha256": None, "skipped": "cap"}],
                "steps": [
                    {
                        "index": 1,
                        "server": "medmcp-tptbox",
                        "tool": "describe_image",
                        "arguments": {"image_path": "/a"},
                        "ok": False,
                        "error": "boom",
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "finished_at": "2026-01-01T00:00:02+00:00",
                        "result": {"shape": [1, 2, 3]},
                    }
                ],
            }
        ],
    }
    text = runrecord.render_report(record)
    assert "P1/ses-1 — failed" in text and "> boom" in text
    assert "`img:main` @ `sha256:d`" in text
    assert "sha256 `aa`" in text and "not hashed (cap)" in text
    assert "2.0 s" in text and '"shape"' in text
    assert "after the preview" in text
