"""Tests for Tier-1 provenance capture (manifest, run log, permissions, report)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from medmcp import provenance

SESSION_ID = "abcd1234-1111-2222-3333-444455556666"


# ── split_tool_name ──────────────────────────────────────────────────────────


class TestSplitToolName:
    """split_tool_name resolves server/tool from known names or convention."""

    def test_known_server_prefix(self) -> None:
        """A name matching a known server splits on that prefix."""
        assert provenance.split_tool_name("medmcp-neuro_skull_strip", ["medmcp-neuro"]) == (
            "medmcp-neuro",
            "skull_strip",
        )

    def test_convention_fallback_without_known_names(self) -> None:
        """An unknown medmcp-* tool still splits via the naming convention."""
        assert provenance.split_tool_name("medmcp-dicom_load_series", []) == (
            "medmcp-dicom",
            "load_series",
        )

    def test_builtin_tool(self) -> None:
        """A bare tool name is attributed to the builtin server."""
        assert provenance.split_tool_name("bash", ["medmcp-neuro"]) == ("builtin", "bash")

    def test_longest_prefix_wins(self) -> None:
        """A more specific server name takes precedence over a shorter one."""
        server, tool = provenance.split_tool_name(
            "medmcp-neuro-ext_run", ["medmcp-neuro", "medmcp-neuro-ext"]
        )
        assert (server, tool) == ("medmcp-neuro-ext", "run")


# ── manifest ─────────────────────────────────────────────────────────────────


class TestManifest:
    """write_manifest captures and round-trips the session environment."""

    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        """A written manifest reads back with stacks and model populated."""
        servers = [{"name": "medmcp-neuro", "version": "0.2.0", "command": "/x/bin/neuro"}]
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            path = provenance.write_manifest(SESSION_ID, servers=servers, model_name="muse-medmcp")
            assert path.exists()
            manifest = provenance.read_manifest(SESSION_ID)

        assert manifest is not None
        assert manifest["session_id"] == SESSION_ID
        assert manifest["stacks"][0]["version"] == "0.2.0"
        assert manifest["model"]["name"] == "muse-medmcp"
        assert "python" in manifest["platform"]

    def test_model_params_pulled_from_config(self, tmp_path: Path) -> None:
        """Model temperature/thinking are pulled from config.toml by alias."""
        (tmp_path / "config.toml").write_text(
            '[[models]]\nname = "muse-medmcp"\nalias = "local"\n'
            'temperature = 1.0\nthinking = "medium"\ninput_price = 0.0\n'
        )
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            provenance.write_manifest(SESSION_ID, servers=[], model_name="local")
            manifest = provenance.read_manifest(SESSION_ID)

        assert manifest is not None
        assert manifest["model"]["thinking"] == "medium"
        # Price fields are intentionally dropped from the manifest.
        assert "input_price" not in manifest["model"]

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        """Reading a manifest that was never written returns None."""
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            assert provenance.read_manifest(SESSION_ID) is None

    def test_captures_container_image_ref(self, tmp_path: Path) -> None:
        """A docker-launched stack records its image ref (the last non-flag arg)."""
        servers = [
            {
                "name": "medmcp-neuro",
                "command": "docker",
                "args": ["run", "--rm", "-i", "-v", "/d:/d", "ghcr.io/medmcp/neuro:main"],
            },
            {"name": "medmcp-dicom", "version": "0.1.0", "command": "/x/bin/dicom"},
        ]
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            provenance.write_manifest(SESSION_ID, servers=servers, model_name="m")
            manifest = provenance.read_manifest(SESSION_ID)

        assert manifest is not None
        assert manifest["stacks"][0]["image"] == "ghcr.io/medmcp/neuro:main"
        # A uv-tool stack carries no image key.
        assert "image" not in manifest["stacks"][1]


# ── run log ──────────────────────────────────────────────────────────────────


class TestRunLog:
    """append_run_event / read_run_events / normalize_tool_event."""

    def test_normalize_includes_optional_fields(self) -> None:
        """Optional fields appear only when provided."""
        event = provenance.normalize_tool_event(
            tool_call_id="call_1",
            title="Skull strip",
            server="medmcp-neuro",
            tool="skull_strip",
            raw_input={"device": "cuda"},
            raw_output=None,
            output_text="ok",
            status="completed",
            decision="allow",
            risks=["file_write"],
            human_readable="Removes the skull.",
            duration_sec=1.2345,
        )
        assert event["server"] == "medmcp-neuro"
        assert event["arguments"] == {"device": "cuda"}
        assert event["output_text"] == "ok"
        assert event["permission_decision"] == "allow"
        assert event["risks"] == ["file_write"]
        assert event["duration_sec"] == 1.234

    def test_append_and_read(self, tmp_path: Path) -> None:
        """Appended events are read back in order with a timestamp added."""
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            provenance.append_run_event(SESSION_ID, {"tool": "a", "status": "completed"})
            provenance.append_run_event(SESSION_ID, {"tool": "b", "status": "failed"})
            events = provenance.read_run_events(SESSION_ID)

        assert [e["tool"] for e in events] == ["a", "b"]
        assert all("ts" in e for e in events)

    def test_read_skips_corrupt_lines(self, tmp_path: Path) -> None:
        """A malformed JSONL line is skipped, not fatal."""
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            provenance.append_run_event(SESSION_ID, {"tool": "a"})
            run_path = provenance.provenance_dir(SESSION_ID) / "run.jsonl"
            with run_path.open("a") as f:
                f.write("{not json\n")
            provenance.append_run_event(SESSION_ID, {"tool": "b"})
            events = provenance.read_run_events(SESSION_ID)

        assert [e["tool"] for e in events] == ["a", "b"]


# ── permissions ──────────────────────────────────────────────────────────────


def test_log_permission_appends_lines(tmp_path: Path) -> None:
    """Permission decisions are appended as tab-separated lines."""
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        provenance.log_permission(SESSION_ID, title="bash: rm", decision="reject")
        provenance.log_permission(SESSION_ID, title="bash: ls", decision="allow")
        log = (provenance.provenance_dir(SESSION_ID) / "permissions.log").read_text()

    lines = log.strip().splitlines()
    assert len(lines) == 2
    assert "\treject\tbash: rm" in lines[0]
    assert "\tallow\tbash: ls" in lines[1]


# ── vibe session lookup ──────────────────────────────────────────────────────


def test_find_vibe_session_dir(tmp_path: Path) -> None:
    """The session dir is located by id prefix and confirmed via meta.json."""
    sess_dir = tmp_path / "logs" / "session" / "session_20260101_000000_abcd1234"
    sess_dir.mkdir(parents=True)
    (sess_dir / "meta.json").write_text(json.dumps({"session_id": SESSION_ID}))
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        found = provenance.find_vibe_session_dir(SESSION_ID)
    assert found == sess_dir


def test_find_vibe_session_dir_missing(tmp_path: Path) -> None:
    """A session with no log dir resolves to None."""
    (tmp_path / "logs" / "session").mkdir(parents=True)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.find_vibe_session_dir(SESSION_ID) is None


def test_purge_session_removes_provenance_and_vibe_logs(tmp_path: Path) -> None:
    """purge_session deletes both the provenance dir and the vibe transcript dir."""
    vibe_dir = tmp_path / "logs" / "session" / "session_20260101_000000_abcd1234"
    vibe_dir.mkdir(parents=True)
    (vibe_dir / "meta.json").write_text(json.dumps({"session_id": SESSION_ID}))
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        provenance.append_run_event(SESSION_ID, {"tool": "a"})
        prov_dir = provenance.provenance_dir(SESSION_ID)
        assert prov_dir.exists()

        provenance.purge_session(SESSION_ID)

        assert not prov_dir.exists()
        assert not vibe_dir.exists()


def test_purge_session_is_safe_when_absent(tmp_path: Path) -> None:
    """Purging a session with nothing on disk does not raise."""
    (tmp_path / "logs" / "session").mkdir(parents=True)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        provenance.purge_session(SESSION_ID)  # no error


# ── compaction chains ────────────────────────────────────────────────────────

CONT_ID = "ef567890-9999-8888-7777-666655554444"
CONT2_ID = "01234567-aaaa-bbbb-cccc-ddddeeeeffff"


def _make_vibe_session(
    root: Path, timestamp: str, session_id: str, parent_id: str | None = None
) -> Path:
    """Create a vibe-style transcript dir with a meta.json (and empty log)."""
    d = root / "logs" / "session" / f"session_{timestamp}_{session_id[:8]}"
    d.mkdir(parents=True)
    meta: dict[str, object] = {"session_id": session_id}
    if parent_id is not None:
        meta["parent_session_id"] = parent_id
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "messages.jsonl").write_text("")
    return d


def test_find_vibe_session_dirs_follows_compaction_chain(tmp_path: Path) -> None:
    """The chain lists the original dir first, then continuations in order."""
    original = _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    cont = _make_vibe_session(tmp_path, "20260101_010000", CONT_ID, parent_id=SESSION_ID)
    cont2 = _make_vibe_session(tmp_path, "20260101_020000", CONT2_ID, parent_id=CONT_ID)
    _make_vibe_session(tmp_path, "20260101_030000", "99999999-0000-0000-0000-000000000000")
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.find_vibe_session_dirs(SESSION_ID) == [original, cont, cont2]


def test_find_vibe_session_dirs_without_continuations(tmp_path: Path) -> None:
    """A session that never compacted yields just its own dir."""
    original = _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.find_vibe_session_dirs(SESSION_ID) == [original]


def test_find_vibe_session_dirs_unknown_session(tmp_path: Path) -> None:
    """An unknown id yields an empty chain."""
    (tmp_path / "logs" / "session").mkdir(parents=True)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.find_vibe_session_dirs(SESSION_ID) == []


def test_vibe_session_parents_maps_backlinks(tmp_path: Path) -> None:
    """Only sessions with a parent_session_id appear, mapped child -> parent."""
    _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    _make_vibe_session(tmp_path, "20260101_010000", CONT_ID, parent_id=SESSION_ID)
    _make_vibe_session(tmp_path, "20260101_020000", CONT2_ID, parent_id=CONT_ID)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.vibe_session_parents() == {CONT_ID: SESSION_ID, CONT2_ID: CONT_ID}


def test_vibe_chain_tip_follows_to_newest_link(tmp_path: Path) -> None:
    """The tip of a compaction chain is the newest continuation's id."""
    _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    _make_vibe_session(tmp_path, "20260101_010000", CONT_ID, parent_id=SESSION_ID)
    _make_vibe_session(tmp_path, "20260101_020000", CONT2_ID, parent_id=CONT_ID)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.vibe_chain_tip(SESSION_ID) == CONT2_ID


def test_vibe_chain_tip_without_chain_is_identity(tmp_path: Path) -> None:
    """A session with no continuation (or unknown) maps to itself."""
    _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.vibe_chain_tip(SESSION_ID) == SESSION_ID
        assert provenance.vibe_chain_tip("99999999-x") == "99999999-x"


def test_chain_walk_stops_at_forks(tmp_path: Path) -> None:
    """A fork (stop id) is not treated as its source's continuation.

    Forks carry the same parent_session_id backlink as compaction
    continuations; without the stop set, resuming the source would land in
    the fork and purging the source would delete it.
    """
    original = _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    fork = _make_vibe_session(tmp_path, "20260101_010000", CONT_ID, parent_id=SESSION_ID)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.find_vibe_session_dirs(SESSION_ID, stop_ids={CONT_ID}) == [original]
        assert provenance.vibe_chain_tip(SESSION_ID, stop_ids={CONT_ID}) == SESSION_ID
        # The fork's own chain still walks (its own id never stops itself).
        assert provenance.find_vibe_session_dirs(CONT_ID, stop_ids={CONT_ID}) == [fork]

        provenance.purge_session(SESSION_ID, stop_ids={CONT_ID})

        assert not original.exists()
        assert fork.exists()  # the branched chat survives its source's deletion


def test_chain_walk_stops_only_at_fork_boundary(tmp_path: Path) -> None:
    """A compaction continuation *behind* a fork boundary is still excluded."""
    _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    _make_vibe_session(tmp_path, "20260101_010000", CONT_ID, parent_id=SESSION_ID)
    # CONT2 continues the fork (e.g. the fork later compacted) — not ours.
    _make_vibe_session(tmp_path, "20260101_020000", CONT2_ID, parent_id=CONT_ID)
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        dirs = provenance.find_vibe_session_dirs(SESSION_ID, stop_ids={CONT_ID})
        assert [d.name.split("_")[-1] for d in dirs] == [SESSION_ID[:8]]


def test_purge_session_removes_compaction_continuations(tmp_path: Path) -> None:
    """Purging a chat also deletes the transcript dirs compaction rolled over to."""
    original = _make_vibe_session(tmp_path, "20260101_000000", SESSION_ID)
    cont = _make_vibe_session(tmp_path, "20260101_010000", CONT_ID, parent_id=SESSION_ID)
    unrelated = _make_vibe_session(
        tmp_path, "20260101_020000", "99999999-0000-0000-0000-000000000000"
    )
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        provenance.append_run_event(SESSION_ID, {"tool": "a"})

        provenance.purge_session(SESSION_ID)

        assert not provenance.provenance_dir(SESSION_ID).exists()
        assert not original.exists()
        assert not cont.exists()
        assert unrelated.exists()


# ── orphan GC ────────────────────────────────────────────────────────────────


def test_purge_orphans_removes_unreferenced_only(tmp_path: Path) -> None:
    """purge_orphans deletes provenance dirs whose id is not referenced."""
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        provenance.append_run_event("keep-1", {"tool": "a"})
        provenance.append_run_event("drop-1", {"tool": "b"})
        provenance.append_run_event("drop-2", {"tool": "c"})

        purged = provenance.purge_orphans({"keep-1"})

        assert sorted(purged) == ["drop-1", "drop-2"]
        assert provenance.provenance_dir("keep-1").exists()
        assert not provenance.provenance_dir("drop-1").exists()
        assert not provenance.provenance_dir("drop-2").exists()


def test_purge_orphans_leaves_vibe_transcripts(tmp_path: Path) -> None:
    """purge_orphans never touches vibe transcript dirs, only provenance."""
    vibe_dir = tmp_path / "logs" / "session" / "session_20260101_000000_abcd1234"
    vibe_dir.mkdir(parents=True)
    (vibe_dir / "meta.json").write_text(json.dumps({"session_id": SESSION_ID}))
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        provenance.append_run_event(SESSION_ID, {"tool": "a"})
        provenance.purge_orphans(set())  # nothing referenced
        assert not provenance.provenance_dir(SESSION_ID).exists()
        assert vibe_dir.exists()


def test_purge_orphans_empty_when_no_records(tmp_path: Path) -> None:
    """With no provenance directory at all, purge_orphans is a no-op."""
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert provenance.purge_orphans(set()) == []


# ── report ───────────────────────────────────────────────────────────────────


class TestReport:
    """render_report / write_report build a documentation-grade summary."""

    def test_render_includes_env_and_steps(self, tmp_path: Path) -> None:
        """The report shows the model, stacks, and each recorded tool call."""
        servers = [{"name": "medmcp-neuro", "version": "0.2.0", "command": "/x"}]
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            provenance.write_manifest(SESSION_ID, servers=servers, model_name="muse-medmcp")
            provenance.append_run_event(
                SESSION_ID,
                {
                    "server": "medmcp-neuro",
                    "tool": "skull_strip",
                    "status": "completed",
                    "permission_decision": "allow",
                    "arguments": {"device": "cuda"},
                    "output": "structured: {'brain_path': 'data/x/brain.nii.gz'}",
                },
            )
            report = provenance.render_report(SESSION_ID)

        assert "muse-medmcp" in report
        assert "medmcp-neuro 0.2.0" in report
        assert "`medmcp-neuro:skull_strip` — completed" in report
        assert "decision: allow" in report
        assert "data/x/brain.nii.gz" in report

    def test_write_report_returns_none_when_empty(self, tmp_path: Path) -> None:
        """No manifest and no events → no report file is written."""
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            assert provenance.write_report(SESSION_ID) is None

    def test_write_report_creates_file(self, tmp_path: Path) -> None:
        """With provenance present, report.md is written to the session dir."""
        with patch.object(provenance, "VIBE_HOME", tmp_path):
            provenance.append_run_event(SESSION_ID, {"server": "builtin", "tool": "bash"})
            path = provenance.write_report(SESSION_ID)
        assert path is not None and path.name == "report.md" and path.exists()
