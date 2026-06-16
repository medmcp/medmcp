"""Tier-1 provenance capture for MedMCP sessions.

Writes a self-contained, append-only record of everything that happened in a
session into ``.vibe/provenance/<session_id>/``:

- ``manifest.json``  — environment capture: medmcp git commit, installed stack
  versions, active model + params, OS/python.
- ``run.jsonl``      — one normalized event per tool call (server, tool, resolved
  arguments, structured output, permission decision, duration, status).
- ``permissions.log``— a human-readable mirror of the permission decisions that
  the ``medmcp.audit`` logger writes to stderr (the stderr trail is unchanged).
- ``report.md``      — a documentation-grade Markdown rendering, generated on
  demand from the manifest + run log.

This module is deliberately free of any UI/vibe-acp dependency so it can
also be driven from the CLI (:mod:`medmcp.provcli`). Callers pass in the data
they already have (active servers, model name). Every write is best-effort at
the call site — provenance must never break a chat.
"""

from __future__ import annotations

import ast
import contextlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

# Resolve the repo root from this file: src/medmcp/provenance.py →
# src/medmcp → src → <root>. VIBE_HOME is module-level so tests can monkeypatch
# it; all path helpers read it at call time rather than caching a derived path.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VIBE_HOME: Path = PROJECT_ROOT / ".vibe"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def provenance_dir(session_id: str) -> Path:
    """Return the provenance directory for *session_id* (not created)."""
    return VIBE_HOME / "provenance" / session_id


def _ensure_dir(session_id: str) -> Path:
    """Return the provenance directory for *session_id*, creating it if needed."""
    d = provenance_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Environment manifest ───────────────────────────────────────────────────


def _git(args: list[str]) -> str | None:
    """Run ``git *args`` in the project root; return stdout or ``None`` on error."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _read_model_config(model_name: str) -> JsonDict:
    """Return the ``[[models]]`` entry for *model_name* from config.toml, or ``{}``.

    The match is by ``name`` first, then by ``alias`` (the active model in
    config.toml is referenced by alias, e.g. ``"local"``).
    """
    config_path = VIBE_HOME / "config.toml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("rb") as f:
            cfg = tomllib.load(f)
    except Exception:
        return {}
    models = cast("list[JsonDict]", cfg.get("models", []))
    for model in models:
        if model.get("name") == model_name or model.get("alias") == model_name:
            return {k: v for k, v in model.items() if k not in ("input_price", "output_price")}
    return {}


def build_manifest(session_id: str, *, servers: list[JsonDict], model_name: str) -> JsonDict:
    """Assemble (but do not write) the environment manifest for a session."""
    return {
        "session_id": session_id,
        "created_at": _utc_now_iso(),
        "medmcp": {
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "stacks": [
            {
                "name": s.get("name"),
                "version": s.get("version"),
                "command": s.get("command"),
            }
            for s in servers
        ],
        "model": {"name": model_name, **_read_model_config(model_name)},
        "platform": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "system": platform.system(),
        },
    }


def write_manifest(session_id: str, *, servers: list[JsonDict], model_name: str) -> Path:
    """Write ``manifest.json`` for *session_id* and return its path."""
    d = _ensure_dir(session_id)
    manifest = build_manifest(session_id, servers=servers, model_name=model_name)
    path = d / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_manifest(session_id: str) -> JsonDict | None:
    """Read back ``manifest.json`` for *session_id*, or ``None`` if absent/invalid."""
    path = provenance_dir(session_id) / "manifest.json"
    if not path.exists():
        return None
    try:
        return cast("JsonDict", json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


# ── Run event log ──────────────────────────────────────────────────────────


def split_tool_name(name: str, server_names: list[str]) -> tuple[str, str]:
    """Split a tool name like ``medmcp-neuro_skull_strip`` into (server, tool).

    Matches against known *server_names* first (longest prefix wins so a server
    name that is itself a prefix of another can't mis-bind). Falls back to
    ``("builtin", name)`` for vibe-acp tools that carry no server prefix.
    """
    for server in sorted(server_names, key=len, reverse=True):
        prefix = f"{server}_"
        if name.startswith(prefix):
            return server, name[len(prefix) :]
    # Convention fallback for stacks not present in *server_names*: MedMCP stack
    # tools are named ``medmcp-<stack>_<tool>``.
    conv = re.match(r"(medmcp-[a-z0-9]+)_(.+)", name)
    if conv:
        return conv.group(1), conv.group(2)
    return "builtin", name


def normalize_tool_event(
    *,
    tool_call_id: str,
    title: str | None,
    server: str,
    tool: str,
    raw_input: object,
    raw_output: object,
    output_text: str | None,
    status: str,
    decision: str | None,
    risks: list[str] | None,
    human_readable: str | None,
    duration_sec: float | None,
) -> JsonDict:
    """Build a normalized run-log event dict from the UI's tool-call state."""
    event: JsonDict = {
        "tool_call_id": tool_call_id,
        "server": server,
        "tool": tool,
        "status": status,
    }
    if title:
        event["title"] = title
    if raw_input is not None:
        event["arguments"] = raw_input
    if raw_output is not None:
        event["output"] = raw_output
    elif output_text is not None:
        event["output_text"] = output_text
    if decision is not None:
        event["permission_decision"] = decision
    if risks:
        event["risks"] = risks
    if human_readable:
        event["explanation"] = human_readable
    if duration_sec is not None:
        event["duration_sec"] = round(duration_sec, 3)
    return event


def append_run_event(session_id: str, event: JsonDict) -> None:
    """Append a single timestamped event to ``run.jsonl`` for *session_id*."""
    d = _ensure_dir(session_id)
    line = json.dumps({"ts": _utc_now_iso(), **event})
    with (d / "run.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_run_events(session_id: str) -> list[JsonDict]:
    """Read all events from ``run.jsonl`` for *session_id* (skips bad lines)."""
    path = provenance_dir(session_id) / "run.jsonl"
    if not path.exists():
        return []
    events: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(cast("JsonDict", json.loads(line)))
        except json.JSONDecodeError:
            continue
    return events


def log_permission(session_id: str, *, title: str, decision: str) -> None:
    """Append a permission decision to ``permissions.log`` for *session_id*.

    This is a persisted mirror of the ``medmcp.audit`` stderr trail; the
    stderr handler is unchanged and must not be silenced.
    """
    d = _ensure_dir(session_id)
    with (d / "permissions.log").open("a", encoding="utf-8") as f:
        f.write(f"{_utc_now_iso()}\t{decision}\t{title}\n")


def record_tool_event(session_id: str, tc_id: str, info: JsonDict, server_names: list[str]) -> None:
    """Append a normalized run-log event for a completed tool call (best-effort).

    ``info`` is the UI's accumulated tool-call state (title, rawInput, status,
    rawOutput/outputText, decision, risks, humanReadable, ``_started`` monotonic
    timestamp). Idempotent per call via the ``_logged`` marker, since
    ``tool_call_update`` can fire more than once. Never raises — provenance
    must not break a chat.
    """
    if info.get("_logged"):
        return
    info["_logged"] = True
    started = info.get("_started")
    duration = time.monotonic() - started if isinstance(started, (int, float)) else None
    title = info.get("title")
    title_str = title if isinstance(title, str) else None
    server, tool = split_tool_name(title_str or tc_id, server_names)
    risks = info.get("risks")
    decision = info.get("decision")
    human_readable = info.get("humanReadable")
    output_text = info.get("outputText")
    event = normalize_tool_event(
        tool_call_id=tc_id,
        title=title_str,
        server=server,
        tool=tool,
        raw_input=info.get("rawInput"),
        raw_output=info.get("rawOutput"),
        output_text=output_text if isinstance(output_text, str) else None,
        status=str(info.get("status") or ""),
        decision=str(decision) if decision is not None else None,
        risks=cast("list[str]", risks) if isinstance(risks, list) else None,
        human_readable=human_readable if isinstance(human_readable, str) else None,
        duration_sec=duration,
    )
    with contextlib.suppress(Exception):
        append_run_event(session_id, event)


# ── Vibe session lookup ──────────────────────────────────────────────────────


def find_vibe_session_dir(session_id: str) -> Path | None:
    """Locate vibe-acp's log dir for *session_id* under ``.vibe/logs/session/``.

    vibe names directories ``session_<timestamp>_<first8-of-uuid>``, so we glob
    on the id prefix and confirm the full id via the directory's ``meta.json``.
    """
    sessions_root = VIBE_HOME / "logs" / "session"
    if not sessions_root.is_dir():
        return None
    short = session_id[:8]
    candidates = sorted(sessions_root.glob(f"session_*_{short}"))
    for candidate in candidates:
        meta = candidate / "meta.json"
        if not meta.exists():
            continue
        try:
            data = cast("JsonDict", json.loads(meta.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("session_id") == session_id:
            return candidate
    # Fall back to the single prefix match even without a meta confirmation.
    return candidates[0] if candidates else None


def list_provenance_sessions() -> list[str]:
    """Return the session ids that currently have a provenance directory."""
    root = VIBE_HOME / "provenance"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def purge_orphans(referenced_ids: set[str]) -> list[str]:
    """Delete provenance dirs whose session id is not in *referenced_ids*.

    A garbage-collection sweep so chats deleted in the UI (or sessions that never
    persisted a thread mapping) don't leak provenance. Returns the purged session
    ids. Only the provenance directory is removed — vibe transcript dirs are left
    untouched, because a compaction continuation is named with an id we don't
    track and could still belong to a live chat.
    """
    purged: list[str] = []
    for session_id in list_provenance_sessions():
        if session_id not in referenced_ids:
            shutil.rmtree(provenance_dir(session_id), ignore_errors=True)
            purged.append(session_id)
    return purged


def purge_session(session_id: str) -> None:
    """Delete all on-disk logs for *session_id*.

    Removes both the provenance directory and vibe-acp's session transcript dir,
    so deleting a chat leaves nothing orphaned on disk. Best-effort: missing
    paths are ignored.
    """
    pdir = provenance_dir(session_id)
    if pdir.exists():
        shutil.rmtree(pdir, ignore_errors=True)
    delete_vibe_transcript(session_id)


def delete_vibe_transcript(session_id: str) -> None:
    """Delete only vibe-acp's transcript dir for *session_id* (keep provenance).

    Used to retire a session that a fork has superseded: when continuing a
    reloaded chat, vibe-acp logs under a new id, leaving the original transcript
    a stale duplicate. Removing it drops the duplicate from vibe's session list
    while the provenance is relocated to the fork via :func:`move_session_record`.
    """
    vibe_dir = find_vibe_session_dir(session_id)
    if vibe_dir is not None and vibe_dir.exists():
        shutil.rmtree(vibe_dir, ignore_errors=True)


def move_session_record(old_id: str, new_id: str) -> None:
    """Relocate the provenance record from *old_id* to *new_id* (best-effort).

    No-op if the source is absent; if the destination already exists the source
    is left untouched rather than clobbering it.
    """
    src = provenance_dir(old_id)
    if not src.exists():
        return
    dst = provenance_dir(new_id)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


# ── Human-readable report ────────────────────────────────────────────────────

_PATH_KEY_RE = re.compile(r"path|dir|file|output|input", re.IGNORECASE)


def _structured_paths(output: object) -> list[str]:
    """Best-effort extraction of file paths from a tool's structured output."""
    paths: list[str] = []
    candidate: JsonDict | None = None
    if isinstance(output, dict):
        candidate = cast("JsonDict", output)
    elif isinstance(output, str):
        match = re.search(r"structured:\s*(\{.*\})", output, re.DOTALL)
        if match:
            try:
                parsed = ast.literal_eval(match.group(1))
                if isinstance(parsed, dict):
                    candidate = cast("JsonDict", parsed)
            except (ValueError, SyntaxError):
                candidate = None
    if candidate is None:
        return paths
    for key, value in candidate.items():
        if isinstance(value, str) and _PATH_KEY_RE.search(str(key)) and "/" in value:
            paths.append(value)
    return paths


def render_report(session_id: str) -> str:
    """Render a documentation-grade Markdown report from the provenance record."""
    manifest = read_manifest(session_id)
    events = read_run_events(session_id)

    lines: list[str] = ["# MedMCP session report\n", f"**Session:** `{session_id}`\n"]

    if manifest is not None:
        created = manifest.get("created_at", "unknown")
        medmcp = cast("JsonDict", manifest.get("medmcp") or {})
        model = cast("JsonDict", manifest.get("model") or {})
        stacks = cast("list[JsonDict]", manifest.get("stacks") or [])
        plat = cast("JsonDict", manifest.get("platform") or {})

        lines.append("## Environment\n")
        lines.append(f"- **Recorded:** {created}")
        lines.append(
            f"- **medmcp:** `{medmcp.get('git_commit') or '?'}` "
            f"(branch `{medmcp.get('git_branch') or '?'}`)"
        )
        model_name = model.get("name", "?")
        temp = model.get("temperature")
        thinking = model.get("thinking")
        model_line = f"- **Model:** `{model_name}`"
        extras = [
            f"{k}={v}" for k, v in (("temperature", temp), ("thinking", thinking)) if v is not None
        ]
        if extras:
            model_line += f" ({', '.join(extras)})"
        lines.append(model_line)
        if stacks:
            rendered = ", ".join(
                f"{s.get('name')} {s.get('version') or ''}".strip() for s in stacks
            )
            lines.append(f"- **Stacks:** {rendered}")
        lines.append(
            f"- **Platform:** {plat.get('platform', '?')}, Python {plat.get('python', '?')}\n"
        )

    lines.append(f"## Tool calls ({len(events)})\n")
    if not events:
        lines.append("_No tool calls were recorded for this session._")
    for i, event in enumerate(events, start=1):
        server = event.get("server", "?")
        tool = event.get("tool", "?")
        status = event.get("status", "?")
        decision = event.get("permission_decision")
        header = f"### {i}. `{server}:{tool}` — {status}"
        if decision:
            header += f" (decision: {decision})"
        lines.append(header)
        explanation = event.get("explanation")
        if isinstance(explanation, str) and explanation:
            lines.append(f"\n> {explanation}\n")
        arguments = event.get("arguments")
        if arguments is not None:
            rendered_args = json.dumps(arguments, indent=2, default=str)
            lines.append(f"```json\n{rendered_args}\n```")
        out_paths = _structured_paths(event.get("output"))
        if out_paths:
            lines.append("**Outputs:** " + ", ".join(f"`{p}`" for p in out_paths))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(session_id: str) -> Path | None:
    """Write ``report.md`` for *session_id*; return its path, or ``None`` if empty.

    Returns ``None`` when there is nothing to report (no manifest and no run
    events), so callers don't create empty files for sessions that never ran.
    """
    if read_manifest(session_id) is None and not read_run_events(session_id):
        return None
    d = _ensure_dir(session_id)
    path = d / "report.md"
    path.write_text(render_report(session_id), encoding="utf-8")
    return path
