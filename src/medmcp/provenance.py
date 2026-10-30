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

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from medmcp.statefile import atomic_write_json, read_json, read_jsonl
from medmcp.toolresult import parse_structured_blob

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
    config.toml is referenced by alias, e.g. ``"local"``). The provider's
    ``api_base`` is added as ``endpoint``, so the record says where the model
    ran — the local server, or a cloud service.
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
            entry = {k: v for k, v in model.items() if k not in ("input_price", "output_price")}
            for provider in cast("list[JsonDict]", cfg.get("providers", [])):
                if provider.get("name") == model.get("provider") and provider.get("api_base"):
                    entry["endpoint"] = provider["api_base"]
            return entry
    return {}


def docker_image_ref(server: JsonDict) -> str | None:
    """Return the container image ref for a ``docker``-launched stack, else ``None``.

    MedMCP container stacks launch via ``docker run … <image>`` with no trailing
    command, so the image is the last non-flag argument (e.g.
    ``ghcr.io/medmcp/neuro:main``). Cheap and offline — no docker call.
    """
    if str(server.get("command", "")) != "docker":
        return None
    args = [str(a) for a in cast("list[Any]", server.get("args", []))]
    for arg in reversed(args):
        if not arg.startswith("-"):
            return None if arg == "run" else arg
    return None


def _stack_manifest_entry(server: JsonDict) -> JsonDict:
    """Build one stack entry for the manifest (name/version/command + image)."""
    entry: JsonDict = {
        "name": server.get("name"),
        "version": server.get("version"),
        "command": server.get("command"),
    }
    image = docker_image_ref(server)
    if image:
        entry["image"] = image
    return entry


def build_manifest(session_id: str, *, servers: list[JsonDict], model_name: str) -> JsonDict:
    """Assemble (but do not write) the environment manifest for a session."""
    return {
        "session_id": session_id,
        "created_at": _utc_now_iso(),
        "medmcp": {
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "stacks": [_stack_manifest_entry(s) for s in servers],
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
    return read_jsonl(provenance_dir(session_id) / "run.jsonl")


# ── MedMCP's own transcript ──────────────────────────────────────────────────
#
# vibe's Unified Harness (≥2.26) stores a chat as a hash-chained journal of
# runtime events — nothing a reader outside vibe should parse. The workspace
# already relays every turn over ACP, so it records the transcript it relays:
# one ``transcript.jsonl`` per chat in the shape of vibe's old ``messages.jsonl``
# (``role`` user / assistant / tool, assistant ``tool_calls`` with ``function``
# ``name`` and JSON-string ``arguments``, tool results by ``tool_call_id``), so
# titles, export and distillation read one format whatever vibe version wrote
# the chat. Written whether or not provenance capture is on: it is the chat's
# own record, not the reproducibility record.

TRANSCRIPT_FILE = "transcript.jsonl"


def transcript_path(session_id: str) -> Path:
    """Where *session_id*'s MedMCP-written transcript lives."""
    return provenance_dir(session_id) / TRANSCRIPT_FILE


def append_transcript(session_id: str, record: JsonDict) -> None:
    """Append one message-shaped record to the chat's transcript (best-effort)."""
    with contextlib.suppress(Exception):
        d = _ensure_dir(session_id)
        with (d / TRANSCRIPT_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _utc_now_iso(), **record}) + "\n")


def transcript_tool_records(
    tool_call_id: str, name: str, arguments: object, output: str, *, status: str
) -> tuple[JsonDict, JsonDict]:
    """The assistant call and the tool result that record one completed tool call."""
    call: JsonDict = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments if arguments is not None else {}),
                },
            }
        ],
    }
    result: JsonDict = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": output,
        "status": status,
    }
    return call, result


def read_transcript(session_id: str, *, stop_ids: Collection[str] = ()) -> list[JsonDict]:
    """A chat's messages: vibe's legacy ``messages.jsonl``, then the MedMCP transcript.

    The legacy file is concatenated across the pre-2.24 compaction chain (see
    :func:`find_vibe_session_dirs`; ``stop_ids`` keeps the walk out of forks), so
    chats written before the workspace kept its own transcript still read whole,
    and one resumed since then reads its old turns followed by the new ones (the
    transcript records live turns only, never a replay). Empty when neither exists.
    """
    messages: list[JsonDict] = []
    trusted = legacy_transcripts()
    for session_dir in find_vibe_session_dirs(session_id, stop_ids=stop_ids):
        path = session_dir / "messages.jsonl"
        if _is_trusted_legacy(path, trusted):
            messages.extend(read_jsonl(path))
    own = transcript_path(session_id)
    if own.exists():
        messages.extend(read_jsonl(own))
    return messages


def copy_transcript(src_id: str, dst_id: str) -> None:
    """Give a branched chat its source's MedMCP transcript (best-effort).

    vibe copies the history it holds into the fork; the workspace's own record
    has to follow, or the branch would read as starting at the fork. Only the
    source's own file is copied: legacy turns reach the branch through vibe's
    ``parent_session_id`` backlink, as for any chain.
    """
    with contextlib.suppress(Exception):
        src = transcript_path(src_id)
        if not src.exists():
            return
        d = _ensure_dir(dst_id)
        shutil.copyfile(src, d / TRANSCRIPT_FILE)


def truncate_transcript(session_id: str, message_id: str) -> bool:
    """Cut the MedMCP transcript back to before the user turn *message_id*.

    The counterpart of vibe's in-place rewind, which truncates its own history
    the same way. Returns whether the turn was found (and the file rewritten);
    a transcript that does not carry the id is left as it is.
    """
    path = transcript_path(session_id)
    if not path.exists():
        return False
    records = read_jsonl(path)
    for index, record in enumerate(records):
        if record.get("role") == "user" and record.get("message_id") == message_id:
            kept = records[:index]
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for rec in kept:
                    f.write(json.dumps(rec) + "\n")
            os.replace(tmp, path)
            return True
    return False


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
    timestamp and ``_started_at`` wall clock). Idempotent per call via the
    ``_logged`` marker, since ``tool_call_update`` can fire more than once. Never
    raises — provenance must not break a chat.
    """
    if info.get("_logged"):
        return
    info["_logged"] = True
    started = info.get("_started")
    duration = time.monotonic() - started if isinstance(started, (int, float)) else None
    title = info.get("title")
    title_str = title if isinstance(title, str) else None
    # The relay stores the canonical tool name (see ``harness``); the title is
    # prose under vibe ≥2.26 and only ever named the tool before that.
    tool_name = info.get("toolName")
    server, tool = split_tool_name(
        tool_name if isinstance(tool_name, str) and tool_name else (title_str or tc_id),
        server_names,
    )
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
#
# Everything under logs/ is written by the agent, which with agent isolation is
# another container that shares only this volume. So nothing here is followed
# through a symlink (a link planted there resolves in *this* container, onto its
# secrets), meta.json is size-capped, and the one file whose content the core
# replays — a legacy messages.jsonl — is read only if it was on disk, unchanged,
# before this version first started (see :func:`record_legacy_transcripts`).

_META_MAX_BYTES = 1 << 20


def _legacy_manifest_path() -> Path:
    """Where the legacy-transcript manifest lives.

    It maps each legacy transcript (vibe < 2.26 wrote messages.jsonl; 2.26 never
    does) found when this version first ran, logs-relative, to its sha256. On the
    state volume, which the agent cannot reach.
    """
    return VIBE_HOME / "state" / "legacy_transcripts.json"


def _sessions_root() -> Path | None:
    """``logs/session``, or ``None`` when absent or not a real directory."""
    logs = VIBE_HOME / "logs"
    root = logs / "session"
    if root.is_symlink() or not root.is_dir() or logs.is_symlink():
        return None
    return root


def _real_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _read_session_meta(session_dir: Path) -> JsonDict | None:
    """Read a vibe session dir's ``meta.json``, or ``None`` if absent/invalid."""
    meta = session_dir / "meta.json"
    try:
        if meta.is_symlink() or meta.stat().st_size > _META_MAX_BYTES:
            return None
        data = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return cast("JsonDict", data) if isinstance(data, dict) else None


def _session_dirs(sessions_root: Path) -> list[Path]:
    """Every vibe session dir: the legacy ``session_*`` dirs and the unified ones."""
    dirs = [d for d in sorted(sessions_root.iterdir()) if _real_dir(d) and d.name != "unified"]
    unified = sessions_root / "unified"
    if _real_dir(unified):
        dirs.extend(d for d in sorted(unified.iterdir()) if _real_dir(d))
    return dirs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_legacy_transcripts() -> int:
    """Fingerprint the legacy transcripts on disk, once; return how many.

    Called at startup, before the agent can run. A no-op once the manifest exists:
    vibe 2.26 writes no messages.jsonl, so any that appears later did not come
    from vibe.
    """
    if _legacy_manifest_path().exists():
        return 0
    manifest: dict[str, str] = {}
    root = _sessions_root()
    if root is not None:
        for session_dir in _session_dirs(root):
            path = session_dir / "messages.jsonl"
            if path.is_file() and not path.is_symlink():
                with contextlib.suppress(OSError):
                    manifest[str(path.relative_to(VIBE_HOME / "logs"))] = _sha256(path)
    atomic_write_json(_legacy_manifest_path(), cast("JsonDict", manifest))
    return len(manifest)


def legacy_transcripts() -> dict[str, str] | None:
    """The legacy-transcript manifest, or ``None`` before it is recorded."""
    data = read_json(_legacy_manifest_path())
    return None if data is None else {str(k): str(v) for k, v in data.items()}


def _is_trusted_legacy(path: Path, manifest: dict[str, str] | None) -> bool:
    """Whether *path* is a legacy transcript the manifest vouches for, unchanged.

    Before the manifest exists (the server records it at start, ahead of the
    agent) a file is as trusted as it always was.
    """
    if not path.is_file() or path.is_symlink():
        return False
    if manifest is None:
        return True
    try:
        rel = str(path.relative_to(VIBE_HOME / "logs"))
        return manifest.get(rel) == _sha256(path)
    except (ValueError, OSError):
        return False


def find_vibe_session_dir(session_id: str) -> Path | None:
    """Locate vibe-acp's log dir for *session_id* under ``.vibe/logs/session/``.

    vibe names directories ``session_<timestamp>_<first8-of-uuid>``, so we glob
    on the id prefix and confirm the full id via the directory's ``meta.json``.
    """
    sessions_root = _sessions_root()
    if sessions_root is None or Path(session_id).name != session_id:
        return None
    # vibe ≥2.26 (Unified Harness) keeps a chat under its full id.
    unified = sessions_root / "unified" / session_id
    if _real_dir(unified) and not (sessions_root / "unified").is_symlink():
        return unified
    short = session_id[:8]
    candidates = [d for d in sorted(sessions_root.glob(f"session_*_{short}")) if _real_dir(d)]
    for candidate in candidates:
        data = _read_session_meta(candidate)
        if data is None:
            continue
        if data.get("session_id") == session_id:
            return candidate
    # Fall back to the single prefix match even without a meta confirmation.
    return candidates[0] if candidates else None


def find_vibe_session_dirs(session_id: str, *, stop_ids: Collection[str] = ()) -> list[Path]:
    """Locate *session_id*'s log dir plus its compaction continuations, in order.

    On context compaction vibe 2.21 to 2.23 rolled the conversation over to a fresh
    session id and dir, recording the predecessor as ``parent_session_id`` in
    the new dir's ``meta.json``. Following those backlinks yields the whole
    chain — original first, then each continuation in creation order — so purge
    and distillation see one chat, not just its pre-compaction prefix. Since
    vibe 2.24 compaction happens in place (a summary envelope is appended to the
    same transcript), so new chats never grow a chain; the walk stays for the
    transcripts that already have one, and the single-dir case is just a chain
    of length one.

    ``stop_ids`` marks children that are chats in their own right and must not
    be walked into: a **fork** carries the same ``parent_session_id`` backlink
    as a compaction continuation, so without the stop set a branched chat would
    be treated as its source's continuation (resume would land in the fork,
    purge would delete it, distillation would mix it in). Callers pass the UI
    session registry's ids — a fork is registered there at creation, while a
    pure continuation never is. *session_id* itself is never a stop.
    """
    root = find_vibe_session_dir(session_id)
    sessions_root = _sessions_root()
    if root is None or sessions_root is None:
        return []
    # One scan builds the parent → children map; session counts are small.
    children: dict[str, list[tuple[Path, str]]] = {}
    for candidate in _session_dirs(sessions_root):
        data = _read_session_meta(candidate)
        if data is None:
            continue
        child_id = data.get("session_id")
        parent_id = data.get("parent_session_id")
        if isinstance(child_id, str) and isinstance(parent_id, str) and parent_id:
            children.setdefault(parent_id, []).append((candidate, child_id))
    stop = set(stop_ids) - {session_id}
    chain: list[Path] = [root]
    queue: list[str] = [session_id]
    seen: set[str] = {session_id}
    while queue:
        for child_dir, child_id in children.get(queue.pop(0), []):
            if child_id in seen or child_id in stop:
                continue  # already walked, or another chat (fork) — not ours
            seen.add(child_id)
            chain.append(child_dir)
            queue.append(child_id)
    return chain


def vibe_session_parents() -> dict[str, str]:
    """Map each vibe session id to its ``parent_session_id`` backlink.

    Only sessions that *have* a parent appear as keys. Both compaction
    continuations and explicit forks carry the backlink — callers that must
    distinguish them need an additional signal (the UI session registry: a
    chat the user can see is never a hidden continuation).
    """
    sessions_root = _sessions_root()
    if sessions_root is None:
        return {}
    parents: dict[str, str] = {}
    for candidate in _session_dirs(sessions_root):
        data = _read_session_meta(candidate)
        if data is None:
            continue
        child = data.get("session_id")
        parent = data.get("parent_session_id")
        if isinstance(child, str) and isinstance(parent, str) and parent:
            parents[child] = parent
    return parents


def vibe_chain_tip(session_id: str, *, stop_ids: Collection[str] = ()) -> str:
    """Follow compaction continuations from *session_id* to the newest link.

    Returns *session_id* itself when it has no continuation (or is unknown).
    Resuming the tip instead of the root restores the post-compaction context;
    the root dir only holds the pre-compaction prefix. ``stop_ids`` keeps the
    walk out of forks (see :func:`find_vibe_session_dirs`). With several
    children (defensive — pure compaction rolls over to at most one) the
    newest dir in the chain wins.
    """
    dirs = find_vibe_session_dirs(session_id, stop_ids=stop_ids)
    if len(dirs) < 2:
        return session_id
    data = _read_session_meta(dirs[-1])
    tip = data.get("session_id") if data is not None else None
    return tip if isinstance(tip, str) and tip else session_id


def vibe_session_title(session_id: str) -> str | None:
    """Read a session's title from its dir's ``meta.json`` (``None`` if absent)."""
    d = find_vibe_session_dir(session_id)
    if d is None:
        return None
    data = _read_session_meta(d)
    title = data.get("title") if data is not None else None
    return title if isinstance(title, str) and title.strip() else None


def vibe_manual_session_title(session_id: str) -> str | None:
    """Return the session's title if vibe records it as user-set, else ``None``.

    vibe marks a title applied through ``_session/set_title`` as
    ``title_source: "manual"`` in ``meta.json`` (its own background titling
    never overwrites one). The workspace writes every rename through, so this
    is a second record of "the user named this chat" that survives the UI
    session registry — which lives in the container layer — being reset.
    """
    d = find_vibe_session_dir(session_id)
    if d is None:
        return None
    data = _read_session_meta(d)
    if data is None or data.get("title_source") != "manual":
        return None
    title = data.get("title")
    return title if isinstance(title, str) and title.strip() else None


def list_provenance_sessions() -> list[str]:
    """Return the session ids that currently have a provenance directory."""
    root = VIBE_HOME / "provenance"
    if not root.is_dir():
        return []
    # The transcript lives in the same dir whether capture is on or off; only a
    # run log means the chat has a provenance record.
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "run.jsonl").exists())


def purge_session(session_id: str, *, stop_ids: Collection[str] = ()) -> None:
    """Delete all on-disk logs for *session_id*.

    Removes the provenance directory and every vibe transcript dir in the
    session's chain (the original plus compaction continuations, see
    :func:`find_vibe_session_dirs` — ``stop_ids`` keeps forks alive), so
    deleting a chat leaves nothing orphaned on disk. Best-effort: missing
    paths are ignored.
    """
    pdir = provenance_dir(session_id)
    if pdir.exists():
        shutil.rmtree(pdir, ignore_errors=True)
    for vibe_dir in find_vibe_session_dirs(session_id, stop_ids=stop_ids):
        shutil.rmtree(vibe_dir, ignore_errors=True)


# ── Human-readable report ────────────────────────────────────────────────────

_PATH_KEY_RE = re.compile(r"path|dir|file|output|input", re.IGNORECASE)


def _structured_paths(output: object) -> list[str]:
    """Best-effort extraction of file paths from a tool's structured output."""
    paths: list[str] = []
    candidate: JsonDict | None = None
    if isinstance(output, dict):
        candidate = cast("JsonDict", output)
    elif isinstance(output, str):
        candidate = parse_structured_blob(output) or None
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
        endpoint = model.get("endpoint")
        model_line = f"- **Model:** `{model_name}`"
        settings = (("temperature", temp), ("thinking", thinking), ("endpoint", endpoint))
        extras = [f"{k}={v}" for k, v in settings if v is not None]
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
