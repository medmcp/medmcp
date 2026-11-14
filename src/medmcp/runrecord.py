"""What a replay run records beyond its progress: environment, checksums, a report.

:mod:`medmcp.runs` keeps a run's progress — which item, which step, ok or not.
That is enough to follow a run and not enough to stand behind its results: it
does not say which build of a stack produced a mask, what the tool answered, or
whether the scan on disk today is the one that was read. This module gathers
those, and :class:`medmcp.runs.RunManager` writes them into the run's folder:

- :func:`environment` — the core's version and build, and for every stack the
  recipe uses the image reference **and** the digest and id of the image that is
  installed now, next to the digest the recipe pinned.
- :func:`checksum` — the SHA-256 and size of a file (or of a folder, as the hash
  of its files' hashes), bounded by a size cap.
- :func:`render_report` — the record as a Markdown document.

Nothing here needs the server, so the ``medmcp`` CLI records runs the same way.

**The checksum cap.** Hashing is I/O the run does not otherwise need, done after
each item off the event loop. A scan is tens to hundreds of megabytes and hashes
in well under a second; a folder of 4-D data can be tens of gigabytes. So a path
larger than :data:`HASH_MAX_BYTES` (2 GiB by default, ``MEDMCP_RUN_HASH_MAX_MB``
to change, ``0`` to turn checksums off) is recorded with its size and the reason
it was not hashed, never silently left out.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from medmcp import provenance
from medmcp.workflow import Recipe

JsonDict = dict[str, Any]

_DEFAULT_HASH_MAX_MB = 2048
_CHUNK = 1 << 20

ImageIdentity = provenance.ImageIdentity


def hash_max_bytes() -> int:
    """The largest path that is hashed, in bytes; ``0`` means checksums are off."""
    raw = os.environ.get("MEDMCP_RUN_HASH_MAX_MB", "").strip()
    try:
        megabytes = int(raw) if raw else _DEFAULT_HASH_MAX_MB
    except ValueError:
        megabytes = _DEFAULT_HASH_MAX_MB
    return max(0, megabytes) * 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum(path: str | Path, *, max_bytes: int | None = None) -> JsonDict:
    r"""The SHA-256 and size of *path*, or why it was not hashed.

    A file is hashed as it is. A folder (a DICOM series) is hashed as the
    SHA-256 over one ``<relative name>\0<file sha256>\n`` line per file, in name
    order, so the value changes when any file is added, removed, renamed or
    altered; symlinks are not followed. Blocking: call it off the event loop.

    Returns ``{"path", "kind", "bytes", "sha256"}`` (plus ``"files"`` for a
    folder), with ``"sha256": None`` and a ``"skipped"`` reason when the path is
    missing, unreadable, over the cap, or checksums are off.
    """
    cap = hash_max_bytes() if max_bytes is None else max_bytes
    target = Path(path)
    entry: JsonDict = {"path": str(path), "kind": "file", "bytes": None, "sha256": None}
    try:
        if target.is_symlink() or not target.exists():
            entry["skipped"] = "not found"
            return entry
        if target.is_dir():
            entry["kind"] = "dir"
            members = sorted(p for p in target.rglob("*") if p.is_file() and not p.is_symlink())
            total = sum(p.stat().st_size for p in members)
            entry["bytes"], entry["files"] = total, len(members)
        else:
            members = []
            total = target.stat().st_size
            entry["bytes"] = total
        if cap <= 0:
            entry["skipped"] = "checksums are turned off"
            return entry
        if total > cap:
            entry["skipped"] = f"larger than the {cap // (1024 * 1024)} MB checksum cap"
            return entry
        if entry["kind"] == "file":
            entry["sha256"] = _file_sha256(target)
        else:
            tree = hashlib.sha256()
            for member in members:
                name = member.relative_to(target).as_posix()
                tree.update(f"{name}\0{_file_sha256(member)}\n".encode())
            entry["sha256"] = tree.hexdigest()
    except OSError as exc:
        entry["skipped"] = f"unreadable: {exc.strerror or exc}"
    return entry


def environment(
    recipe: Recipe, servers: list[JsonDict], *, image_identity: ImageIdentity | None = None
) -> JsonDict:
    """What a run of *recipe* ran on: the core, and each stack the recipe uses.

    For a stack: the image reference its launch config names, the digest and id
    of the image present under that reference now (``image_identity``; a locally
    built image has an id but no registry digest), and the digest the recipe
    pinned at distillation with whether the two agree. Blocking (one
    ``docker image inspect`` per stack): call it off the event loop.
    """
    by_name = {str(s.get("name")): s for s in servers}
    pinned = {r.stack: r for r in recipe.requires}
    stacks: list[JsonDict] = []
    for name in sorted({s.server for s in recipe.steps if s.server != "builtin"}):
        entry: JsonDict = {"name": name, "installed": name in by_name}
        server = by_name.get(name)
        if server is not None:
            entry["version"] = server.get("version")
            image = provenance.docker_image_ref(server)
            if image:
                entry["image"] = image
                identity = image_identity(image) if image_identity is not None else {}
                entry["digest"] = identity.get("digest")
                entry["image_id"] = identity.get("id")
        pin = pinned.get(name)
        if pin is not None and pin.digest:
            entry["pinned_digest"] = pin.digest
            if entry.get("digest"):
                entry["matches_pin"] = entry["digest"] == pin.digest
        stacks.append(entry)
    return {
        "medmcp": provenance.core_identity(),
        "stacks": stacks,
        # A replay calls the tools directly; no model takes part. The model that
        # wrote the workflow is in the recipe's ``source`` when it was recorded.
        "model": None,
        "recorded_model": recipe.source.get("model") or None,
        "platform": provenance.platform_identity(),
        "checksum_cap_bytes": hash_max_bytes(),
    }


# ── A readable report ─────────────────────────────────────────────────────────


def _seconds(started: str, finished: str) -> float | None:
    try:
        delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    except ValueError:
        return None
    return round(delta.total_seconds(), 3)


def _fmt_checksum(entry: JsonDict) -> str:
    sha = entry.get("sha256")
    size = entry.get("bytes")
    detail = f"sha256 `{sha}`" if sha else f"not hashed ({entry.get('skipped', 'unknown')})"
    return f"`{entry.get('path')}` — {size if size is not None else '?'} bytes, {detail}"


def render_report(record: JsonDict) -> str:
    """Render a full run record (``RunRecord.to_dict(full=True)``) as Markdown.

    The same register as a chat's provenance report: environment first, then
    what happened per item, with the tool's structured result for every step.
    """
    import json

    lines: list[str] = ["# MedMCP run report\n", f"**Run:** `{record.get('id')}`\n"]
    lines.append(f"- **Workflow:** `{record.get('workflow')}`")
    lines.append(f"- **Status:** {record.get('status')}")
    lines.append(f"- **Started:** {record.get('started_at')}")
    lines.append(f"- **Finished:** {record.get('finished_at') or '—'}")
    lines.append(
        f"- **Items:** {record.get('total')} "
        f"({record.get('succeeded')} succeeded, {record.get('failed')} failed)"
    )
    manifest = cast("JsonDict", record.get("manifest") or {})
    if manifest:
        lines.append(
            f"- **Manifest:** `{manifest.get('path')}` (sha256 `{manifest.get('sha256')}`)"
        )
        selection = cast("JsonDict", manifest.get("confirmed_selection") or {})
        if selection:
            source = cast("JsonDict", selection.get("source_manifest") or {})
            index = cast("JsonDict", selection.get("index") or {})
            lines.append(
                f"- **Confirmed selection:** `{selection.get('path')}` "
                f"(sha256 `{selection.get('sha256')}`), confirmed at "
                f"{selection.get('confirmed_at') or '?'} under review rule set "
                f"{selection.get('rule_version') or '?'}, {selection.get('overrides', 0)} "
                f"override(s), {selection.get('excluded', 0)} subject-session(s) left out"
            )
            lines.append(
                f"  - from the selection `{source.get('path') or '?'}` "
                f"(sha256 `{source.get('sha256') or '?'}`) of the index "
                f"`{index.get('path') or '?'}` (sha256 `{index.get('sha256') or '?'}`)"
            )
            if selection.get("error"):
                lines.append(f"  - **Note:** {selection.get('error')}")
    if record.get("output_root"):
        lines.append(f"- **Output folder:** `{record.get('output_root')}`")
    confirmation = cast("JsonDict", record.get("confirmation") or {})
    if confirmation:
        how = "after the preview" if confirmation.get("previewed") else "without a preview"
        lines.append(
            f"- **Confirmed:** {how}, by {confirmation.get('by', '?')} "
            f"at {confirmation.get('confirmed_at', '?')}"
        )

    env = cast("JsonDict", record.get("environment") or {})
    if env:
        core = cast("JsonDict", env.get("medmcp") or {})
        lines.append("\n## Environment\n")
        lines.append(
            f"- **medmcp:** {core.get('version') or '?'} "
            f"(build `{core.get('build') or '?'}`, commit `{core.get('git_commit') or '?'}`)"
        )
        for stack in cast("list[JsonDict]", env.get("stacks") or []):
            line = f"- **{stack.get('name')}:** `{stack.get('image') or 'not a container'}`"
            if stack.get("digest"):
                line += f" @ `{stack.get('digest')}`"
            elif stack.get("image_id"):
                line += f" (no registry digest; image id `{stack.get('image_id')}`)"
            if stack.get("matches_pin") is False:
                line += f" — differs from the recorded `{stack.get('pinned_digest')}`"
            lines.append(line)
        plat = cast("JsonDict", env.get("platform") or {})
        lines.append(
            f"- **Platform:** {plat.get('platform', '?')}, Python {plat.get('python', '?')}"
        )

    items = cast("list[JsonDict]", record.get("items") or [])
    lines.append(f"\n## Items ({len(items)})\n")
    for index, item in enumerate(items, start=1):
        state = {True: "ok", False: "failed", None: "not run"}[cast("bool | None", item.get("ok"))]
        lines.append(f"### {index}. {item.get('label') or f'item {index}'} — {state}\n")
        if item.get("error"):
            lines.append(f"> {str(item.get('error')).splitlines()[0]}\n")
        for entry in cast("list[JsonDict]", item.get("input_files") or []):
            lines.append(f"- in: {_fmt_checksum(entry)}")
        for entry in cast("list[JsonDict]", item.get("output_files") or []):
            lines.append(f"- out: {_fmt_checksum(entry)}")
        for step in cast("list[JsonDict]", item.get("steps") or []):
            took = _seconds(str(step.get("started_at", "")), str(step.get("finished_at", "")))
            lines.append(
                f"\n**Step {step.get('index')}** `{step.get('server')}:{step.get('tool')}` — "
                f"{'ok' if step.get('ok') else 'failed'}"
                + (f", {took} s" if took is not None else "")
            )
            lines.append(
                "```json\n" + json.dumps(step.get("arguments"), indent=2, default=str) + "\n```"
            )
            if step.get("error"):
                lines.append(f"Error: {str(step.get('error')).splitlines()[0]}")
            if step.get("result") is not None:
                lines.append(
                    "Result:\n```json\n"
                    + json.dumps(step.get("result"), indent=2, default=str)
                    + "\n```"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
