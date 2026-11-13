"""What a stack job is, in terms both the core and the docker guard check.

A *job* is one approved stack tool call run to completion in its own detached
container, by the entrypoint the stack's image names for that purpose (see
:mod:`medmcp.jobs` for the why and the how). This module holds only the shapes:
the declaration a stack image carries in its ``org.medmcp.stack`` label, the
labels a job container carries, and its name. Stdlib-only, so the guard — a
separate, small process — can import it.

The declaration::

    {"name": "medmcp-nnunet", "gpu": true,
     "job": {"command": ["medmcp-nnunet-job", "call"],
             "tools": ["start_training", "resume_training"],
             "shm_size": "16g", "watch_sec": 15}}

``command`` is the fixed start of the job's command line; the core appends the
tool's name and its approved arguments as one JSON argument, nothing else.
``tools`` are the only tools whose calls may become a job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

JsonDict = dict[str, Any]

# Every label a job container carries starts with this; no other container, and
# no image, may carry one (the guard refuses both).
LABEL_PREFIX = "org.medmcp.job"
JOB_LABEL = LABEL_PREFIX  # "1"
ID_LABEL = f"{LABEL_PREFIX}.id"
STACK_LABEL = f"{LABEL_PREFIX}.stack"
TOOL_LABEL = f"{LABEL_PREFIX}.tool"
# The chat whose person approved the call (empty when the call had none).
SESSION_LABEL = f"{LABEL_PREFIX}.session"
# Which workspace the job works on: a core lists and stops only its own.
WORKSPACE_LABEL = f"{LABEL_PREFIX}.workspace"
GPU_LABEL = f"{LABEL_PREFIX}.gpu"
# The image reference the stack was installed as (the container itself is
# created from the image id that reference resolved to when the job started).
IMAGE_LABEL = f"{LABEL_PREFIX}.image"
# A short name the stack gave the job, for people; never used to find anything.
NAME_LABEL = f"{LABEL_PREFIX}.name"

# The image label a stack's launch metadata lives in (as medmcp.settings.STACK_LABEL).
STACK_IMAGE_LABEL = "org.medmcp.stack"

NAME_PREFIX = "medmcp-job-"
_ID_RE = re.compile(r"[0-9a-f]{12}")
_COMMAND_PART_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,127}")
_TOOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_SIZE_RE = re.compile(r"\d{1,6}[kmgKMG]?")
_DISPLAY_RE = re.compile(r"[^A-Za-z0-9_.:=+@ -]")

MAX_WATCH_SEC = 30.0
DEFAULT_WATCH_SEC = 3.0


@dataclass(frozen=True)
class JobRunner:
    """A stack's job declaration, validated."""

    command: tuple[str, ...]
    tools: frozenset[str]
    shm_size: str | None = None
    watch_sec: float = DEFAULT_WATCH_SEC

    def manifest(self) -> JsonDict:
        """The declaration as written into ``stacks.d/<name>.toml``."""
        out: JsonDict = {"command": list(self.command), "tools": sorted(self.tools)}
        if self.shm_size:
            out["shm_size"] = self.shm_size
        if self.watch_sec != DEFAULT_WATCH_SEC:
            out["watch_sec"] = self.watch_sec
        return out


def parse_declaration(raw: object) -> JobRunner | None:
    """The ``job`` entry of a stack label or manifest, or ``None`` if it is not one.

    Strict on purpose: the command becomes the start of a container's command
    line, so a part that is not a plain word or path — an option, anything with
    a space or shell syntax — makes the whole declaration void rather than
    being cleaned up.
    """
    if not isinstance(raw, dict):
        return None
    decl = cast("JsonDict", raw)
    command: object = decl.get("command")
    tools: object = decl.get("tools")
    if not isinstance(command, list) or not isinstance(tools, list):
        return None
    parts = cast("list[object]", command)
    names = cast("list[object]", tools)
    if not 1 <= len(parts) <= 4 or not 1 <= len(names) <= 16:
        return None
    if not all(isinstance(p, str) and _COMMAND_PART_RE.fullmatch(p) for p in parts):
        return None
    if not all(isinstance(n, str) and _TOOL_RE.fullmatch(n) for n in names):
        return None
    shm: object = decl.get("shm_size")
    shm_size = shm if isinstance(shm, str) and _SIZE_RE.fullmatch(shm) else None
    watch: object = decl.get("watch_sec")
    watch_sec = DEFAULT_WATCH_SEC
    if isinstance(watch, (int, float)) and not isinstance(watch, bool):
        watch_sec = min(max(float(watch), 0.0), MAX_WATCH_SEC)
    return JobRunner(
        command=tuple(cast("list[str]", parts)),
        tools=frozenset(cast("list[str]", names)),
        shm_size=shm_size,
        watch_sec=watch_sec,
    )


def valid_id(job_id: str) -> bool:
    """Whether *job_id* has the shape the core gives job ids."""
    return bool(_ID_RE.fullmatch(job_id))


def container_name(job_id: str) -> str:
    """The name of the container that runs job *job_id*."""
    return f"{NAME_PREFIX}{job_id}"


def display_name(raw: object) -> str:
    """A stack-supplied job name made safe to store in a label and show."""
    return _DISPLAY_RE.sub("_", str(raw))[:120] if isinstance(raw, str) else ""


def carries_job_label(labels: object) -> bool:
    """Whether a label map claims, in any way, to belong to a job."""
    if not isinstance(labels, dict):
        return False
    return any(str(k).startswith(LABEL_PREFIX) for k in cast("JsonDict", labels))
