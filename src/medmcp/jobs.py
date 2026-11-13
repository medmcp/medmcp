"""Stack jobs: an approved tool call that outlives the pool, the chat and the core.

Every stack tool runs in the :class:`~medmcp.backend_pool.BackendPool`, whose
containers are removed when idle (two minutes), when another GPU stack needs the
slot, and when the core stops. That is right for a tool call and wrong for work
that takes hours — an nnU-Net training started by ``start_training`` died about
two minutes after the last tool call. A *job* is the answer: the core starts one
more container from the same stack image, with the same isolation and the same
single mount as the pooled one, detached and outside the pool, and leaves it
running until its work is done or someone stops it.

How a job comes to be, and what keeps each step honest:

1. The stack's image says, in its ``org.medmcp.stack`` label, which fixed
   entrypoint runs its jobs and which of its tools may ask for one
   (:mod:`medmcp.jobspec`). Read from the image when the job starts — not from
   the ``stacks.d`` manifest, which is a file — and checked again by the docker
   guard against the same image.
2. The model calls such a tool (``start_training``); the person approves that
   call, with those arguments, in the dialog as always.
3. The broker spends the approval and runs the call in the pooled stack, which
   validates it and answers with a ``medmcp_job`` entry in its result instead of
   starting the work itself (it does that only when the core offered:
   ``MEDMCP_JOBS=1`` in its environment).
4. The broker hands that result here (:func:`after_call`). The request can only
   arrive this way — on the result of a call whose approval was just spent — so
   nothing that merely reaches the broker socket can ask for a job.
5. The job container runs ``<entrypoint> <tool> <arguments as JSON>``: the name
   and arguments are the approved call's, byte for byte as the ledger matched
   them. Neither the model nor the stack's answer contributes a word to the
   command line; the answer carries only a display name.

A job has no network (also for a stack that was granted egress: a container that
holds the workspace for days is not given a way out), no capabilities beyond the
pooled stack's, a task limit, and the workspace as its only mount.

Jobs are not kept in any file. A job *is* its container, found by label, so a
restarted or updated core sees the jobs that were running before it, and nothing
the agent can write to decides what counts as one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
import logging
import re
import secrets
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import mcp.types as mcp_types

from medmcp import jobspec, posture
from medmcp.jobspec import JobRunner

log: logging.Logger = logging.getLogger(__name__)
_audit: logging.Logger = logging.getLogger("medmcp.audit")

JsonDict = dict[str, Any]

# Set in a pooled stack's environment when the core will act on a job request.
OFFER_ENV = "MEDMCP_JOBS"
# The key, in a tool's structured result, under which a stack asks for a job and
# under which the core answers.
RESULT_KEY = "medmcp_job"
# Told to the job which job it is.
JOB_ID_ENV = "MEDMCP_JOB_ID"
# Where the job may write its progress: a small JSON file in the workspace,
# ``{"fraction": 0.41, "text": "epoch 41/100"}``. Optional for a job; read by
# the core for the jobs list, and treated as untrusted (see read_progress).
PROGRESS_ENV = "MEDMCP_JOB_PROGRESS"
_PROGRESS_TEXT_MAX = 120

# Largest /dev/shm a stack's declaration can ask for (a tmpfs cap, counted
# against the host's memory once used).
MAX_SHM_BYTES = 64 * 1024**3
# One argv element; far below the kernel's limit, far above any real call.
MAX_ARGUMENT_BYTES = 64 * 1024
# Finished job containers kept for the list; older ones are removed at the next start.
KEEP_FINISHED = 20
STOP_GRACE_SEC = 30

_DROPPED_FLAGS = frozenset({"--rm", "-i", "--interactive"})
_DROPPED_VALUE_FLAGS = frozenset({"--name", "--network", "--entrypoint"})
_REQUIRED_FLAGS = ("--cap-drop", "--security-opt", "--pids-limit")
_SIZE_UNITS = {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
_NO_TIME = "0001-01-01T00:00:00Z"

DockerFn = Callable[[list[str], float], "subprocess.CompletedProcess[str]"]


class JobError(Exception):
    """A job that was not started, stopped or found; the message is for the person."""


def _docker(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run the ``docker`` CLI (list args, no shell); raise :class:`JobError` on failure."""
    try:
        result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise JobError("docker is not available") from exc
    except subprocess.TimeoutExpired as exc:
        raise JobError(f"docker {args[0]} timed out") from exc
    if result.returncode != 0:
        raise JobError(result.stderr.strip() or f"docker {args[0]} failed")
    return result


def _size_bytes(value: str) -> int | None:
    match = re.fullmatch(r"(\d+)([bkmgBKMG]?)", value.strip())
    return int(match.group(1)) * _SIZE_UNITS[match.group(2).lower()] if match else None


def offer_args(args: list[str]) -> list[str]:
    """A pooled stack's ``docker run`` *args* with the job offer in its environment."""
    if not args or args[0] != "run":
        return args
    return ["run", "-e", f"{OFFER_ENV}=1", *args[1:]]


def build_run_args(
    server_args: list[str],
    runner: JobRunner,
    *,
    image: str,
    labels: dict[str, str],
    job_id: str,
    tool: str,
    arguments: str,
    progress_file: str | None = None,
) -> list[str]:
    """The ``docker run`` that starts a job, from the stack's own (hardened) args.

    Keeps the stack's isolation flags, limits, GPU device and workspace mount;
    drops what makes it an interactive, removable, possibly networked server;
    and ends in the declared entrypoint, the tool and its arguments.

    Raises:
        JobError: the stack is not a hardened container stack.
    """
    split = posture.split_run_args(server_args)
    if split is None:
        raise JobError("this stack does not run as a container, so it cannot run jobs")
    flags, _, _ = split
    if any(flag not in flags for flag in _REQUIRED_FLAGS):
        raise JobError("this stack's container is not hardened; refusing to run a job from it")
    kept: list[str] = []
    index = 0
    while index < len(flags):
        flag = flags[index]
        if flag in _DROPPED_FLAGS:
            index += 1
        elif flag in _DROPPED_VALUE_FLAGS:
            index += 2
        elif flag == "--init":
            kept.append(flag)
            index += 1
        else:
            value = flags[index + 1] if index + 1 < len(flags) else ""
            if flag == "--shm-size" and runner.shm_size:
                have, want = _size_bytes(value), _size_bytes(runner.shm_size)
                if have is not None and want is not None and have < want <= MAX_SHM_BYTES:
                    value = runner.shm_size
            kept += [flag, value]
            index += 2
    out = ["run", "-d", "--network", "none", "--name", jobspec.container_name(job_id)]
    for key in sorted(labels):
        out += ["--label", f"{key}={labels[key]}"]
    out += ["-e", f"{JOB_ID_ENV}={job_id}"]
    if progress_file:
        out += ["-e", f"{PROGRESS_ENV}={progress_file}"]
    out += [*kept, image, *runner.command, tool, arguments]
    return out


def progress_path(workspace: str, job_id: str) -> Path:
    """The progress file a job may write: ``<workspace>/.medmcp/jobs/<id>.json``."""
    return Path(workspace) / ".medmcp" / "jobs" / f"{job_id}.json"


def read_progress(workspace: str, job_id: str) -> JsonDict | None:
    """A job's last progress report, reduced to what the UI shows.

    The file is written by the stack, so nothing is taken on trust: only a
    fraction clamped to 0..1 and a single line of text, cut to a short length,
    survive; anything else, or a file that does not parse, is ignored.
    """
    path = progress_path(workspace, job_id)
    try:
        if path.stat().st_size > 4096:
            return None
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        updated = path.stat().st_mtime
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    data = cast("JsonDict", raw)
    out: JsonDict = {"updated_at": updated}
    fraction = data.get("fraction")
    if isinstance(fraction, int | float) and not isinstance(fraction, bool):
        out["fraction"] = min(1.0, max(0.0, float(fraction)))
    text = data.get("text")
    if isinstance(text, str):
        line = " ".join(text.split())
        out["text"] = line[:_PROGRESS_TEXT_MAX]
    return out if len(out) > 1 else None


@dataclass(frozen=True)
class Job:
    """One job container, as the daemon reports it."""

    id: str
    stack: str
    tool: str
    session: str
    name: str
    gpu: bool
    image: str
    image_id: str
    state: str
    exit_code: int | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    arguments: JsonDict | None

    @property
    def running(self) -> bool:
        """Whether the container is (about to be) running."""
        return self.state in {"created", "running", "restarting", "paused"}

    @property
    def status(self) -> str:
        """``running``, ``finished`` (exit 0), ``stopped`` (by a signal) or ``failed``."""
        if self.running:
            return "running"
        if self.exit_code == 0:
            return "finished"
        return "stopped" if self.exit_code in {137, 143} else "failed"

    def public(self) -> JsonDict:
        """The job as the API and a tool result show it."""
        return {
            "id": self.id,
            "stack": self.stack,
            "tool": self.tool,
            "name": self.name,
            "session": self.session,
            "gpu": self.gpu,
            "image": self.image,
            "image_id": self.image_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "arguments": self.arguments,
            "container": jobspec.container_name(self.id),
        }


def _job_from_record(record: JsonDict, workspace: str) -> Job | None:
    """A ``docker container inspect`` record as a :class:`Job`, if it is one of ours.

    The labels alone do not make a job: the name must be the one the core gives
    the job with that id, which no stack manifest can ask for.
    """
    config = cast("JsonDict", record.get("Config") or {})
    labels = cast("JsonDict", config.get("Labels") or {})
    job_id = str(labels.get(jobspec.ID_LABEL) or "")
    if labels.get(jobspec.JOB_LABEL) != "1" or not jobspec.valid_id(job_id):
        return None
    if str(record.get("Name") or "").lstrip("/") != jobspec.container_name(job_id):
        return None
    if labels.get(jobspec.WORKSPACE_LABEL) != workspace:
        return None
    state = cast("JsonDict", record.get("State") or {})
    status = str(state.get("Status") or "")
    exited = status in {"exited", "dead"}
    code = state.get("ExitCode")
    arguments: JsonDict | None = None
    cmd = cast("list[Any]", config.get("Cmd") or [])
    if cmd:
        try:
            parsed: object = json.loads(str(cmd[-1]))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            arguments = cast("JsonDict", parsed)

    def when(key: str) -> str | None:
        value = str(state.get(key) or "")
        return value if value and value != _NO_TIME else None

    return Job(
        id=job_id,
        stack=str(labels.get(jobspec.STACK_LABEL) or ""),
        tool=str(labels.get(jobspec.TOOL_LABEL) or ""),
        session=str(labels.get(jobspec.SESSION_LABEL) or ""),
        name=str(labels.get(jobspec.NAME_LABEL) or ""),
        gpu=labels.get(jobspec.GPU_LABEL) == "1",
        image=str(labels.get(jobspec.IMAGE_LABEL) or ""),
        image_id=str(record.get("Image") or ""),
        state=status,
        exit_code=int(code) if exited and isinstance(code, int) else None,
        created_at=str(record.get("Created") or ""),
        started_at=when("StartedAt"),
        finished_at=when("FinishedAt") if exited else None,
        arguments=arguments,
    )


@dataclass(frozen=True)
class Started:
    """A job that was just started, with what provenance records about it."""

    job: Job
    image_digest: str | None
    command: list[str]


class JobManager:
    """Starts, lists and stops this workspace's job containers (blocking calls)."""

    def __init__(
        self,
        *,
        workspace: str,
        resolve_server: Callable[[str], JsonDict | None],
        max_jobs: Callable[[], int],
        docker: DockerFn = _docker,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Manage the jobs of *workspace*.

        Args:
            workspace: The workspace path jobs mount and are labelled with.
            resolve_server: An active stack's server entry (``command``, ``args``)
                by name, or ``None``.
            max_jobs: How many jobs may run at once on this machine.
            docker: Runs the docker CLI (injected by tests).
            sleep: Waits (injected by tests).
        """
        self._workspace = workspace
        self._resolve = resolve_server
        self._max_jobs = max_jobs
        self._docker = docker
        self._sleep = sleep
        self._lock = threading.Lock()

    @property
    def workspace(self) -> str:
        """The workspace these jobs belong to."""
        return self._workspace

    # ── reading ──────────────────────────────────────────────────────────────

    def list(self) -> list[Job]:
        """This workspace's jobs, running and finished, newest first."""
        listed = self._docker(
            [
                "ps",
                "-aq",
                "--no-trunc",
                "--filter",
                f"label={jobspec.JOB_LABEL}=1",
                "--filter",
                f"label={jobspec.WORKSPACE_LABEL}={self._workspace}",
            ],
            30,
        )
        ids = listed.stdout.split()
        if not ids:
            return []
        # A container removed between the two calls fails the whole inspect.
        try:
            raw = self._docker(["container", "inspect", *ids], 30).stdout
        except JobError:
            return []
        try:
            records: object = json.loads(raw)
        except json.JSONDecodeError:
            return []
        jobs = [
            job
            for record in cast("list[JsonDict]", records if isinstance(records, list) else [])
            if (job := _job_from_record(record, self._workspace)) is not None
        ]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> Job:
        """The job *job_id*.

        Raises:
            JobError: there is no such job in this workspace.
        """
        if jobspec.valid_id(job_id):
            for job in self.list():
                if job.id == job_id:
                    return job
        raise JobError(f"no job {job_id!r}")

    def logs(self, job_id: str, tail: int = 200) -> str:
        """The last *tail* lines the job's container wrote (stdout and stderr)."""
        job = self.get(job_id)
        result = self._docker(
            ["logs", "--tail", str(max(1, min(tail, 2000))), jobspec.container_name(job.id)], 30
        )
        return result.stdout + result.stderr

    def running_on_machine(self) -> int:
        """How many job containers are running on this daemon, any workspace's.

        The limit is about the machine's GPU and memory, which a second MedMCP
        instance on the same host shares.
        """
        listed = self._docker(["ps", "-q", "--filter", f"label={jobspec.JOB_LABEL}=1"], 30)
        return len(listed.stdout.split())

    # ── starting ─────────────────────────────────────────────────────────────

    def _image(self, image: str) -> tuple[str, str | None, JobRunner | None, str]:
        """(image id, registry digest, job declaration, stack name) of a present image."""
        try:
            raw = self._docker(["image", "inspect", "--format", "{{json .}}", image], 30).stdout
            record = cast("JsonDict", json.loads(raw))
        except (JobError, json.JSONDecodeError) as exc:
            raise JobError(f"the stack's image {image} cannot be read: {exc}") from exc
        labels = cast("JsonDict", cast("JsonDict", record.get("Config") or {}).get("Labels") or {})
        try:
            meta: object = json.loads(str(labels.get(jobspec.STACK_IMAGE_LABEL) or ""))
        except json.JSONDecodeError:
            meta = None
        stack_meta = cast("JsonDict", meta) if isinstance(meta, dict) else {}
        digests = [str(d) for d in cast("list[Any]", record.get("RepoDigests") or [])]
        digest = digests[0].partition("@")[2] if digests else None
        return (
            str(record.get("Id") or ""),
            digest or None,
            jobspec.parse_declaration(stack_meta.get("job")),
            str(stack_meta.get("name") or ""),
        )

    def declares(self, stack: str, tool: str) -> bool:
        """Whether *stack*'s manifest says *tool* may ask for a job (cheap; not the authority)."""
        server = self._resolve(stack)
        runner = jobspec.parse_declaration(server.get("job")) if server else None
        return runner is not None and tool in runner.tools

    def start(
        self, stack: str, tool: str, arguments: JsonDict, *, session: str = "", name: str = ""
    ) -> Started:
        """Start the job for an approved call of *tool* on *stack*.

        Raises:
            JobError: the stack declares no such job, the limit is reached, or
                the container did not start.
        """
        server = self._resolve(stack)
        if server is None or str(server.get("command") or "").rsplit("/", 1)[-1] != "docker":
            raise JobError(f"{stack} is not an installed container stack")
        server_args = [str(a) for a in cast("list[Any]", server.get("args") or [])]
        split = posture.split_run_args(server_args)
        if split is None:
            raise JobError(f"{stack} is not an installed container stack")
        flags, image_ref, _ = split
        encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        if len(encoded.encode()) > MAX_ARGUMENT_BYTES:
            raise JobError("the call's arguments are too large to hand to a job")

        with self._lock:
            image_id, digest, runner, image_stack = self._image(image_ref)
            if runner is None or image_stack != stack or not image_id:
                raise JobError(f"{stack}'s image does not declare a job runner")
            if tool not in runner.tools:
                raise JobError(f"{stack}'s image does not run {tool} as a job")
            limit = max(1, self._max_jobs())
            running = self.running_on_machine()
            if running >= limit:
                raise JobError(
                    f"{running} background job{'s are' if running != 1 else ' is'} already "
                    f"running and this machine runs at most {limit} at a time. Wait for it "
                    "to finish or stop it first; nothing was queued."
                )
            self._prune()
            job_id = secrets.token_hex(6)
            gpu = any("nvidia.com/gpu" in f for f in flags)
            labels = {
                jobspec.JOB_LABEL: "1",
                jobspec.ID_LABEL: job_id,
                jobspec.STACK_LABEL: stack,
                jobspec.TOOL_LABEL: tool,
                jobspec.SESSION_LABEL: jobspec.display_name(session),
                jobspec.WORKSPACE_LABEL: self._workspace,
                jobspec.GPU_LABEL: "1" if gpu else "0",
                jobspec.IMAGE_LABEL: image_ref,
                jobspec.NAME_LABEL: jobspec.display_name(name),
            }
            # By id: the image that was just read is the image that runs, even if
            # the tag is moved in between.
            progress = progress_path(self._workspace, job_id)
            try:
                progress.parent.mkdir(parents=True, exist_ok=True)
                progress_file: str | None = str(progress)
            except OSError:
                progress_file = None  # no progress, but the job still runs
            run_args = build_run_args(
                server_args,
                runner,
                image=image_id,
                labels=labels,
                job_id=job_id,
                tool=tool,
                arguments=encoded,
                progress_file=progress_file,
            )
            self._docker(run_args, 120)
            _audit.info(
                "job %s started: %s.%s image=%s session=%s", job_id, stack, tool, image_id, session
            )
        job = self._watch(job_id, runner.watch_sec)
        return Started(job=job, image_digest=digest, command=[*runner.command, tool, encoded])

    def _watch(self, job_id: str, watch_sec: float) -> Job:
        """Follow a new job briefly, so one that dies at once is reported as that."""
        deadline = time.monotonic() + watch_sec
        job = self.get(job_id)
        while job.running and time.monotonic() < deadline:
            self._sleep(0.5)
            job = self.get(job_id)
        if not job.running and job.exit_code != 0:
            tail = ""
            with contextlib.suppress(JobError):
                tail = self.logs(job_id, 20).strip()
            _audit.warning("job %s exited at once with code %s", job_id, job.exit_code)
            raise JobError(
                f"the job ended right after it started (exit code {job.exit_code})"
                + (f": {tail[-800:]}" if tail else "")
            )
        return job

    def _prune(self) -> None:
        """Remove finished job containers beyond the newest :data:`KEEP_FINISHED`."""
        finished = [j for j in self.list() if not j.running]
        for job in finished[KEEP_FINISHED:]:
            try:
                self._docker(["rm", jobspec.container_name(job.id)], 30)
            except JobError as exc:
                log.debug("could not remove finished job %s: %s", job.id, exc)

    # ── stopping ─────────────────────────────────────────────────────────────

    def stop(self, job_id: str) -> Job:
        """Stop a running job (SIGTERM, then SIGKILL after a grace period)."""
        job = self.get(job_id)
        if job.running:
            self._docker(
                ["stop", "--time", str(STOP_GRACE_SEC), jobspec.container_name(job.id)],
                STOP_GRACE_SEC + 30,
            )
            _audit.info("job %s stopped on request", job.id)
        return self.get(job_id)

    def remove(self, job_id: str) -> None:
        """Remove a finished job's container.

        Raises:
            JobError: the job is still running (stop it first).
        """
        job = self.get(job_id)
        if job.running:
            raise JobError(f"job {job_id} is still running; stop it first")
        self._docker(["rm", jobspec.container_name(job.id)], 30)


# ── the tool result ──────────────────────────────────────────────────────────


def _json_object(text: str) -> JsonDict | None:
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return None
    return cast("JsonDict", parsed) if isinstance(parsed, dict) else None


def find_request(result: mcp_types.CallToolResult) -> JsonDict | None:
    """The job request in a stack's result, or ``None``.

    A stack asks by putting an object under :data:`RESULT_KEY` at the top of its
    structured result (which FastMCP also renders as one JSON text block).
    """
    candidates: list[JsonDict] = []
    if isinstance(result.structuredContent, dict):
        candidates.append(result.structuredContent)
    for block in result.content:
        if isinstance(block, mcp_types.TextContent):
            parsed = _json_object(block.text)
            if parsed is not None:
                candidates.append(parsed)
    for candidate in candidates:
        request = candidate.get(RESULT_KEY)
        if isinstance(request, dict):
            return cast("JsonDict", request)
    return None


def with_outcome(result: mcp_types.CallToolResult, outcome: JsonDict) -> mcp_types.CallToolResult:
    """*result* with the stack's request replaced by what the core did about it."""
    structured = result.structuredContent
    if isinstance(structured, dict) and RESULT_KEY in structured:
        structured = {**structured, RESULT_KEY: outcome}
    content: list[mcp_types.ContentBlock] = []
    for block in result.content:
        parsed = _json_object(block.text) if isinstance(block, mcp_types.TextContent) else None
        if parsed is not None and RESULT_KEY in parsed:
            text = json.dumps({**parsed, RESULT_KEY: outcome}, indent=2)
            content.append(mcp_types.TextContent(type="text", text=text))
        else:
            content.append(block)
    return result.model_copy(update={"content": content, "structuredContent": structured})


def refused(reason: str) -> mcp_types.CallToolResult:
    """The result of a call whose job the core did not start."""
    text = f"MedMCP did not start the background job: {reason}"
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)], isError=True
    )


async def after_call(
    manager: JobManager,
    stack: str,
    tool: str,
    arguments: JsonDict,
    session: str,
    result: mcp_types.CallToolResult,
    on_started: Callable[[Started], None] | None = None,
) -> mcp_types.CallToolResult:
    """Start the job an approved call's result asks for, and say so in the result.

    A result without a request is returned untouched, as is a failed call (it
    starts nothing). *on_started* is run for a started job (provenance).
    """
    request = find_request(result)
    if request is None or result.isError:
        return result
    try:
        started = await asyncio.to_thread(
            lambda: manager.start(
                stack, tool, arguments, session=session, name=str(request.get("name") or "")
            )
        )
    except JobError as exc:
        _audit.warning("job for %s.%s not started: %s", stack, tool, exc)
        return refused(str(exc))
    except Exception as exc:  # a failure here must reach the model as a failure
        log.exception("job for %s.%s failed to start", stack, tool)
        return refused(str(exc))
    if on_started is not None:
        try:
            on_started(started)
        except Exception:
            log.warning("could not record job %s", started.job.id, exc_info=True)
    job = started.job
    return with_outcome(
        result,
        {
            "started": True,
            "id": job.id,
            "status": job.status,
            "note": (
                "MedMCP runs this as a background job in its own container. It keeps "
                "running when the chat is idle or closed and when MedMCP restarts, until "
                "it finishes or is stopped."
            ),
        },
    )
