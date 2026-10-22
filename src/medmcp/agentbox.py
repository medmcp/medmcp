"""The agent in its own container (``MEDMCP_AGENT_ISOLATION=container``).

In the core container everything ran as one user: the server, the agent and
every command the agent was allowed to run. That user holds the docker socket,
the state volume (external-MCP and cloud-model secrets, the access token) and the
files that decide what the agent may do without asking. One approved ``bash``
call could therefore reach all of it.

With isolation on, the core starts vibe-acp as a sibling container instead of a
subprocess, through the same socket it starts stacks with, and speaks the same
JSON-RPC to it over the ``docker run -i`` stdio. That container gets:

- the workspace, read-write, at its host path (as the stacks do);
- the agent home — ``config.toml``, the agent profile, ``hooks.toml``, the
  prompts, the trusted-folder list — **read-only**, published by the core from
  its own ``.vibe`` after every config sync (:func:`publish_home`);
- the session logs, read-write, where the core reads transcripts back;
- ``stacks.d`` read-only, for the skills it names;
- the broker socket, inside the agent home, through which every stack tool runs
  in the core's pool — so the core stays the only process that starts containers;
- an internal network shared with the core and nothing else: the model, a cloud
  model and external MCP servers are reached through the core's egress gateway
  (:mod:`medmcp.gateway`), which holds their credentials.

No docker socket, no state volume, no provenance or workflow volumes, no docker
credentials, all capabilities dropped but ``DAC_OVERRIDE`` (a rootful host's
workspace is owned by someone else), no privilege escalation, a read-only root
file system. A host-native install has no second user to give the agent and keeps
the subprocess; :mod:`medmcp.posture` is what guards it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from medmcp.acp import VIBE_HOME, JsonDict
from medmcp.statefile import atomic_write_toml

log: logging.Logger = logging.getLogger(__name__)

ISOLATION_ENV = "MEDMCP_AGENT_ISOLATION"
# The agent home, on its own volume, at the same path in both containers so the
# paths written into config.toml (the broker socket, the prompts) hold in both.
AGENT_HOME: Path = VIBE_HOME / "agent-home"
# The files of the core's .vibe the agent reads. Nothing else is published.
_HOME_FILES: tuple[str, ...] = ("config.toml", "hooks.toml", "trusted_folders.toml")
_HOME_DIRS: tuple[str, ...] = ("agents", "prompts")
# Label naming the core that started an agent container, so a restarted core
# removes its own leftovers and never another core's.
OWNER_LABEL = "org.medmcp.agent-of"
# Variables passed on to the agent: its own configuration, the model's, the
# locale. Credentials it is owed are added by name (the client's extra_env).
_FORWARD_PREFIXES: tuple[str, ...] = ("MEDMCP_", "OLLAMA_", "LC_")
_FORWARD_NAMES: frozenset[str] = frozenset({"LANG", "TZ"})
# Where the path guard keeps its retry ledger: the agent home is read-only.
_PATHGUARD_LEDGER = "/tmp/pathguard_retries.json"

# The egress gateway's base URL once it listens (set by the server at start); the
# agent's copy of config.toml names it instead of the real endpoints.
_gateway_base: str | None = None


def set_gateway_base(url: str | None) -> None:
    """Record where the agent reaches the egress gateway."""
    global _gateway_base
    _gateway_base = url


def gateway_base() -> str | None:
    """Where the agent reaches the egress gateway, once it listens."""
    return _gateway_base


def isolation_enabled() -> bool:
    """Whether the agent runs in its own container (set by both compose files)."""
    return os.environ.get(ISOLATION_ENV, "").strip().lower() == "container"


class IsolationUnavailableError(RuntimeError):
    """Isolation is on but the core cannot say what to give the agent container."""


# ── the agent home ───────────────────────────────────────────────────────────


def _copy_atomic(src: Path, dst: Path) -> None:
    """Copy *src* over *dst* so a reader never sees half a file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
            shutil.copyfileobj(inp, out)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def publish_home(
    vibe_home: Path = VIBE_HOME, agent_home: Path = AGENT_HOME, config: JsonDict | None = None
) -> None:
    """Mirror what the agent reads from *vibe_home* into *agent_home*.

    Runs after every config sync. Files are replaced atomically; a file gone from
    the source is removed from the copy, so an agent profile or prompt the core
    dropped does not linger where vibe would still find it. ``logs/`` is created
    as the mount point for the session-log volume. *config*, when given, is
    written as the agent's ``config.toml`` instead of a copy of the core's (the
    gateway's view of it, see :func:`medmcp.gateway.agent_config`).
    """
    agent_home.mkdir(parents=True, exist_ok=True)
    (agent_home / "logs").mkdir(exist_ok=True)
    for name in _HOME_FILES:
        src = vibe_home / name
        if name == "config.toml" and config is not None:
            atomic_write_toml(agent_home / name, config)
            os.chmod(agent_home / name, 0o644)
        elif src.is_file():
            _copy_atomic(src, agent_home / name)
    trusted = agent_home / "trusted_folders.toml"
    if not trusted.exists():
        # vibe creates it on start otherwise, which a read-only home refuses.
        trusted.write_text("trusted = []\nuntrusted = []\n")
    for sub in _HOME_DIRS:
        src_dir, dst_dir = vibe_home / sub, agent_home / sub
        dst_dir.mkdir(exist_ok=True)
        wanted: set[str] = set()
        if src_dir.is_dir():
            wanted = {p.name for p in src_dir.iterdir() if p.is_file()}
        for name in sorted(wanted):
            _copy_atomic(src_dir / name, dst_dir / name)
        for stale in dst_dir.iterdir():
            if stale.is_file() and stale.name not in wanted:
                stale.unlink()


# ── the container ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentContainer:
    """What the agent container is started from, read off the running core."""

    owner: str
    image: str
    network: str
    gateway_host: str
    mounts: tuple[str, ...]


def _mount_source(mount: JsonDict) -> str:
    """A ``docker inspect`` mount as the source half of ``-v``."""
    if mount.get("Type") == "volume":
        return str(mount["Name"])
    return str(mount["Source"])


def plan(core: JsonDict, workspace: str, internal: set[str]) -> AgentContainer:
    """The agent container for the core described by *core* (``docker inspect``).

    The agent gets the core's image by id (so a pulled newer tag does not change
    what runs), the core's network among *internal* (docker networks with no
    route out), and four of its mounts: the workspace, the agent home
    (read-only), the session logs, and ``stacks.d`` (read-only). The core's
    address on that network is where the egress gateway listens.
    """
    by_dest = {str(m.get("Destination")): m for m in cast("list[JsonDict]", core.get("Mounts", []))}
    missing = [
        dest
        for dest in (workspace, str(AGENT_HOME), str(VIBE_HOME / "logs"))
        if dest not in by_dest
    ]
    if missing:
        raise IsolationUnavailableError(
            f"the core container has no mount at {', '.join(missing)}; "
            "agent isolation needs the compose file's vibe-agent and vibe-logs volumes"
        )
    networks = cast(
        "dict[str, JsonDict]",
        cast("JsonDict", core.get("NetworkSettings", {})).get("Networks") or {},
    )
    shared = sorted(n for n in networks if n in internal)
    if not shared:
        raise IsolationUnavailableError(
            "the core container is on no internal network; agent isolation needs the "
            "compose file's agent network (internal: true)"
        )
    gateway_host = str(networks[shared[0]].get("IPAddress") or "")
    if not gateway_host:
        raise IsolationUnavailableError(f"the core has no address on {shared[0]}")
    mounts = [
        f"{_mount_source(by_dest[workspace])}:{workspace}",
        f"{_mount_source(by_dest[str(AGENT_HOME)])}:{AGENT_HOME}:ro",
        f"{_mount_source(by_dest[str(VIBE_HOME / 'logs')])}:{AGENT_HOME / 'logs'}",
    ]
    stacks = by_dest.get("/app/stacks.d")
    if stacks is not None:
        mounts.append(f"{_mount_source(stacks)}:/app/stacks.d:ro")
    name = str(core.get("Name", "")).lstrip("/") or str(core.get("Id", ""))[:12]
    return AgentContainer(
        owner=name,
        image=str(core["Image"]),
        network=shared[0],
        gateway_host=gateway_host,
        mounts=tuple(mounts),
    )


def forwarded_names(env: Mapping[str, str], owed: Mapping[str, str]) -> list[str]:
    """The variables of *env* the agent container receives, by name."""
    names = {k for k in env if k.startswith(_FORWARD_PREFIXES) or k in _FORWARD_NAMES or k in owed}
    return sorted(names)


def docker_argv(box: AgentContainer, names: list[str], container_name: str) -> list[str]:
    """The ``docker run`` that starts the agent (values come from the CLI's env)."""
    argv = [
        "docker",
        "run",
        "-i",
        "--rm",
        "--init",
        "--name",
        container_name,
        "--label",
        f"{OWNER_LABEL}={box.owner}",
        "--network",
        box.network,
        "--cap-drop",
        "ALL",
        "--cap-add",
        "DAC_OVERRIDE",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "4096",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/root",
        "-w",
        "/app",
    ]
    for mount in box.mounts:
        argv += ["-v", mount]
    argv += ["-e", f"VIBE_HOME={AGENT_HOME}", "-e", f"MEDMCP_PATHGUARD_LEDGER={_PATHGUARD_LEDGER}"]
    for name in names:
        if name not in {"VIBE_HOME", "MEDMCP_PATHGUARD_LEDGER"}:
            argv += ["-e", name]
    argv += ["--entrypoint", "medmcp-agent", box.image]
    return argv


def _docker(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def inspect_self() -> JsonDict:
    """``docker inspect`` of the container this process runs in."""
    ref = os.environ.get("HOSTNAME") or socket.gethostname()
    result = _docker("inspect", ref)
    if result.returncode != 0:
        raise IsolationUnavailableError(f"cannot inspect this container: {result.stderr.strip()}")
    import json

    items = cast("list[JsonDict]", json.loads(result.stdout))
    if not items:
        raise IsolationUnavailableError("docker inspect returned nothing for this container")
    return items[0]


def internal_networks(names: list[str]) -> set[str]:
    """Which of the docker networks *names* are internal (no route out)."""
    if not names:
        return set()
    result = _docker("network", "inspect", *names)
    if result.returncode != 0:
        raise IsolationUnavailableError(f"cannot inspect networks: {result.stderr.strip()}")
    import json

    return {
        str(n["Name"])
        for n in cast("list[JsonDict]", json.loads(result.stdout))
        if n.get("Internal")
    }


def remove_containers(owner: str) -> int:
    """Remove every agent container *owner* started; return how many there were."""
    try:
        listed = _docker("ps", "-aq", "--filter", f"label={OWNER_LABEL}={owner}")
    except (OSError, subprocess.TimeoutExpired):
        return 0
    ids = listed.stdout.split()
    if ids:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            _docker("rm", "-f", *ids)
    return len(ids)


class ContainerLauncher:
    """Starts vibe-acp in an agent container (see the module docstring).

    Implements :class:`medmcp.acp.AgentLauncher`. The plan is read off the running
    core once, on first use; a core that cannot be inspected or lacks the volumes
    refuses to start the agent rather than fall back to running it unisolated.
    """

    def __init__(self, workspace: str) -> None:
        """Isolate the agent for *workspace* (the host path the core uses)."""
        self._workspace = workspace
        self._box: AgentContainer | None = None
        self._current: str | None = None

    def prepare(self) -> AgentContainer:
        """Read the plan off the running core (once) and clear leftover agents."""
        if self._box is None:
            core = inspect_self()
            networks = list(
                cast(
                    "JsonDict",
                    cast("JsonDict", core.get("NetworkSettings", {})).get("Networks") or {},
                )
            )
            self._box = plan(core, self._workspace, internal_networks(networks))
            removed = remove_containers(self._box.owner)
            if removed:
                log.info("removed %d leftover agent container(s)", removed)
        return self._box

    def argv(
        self, env: Mapping[str, str], owed: Mapping[str, str]
    ) -> tuple[list[str], dict[str, str]]:
        """The command that starts the agent, and the environment for it."""
        box = self.prepare()
        self._current = f"{box.owner}-agent-{secrets.token_hex(3)}"
        argv = docker_argv(box, forwarded_names(env, owed), self._current)
        # The docker CLI itself needs the core's environment (DOCKER_HOST, its
        # config); -e NAME then copies each forwarded value from it.
        return argv, {**os.environ, **env}

    async def cleanup(self) -> None:
        """Make sure the container is gone once its docker CLI has exited."""
        name, self._current = self._current, None
        if name is not None:
            import asyncio

            await asyncio.to_thread(_docker, "rm", "-f", name)


# ── inside the agent container ───────────────────────────────────────────────


def main() -> None:
    """``medmcp-agent``: the agent container's entry point.

    Becomes vibe-acp, whose stdio is the ``docker run -i`` the core holds. The
    model is reached through the core's egress gateway, which forwards to the
    core's tool-call shim, so nothing else runs here.
    """
    os.execvp("vibe-acp", ["vibe-acp"])
