"""Release notifications and the in-place update of a containerized install.

Two halves, both deliberately small:

**The check.** One anonymous ``GET`` of the newest release (GitHub's
``releases/latest``, or a mirror serving the same JSON shape via
``MEDMCP_UPDATE_URL``), at most once a day, never on the boot path. The
result lands in ``.vibe/state/update.json`` — on the ``vibe-state`` volume, so
"already told about v0.3.0" survives the very recreate the update performs.
``MEDMCP_UPDATE_CHECK=0`` turns the check off for air-gapped or privacy-minded
deployments; nothing but the request itself ever leaves the machine.

**The plan.** A container cannot recreate itself, so applying an update means
pulling the new core image and starting a one-shot *helper* from it
(:mod:`medmcp.updater`) that runs ``docker compose up -d`` for the project the
operator installed with. Everything the helper needs is derived here from the
running container's own ``docker inspect`` record — the compose project, the
socket, the state volume, and the values the operator set at install time —
so no host file is read and nothing is guessed. What cannot be derived keeps
the compose default, and an install this module cannot account for (a source
build, hand-managed compose files) is refused with the host command to run
instead, never half-applied.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import socket
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx

from medmcp.acp import JsonDict
from medmcp.settings import VIBE_STATE_DIR, atomic_write_json

log: logging.Logger = logging.getLogger("medmcp.update")

DEFAULT_UPDATE_URL: str = "https://api.github.com/repos/medmcp/medmcp/releases/latest"
IMAGE_REPO: str = "ghcr.io/medmcp/core"
UPDATER_CONTAINER: str = "medmcp-updater"
UPDATE_STATE_PATH: Path = VIBE_STATE_DIR / "update.json"
UPDATE_RESULT_PATH: Path = VIBE_STATE_DIR / "update_result.json"
# The compose file this image was released with, baked in by the Dockerfile.
# The helper runs compose against its own copy; this one is read for the
# defaults the operator did not override (see ``compose_defaults``).
BAKED_COMPOSE_PATH: Path = Path("/app/docker-compose.ghcr.yml")

CHECK_INTERVAL_SEC: float = 24 * 3600
FIRST_CHECK_DELAY_SEC: float = 60.0
MANUAL_CHECK_MIN_GAP_SEC: float = 60.0
FETCH_TIMEOUT_SEC: float = 10.0

_COMPOSE_LABEL_PREFIX = "com.docker.compose."
_SOCKET_TARGET = "/var/run/docker.sock"
_DOCKER_CONFIG_TARGET = "/run/host-docker-config.json"
_STATE_TARGET = "/app/.vibe/state"
_OLLAMA_MODELS_TARGET = "/root/.ollama"
# The variables the published compose file interpolates into the core service
# (plus the ones shared with the llm service). Read back from the container's
# environment; a value equal to the compose default is not carried, so a
# release that changes a default is not pinned to the old one by the update.
_CORE_ENV_VARS: tuple[str, ...] = (
    "MEDMCP_WORKSPACE",
    "MEDMCP_GPU",
    "OLLAMA_MODEL",
    "MEDMCP_CATALOG_URL",
    "MEDMCP_STACK_POOL",
    "MEDMCP_LLM_SHIM",
    "MEDMCP_SHIM_RELAY",
    "GHCR_USER",
    "GHCR_TOKEN",
    "MEDMCP_UPDATE_CHECK",
    "MEDMCP_UPDATE_URL",
)
_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")
# What may follow ``X.Y.Z-`` in a tag: the tag is used as an image tag verbatim.
_PRERELEASE_RE = re.compile(r"[A-Za-z0-9.]*")


# ── Versions ─────────────────────────────────────────────────────────────────


def parse_version(text: str) -> tuple[tuple[int, ...], str] | None:
    """``"v0.3.0-rc1"`` → ``((0, 3, 0), "rc1")``; ``None`` for anything else.

    Only the release shape this project tags is understood. ``0+unknown``
    (a checkout without metadata) and arbitrary strings parse to ``None`` and
    therefore never compare as older than a release.
    """
    body = text.strip()
    body = body.removeprefix("v")
    core, _, pre = body.partition("-")
    parts = core.split(".")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    if not _PRERELEASE_RE.fullmatch(pre):
        return None
    return tuple(int(p) for p in parts), pre


def is_newer(candidate: str, current: str) -> bool:
    """Whether *candidate* is a strictly newer release than *current*.

    A pre-release counts as older than its base version, so an install on
    ``0.3.0`` is never offered ``0.3.0-rc2``; ``releases/latest`` excludes
    pre-releases anyway, this is the belt to that suspender.
    """
    a = parse_version(candidate)
    b = parse_version(current)
    if a is None or b is None:
        return False
    if a[0] != b[0]:
        return a[0] > b[0]
    # Same numbers: only "release beats pre-release" is an ordering we assert.
    return bool(b[1]) and not a[1]


# ── The check ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReleaseInfo:
    """The newest release as the check saw it."""

    version: str
    tag: str
    notes: str
    url: str
    published_at: str

    def as_dict(self) -> JsonDict:
        """Wire/state form."""
        return asdict(self)


def parse_release(data: object) -> ReleaseInfo | None:
    """Read a GitHub release object (or a mirror's copy of one); ``None`` if unusable."""
    if not isinstance(data, dict):
        return None
    obj = cast("JsonDict", data)
    tag = str(obj.get("tag_name") or "").strip()
    if parse_version(tag) is None:
        return None
    url = str(obj.get("html_url") or "")
    return ReleaseInfo(
        version=tag.removeprefix("v"),
        tag=tag,
        notes=str(obj.get("body") or ""),
        # The page link is the one thing from the document the browser is
        # pointed at; anything but https is dropped rather than rendered.
        url=url if url.startswith("https://") else "",
        published_at=str(obj.get("published_at") or ""),
    )


def check_enabled() -> bool:
    """Whether release checks are permitted at all.

    ``MEDMCP_UPDATE_CHECK=0`` (or false/no/off) is the deployment's say — set
    in the compose environment, it forbids every check, manual ones included,
    and the UI shows the switch as pinned. The operator's own choice is the
    ``auto_check`` state key (see :func:`auto_check_enabled`).
    """
    return os.environ.get("MEDMCP_UPDATE_CHECK", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def auto_check_enabled(state: JsonDict | None = None) -> bool:
    """Whether the daily unattended check runs (permitted, and not switched off in Settings)."""
    if not check_enabled():
        return False
    state = state if state is not None else load_state()
    return bool(state.get("auto_check", True))


def set_auto_check(enabled: bool) -> JsonDict:
    """Switch the daily check on or off from the UI; manual checks stay available."""
    state = load_state()
    state["auto_check"] = bool(enabled)
    save_state(state)
    return state


def update_url() -> str:
    """Where the newest release is read from (``MEDMCP_UPDATE_URL`` for a mirror)."""
    return os.environ.get("MEDMCP_UPDATE_URL", "").strip() or DEFAULT_UPDATE_URL


async def fetch_latest_release(
    url: str | None = None, *, client: httpx.AsyncClient | None = None
) -> ReleaseInfo:
    """Fetch and parse the newest release; raises on any failure.

    A ``file:`` URL or a bare path reads a local file, which is how an
    air-gapped mirror (or a test) supplies the release document.
    """
    target = url or update_url()
    if target.startswith("file:") or target.startswith("/"):
        path = Path(target.removeprefix("file://").removeprefix("file:"))
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        headers = {
            "Accept": "application/vnd.github+json",
            # GitHub requires a User-Agent; it names the product, not the install.
            # Nothing else about the machine, the workspace, or its use is sent.
            "User-Agent": "medmcp",
        }
        if client is None:
            async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SEC) as own:
                response = await own.get(target, headers=headers)
        else:
            response = await client.get(target, headers=headers)
        response.raise_for_status()
        payload = response.json()
    info = parse_release(payload)
    if info is None:
        raise ValueError(f"no usable release in the document at {target}")
    return info


def _default_state() -> JsonDict:
    return {
        "checked_at": None,
        "latest": None,
        "error": None,
        "dismissed": None,
        "last_result": None,
        "simulate": None,
        "auto_check": True,
    }


def load_state() -> JsonDict:
    """The persisted check state (defaults when absent or unreadable)."""
    state = _default_state()
    if not UPDATE_STATE_PATH.exists():
        return state
    try:
        data = cast("JsonDict", json.loads(UPDATE_STATE_PATH.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read %s: %s", UPDATE_STATE_PATH, exc)
        return state
    for key in state:
        if key in data:
            state[key] = data[key]
    return state


def save_state(state: JsonDict) -> None:
    """Persist the check state atomically."""
    atomic_write_json(UPDATE_STATE_PATH, state)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _age_seconds(stamp: object) -> float | None:
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (datetime.now(UTC) - then).total_seconds()


def check_due(state: JsonDict, *, interval: float = CHECK_INTERVAL_SEC) -> bool:
    """Whether the periodic check should run now (never checked, or older than *interval*)."""
    age = _age_seconds(state.get("checked_at"))
    return age is None or age >= interval


async def run_check(*, force: bool = False) -> JsonDict:
    """Refresh the release record and return the new state.

    Disabled checks return the stored state untouched. A failed fetch keeps the
    last known release and records the error, so a transient outage never turns
    a known update into "up to date". Manual checks are rate-limited so the
    button cannot hammer the registry.
    """
    state = load_state()
    if not check_enabled():
        return state
    if force:
        age = _age_seconds(state.get("checked_at"))
        if age is not None and age < MANUAL_CHECK_MIN_GAP_SEC:
            return state
    elif not auto_check_enabled(state):
        return state
    try:
        info = await fetch_latest_release()
    except Exception as exc:  # network, HTTP, JSON — all are "could not check"
        log.info("release check failed: %s", exc)
        state["error"] = str(exc)
    else:
        state["latest"] = info.as_dict()
        state["error"] = None
        log.info("release check: latest is %s", info.tag)
    state["checked_at"] = _now_iso()
    save_state(state)
    return state


def dismiss(version: str) -> JsonDict:
    """Hide the header notice for *version* (Settings still shows it)."""
    state = load_state()
    state["dismissed"] = version
    save_state(state)
    return state


def ack_result() -> JsonDict:
    """Forget the outcome of the last update once the UI has shown it."""
    state = load_state()
    state["last_result"] = None
    save_state(state)
    return state


def collect_updater_result() -> JsonDict | None:
    """Fold the helper's ``update_result.json`` into the state; return it if any.

    Called on boot: the helper runs after the core it was started from has
    died, so the outcome can only be read by the *next* core. The helper
    container itself is removed once its record is in, so ``docker ps -a``
    does not accumulate one per update.
    """
    if not UPDATE_RESULT_PATH.exists():
        return None
    try:
        result = cast("JsonDict", json.loads(UPDATE_RESULT_PATH.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read %s: %s", UPDATE_RESULT_PATH, exc)
        return None
    state = load_state()
    state["last_result"] = result
    save_state(state)
    with contextlib.suppress(OSError):
        UPDATE_RESULT_PATH.unlink()
    remove_stale_updater()
    return result


# ── Rehearsal ────────────────────────────────────────────────────────────────
# Writing ``{"simulate": "<version>"}`` into the state file (alongside whatever
# else is there) makes that version show up as a release and lets the whole
# flow be walked through — notice, notes, confirmation, download, restart,
# outcome — without docker: the download is scripted and the "restart" flips
# the version this process reports. Nothing is pulled, started, or replaced.
# The key clears itself when the rehearsal completes, and a real restart of the
# server ends the pretence. For demos and for checking the UI on an install
# that cannot be updated from here.

_simulated_version: str | None = None


def rehearsal_target(state: JsonDict, current_version: str) -> str | None:
    """The version a rehearsal is set up for, if one is and it would count as an update."""
    target = state.get("simulate")
    if isinstance(target, str) and target and is_newer(target, current_version):
        return target.removeprefix("v")
    return None


def rehearsal_release(version: str) -> JsonDict:
    """The stand-in release record a rehearsal presents."""
    return {
        "version": version,
        "tag": f"v{version}",
        "notes": (
            "### Rehearsal\n\n"
            "This release does not exist. Walking through the update from here "
            "downloads nothing and changes nothing; the restart is pretended and the "
            "version shown afterwards is not real until the server restarts.\n\n"
            "### Added\n\n- The workspace says when a new MedMCP release is out and "
            "shows its notes.\n- Update from the UI, with a rollback if the new "
            "release does not start."
        ),
        "url": "",
        "published_at": _now_iso(),
    }


def effective_version(current_version: str) -> str:
    """What this process reports as its version: the real one, or the rehearsed one."""
    return _simulated_version or current_version


_REHEARSAL_LINES: tuple[str, ...] = (
    "{tag}: Pulling from medmcp/core",
    "4f4fb700ef54: Pulling fs layer",
    "9c1b6dd6c1e6: Pulling fs layer",
    "4f4fb700ef54: Downloading [=========>          ]  112MB/512MB",
    "4f4fb700ef54: Downloading [==================> ]  498MB/512MB",
    "4f4fb700ef54: Pull complete",
    "9c1b6dd6c1e6: Pull complete",
    "Digest: sha256:0000000000000000000000000000000000000000000000000000000000000000",
    "Status: Downloaded newer image for ghcr.io/medmcp/core:{tag}",
)


async def run_rehearsal(
    version: str,
    on_progress: Callable[[str], None],
    *,
    previous: str,
    step_delay: float = 0.5,
    restart_delay: float = 6.0,
) -> None:
    """Play the download, record the outcome, and pretend the restart.

    The version flip is delayed so the "restarting" stage is visible before the
    page finds the new version on ``/healthz`` and reloads.
    """
    tag = f"v{version}"
    for line in _REHEARSAL_LINES:
        on_progress(line.format(tag=tag))
        await asyncio.sleep(step_delay)
    state = load_state()
    state["simulate"] = None
    state["last_result"] = {
        "from": previous,
        "to": tag,
        "status": "ok",
        "detail": "rehearsal — nothing was installed",
        "at": _now_iso(),
    }
    save_state(state)

    def flip() -> None:
        global _simulated_version
        _simulated_version = version

    if restart_delay > 0:
        asyncio.get_running_loop().call_later(restart_delay, flip)
    else:
        flip()


def updater_pending() -> bool:
    """Whether an update helper is (or was) at work whose outcome has not been read."""
    if UPDATE_RESULT_PATH.exists():
        return True
    return inspect_ref(UPDATER_CONTAINER) is not None


# ── The plan ─────────────────────────────────────────────────────────────────


class UpdateNotApplicableError(Exception):
    """This install cannot be updated from the UI; ``str()`` says why.

    *command*, when given, is the host command that updates this particular
    install (its own compose files), shown in place of the generic one.
    """

    def __init__(self, reason: str, *, command: str | None = None) -> None:
        """Record the reason and, optionally, the install-specific host command."""
        super().__init__(reason)
        self.command = command


@dataclass
class ApplyPlan:
    """Everything the helper needs, derived from the running container."""

    container_id: str
    image_repo: str
    current_tag: str
    project: str
    project_dir: str
    socket_source: str
    state_volume: str
    env: dict[str, str]
    docker_config_source: str | None = None
    # The registry digest the running image was pulled under, when known: a
    # rollback then goes back to exactly this image rather than to a tag.
    current_digest: str | None = None

    def image(self, tag: str) -> str:
        """The image reference for release *tag*."""
        return f"{self.image_repo}:{tag}"


def _run(args: list[str], *, timeout: float = 60.0) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def inspect_ref(ref: str) -> JsonDict | None:
    """``docker inspect`` one container; ``None`` if absent or unreadable."""
    try:
        out = _run(["inspect", ref], timeout=30)
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return None
    try:
        items = cast("list[JsonDict]", json.loads(out))
    except json.JSONDecodeError:
        return None
    return items[0] if items else None


def self_container_id() -> str:
    """The id docker gave this container (its default hostname)."""
    return os.environ.get("HOSTNAME") or socket.gethostname()


def find_project_container(project: str, service: str) -> JsonDict | None:
    """The container compose runs for *service* in *project*, if any."""
    try:
        out = _run(
            [
                "ps",
                "-a",
                "-q",
                "--filter",
                f"label={_COMPOSE_LABEL_PREFIX}project={project}",
                "--filter",
                f"label={_COMPOSE_LABEL_PREFIX}service={service}",
            ],
            timeout=30,
        )
    except (RuntimeError, OSError, subprocess.TimeoutExpired):
        return None
    ids = out.split()
    return inspect_ref(ids[0]) if ids else None


def compose_defaults(text: str) -> dict[str, str]:
    """``${VAR:-default}`` occurrences in a compose file → ``{VAR: default}``."""
    return {m.group(1): m.group(2) for m in _DEFAULT_RE.finditer(text)}


def _installed_by_one_liner(config_files: str) -> bool:
    """Whether the compose files on record are ones the workspace may rewrite.

    Two spellings: the ``oci://`` one-liner (compose caches the artifact under
    ``~/.cache/docker-compose/``) and the copy baked into the core image, which
    is what a previous UI update left on the labels. Anything else is a file the
    operator maintains by hand, and the update stays out of it.
    """
    files = [f for f in config_files.split(",") if f]
    if not files:
        return False
    return all("/.cache/docker-compose/" in f or f == str(BAKED_COMPOSE_PATH) for f in files)


def _labels(info: JsonDict) -> dict[str, str]:
    config = cast("JsonDict", info.get("Config") or {})
    return cast("dict[str, str]", config.get("Labels") or {})


def _env(info: JsonDict) -> dict[str, str]:
    config = cast("JsonDict", info.get("Config") or {})
    out: dict[str, str] = {}
    for item in cast("list[str]", config.get("Env") or []):
        key, sep, value = item.partition("=")
        if sep:
            out[key] = value
    return out


def _mounts(info: JsonDict) -> list[JsonDict]:
    return cast("list[JsonDict]", info.get("Mounts") or [])


def _mount_at(info: JsonDict, destination: str) -> JsonDict | None:
    for mount in _mounts(info):
        if mount.get("Destination") == destination:
            return mount
    return None


def plan_apply(
    self_info: JsonDict,
    llm_info: JsonDict | None,
    *,
    defaults: dict[str, str] | None = None,
) -> ApplyPlan:
    """Derive the helper's inputs from the running core's inspect record.

    Raises :class:`UpdateNotApplicableError` with the operator-facing reason for any
    install the helper is not designed to rewrite.
    """
    defaults = defaults or {}
    config = cast("JsonDict", self_info.get("Config") or {})
    image = str(config.get("Image") or "")
    repo, _, tag = image.rpartition(":")
    if repo != IMAGE_REPO or not tag:
        raise UpdateNotApplicableError(
            "This MedMCP runs from a locally built image, not a published release. "
            "Update it from the source checkout."
        )

    labels = _labels(self_info)
    project = labels.get(f"{_COMPOSE_LABEL_PREFIX}project", "")
    config_files = labels.get(f"{_COMPOSE_LABEL_PREFIX}project.config_files", "")
    project_dir = labels.get(f"{_COMPOSE_LABEL_PREFIX}project.working_dir", "")
    if not project:
        raise UpdateNotApplicableError(
            "This container is not managed by docker compose. Update it the way it was started."
        )
    if not _installed_by_one_liner(config_files) or not project_dir:
        files = config_files or "<your compose files>"
        cmd = f"docker compose -f {files.replace(',', ' -f ')} up -d --pull always"
        raise UpdateNotApplicableError(
            "This install uses local compose files, which the workspace does not rewrite. "
            "Update it from the host with its own files.",
            command=cmd,
        )

    sock = _mount_at(self_info, _SOCKET_TARGET)
    if sock is None or not str(sock.get("Source") or ""):
        raise UpdateNotApplicableError("The docker socket is not mounted; the update needs it.")
    state = _mount_at(self_info, _STATE_TARGET)
    if state is None or state.get("Type") != "volume" or not str(state.get("Name") or ""):
        raise UpdateNotApplicableError(
            f"{_STATE_TARGET} is not on a named volume; the update cannot report its outcome."
        )
    docker_config = _mount_at(self_info, _DOCKER_CONFIG_TARGET)

    env: dict[str, str] = {}
    have = _env(self_info)
    for key in _CORE_ENV_VARS:
        # An empty value never overrides a ``${VAR:-default}`` in compose, so it
        # is the same as unset; carrying it would only pin a blank.
        if have.get(key) and have[key] != defaults.get(key, ""):
            env[key] = have[key]
    if "MEDMCP_WORKSPACE" not in env:
        raise UpdateNotApplicableError("MEDMCP_WORKSPACE is not set on this container.")
    socket_source = str(sock["Source"])
    env["XDG_RUNTIME_DIR"] = os.path.dirname(socket_source)
    docker_config_source = str(docker_config.get("Source") or "") if docker_config else None
    if docker_config_source:
        home, sep, _ = docker_config_source.rpartition("/.docker/")
        if sep:
            env["HOME"] = home

    if llm_info is not None:
        llm_image = str(cast("JsonDict", llm_info.get("Config") or {}).get("Image") or "")
        if llm_image and llm_image != defaults.get("OLLAMA_IMAGE"):
            env["OLLAMA_IMAGE"] = llm_image
        keep = _env(llm_info).get("OLLAMA_KEEP_ALIVE")
        if keep is not None and keep != defaults.get("OLLAMA_KEEP_ALIVE"):
            env["OLLAMA_KEEP_ALIVE"] = keep
        models = _mount_at(llm_info, _OLLAMA_MODELS_TARGET)
        if models is not None and models.get("Type") == "bind":
            env["OLLAMA_MODELS_DIR"] = str(models.get("Source") or "")

    current_digest: str | None = None
    for entry in cast(
        "list[str]", cast("JsonDict", self_info.get("Image_RepoDigests") or {}) or []
    ):
        entry_repo, _, entry_digest = entry.partition("@")
        if entry_repo == repo and entry_digest.startswith("sha256:"):
            current_digest = entry_digest
            break
    return ApplyPlan(
        current_digest=current_digest,
        container_id=str(self_info.get("Id") or self_container_id()),
        image_repo=repo,
        current_tag=tag,
        project=project,
        project_dir=project_dir,
        socket_source=socket_source,
        state_volume=str(state["Name"]),
        docker_config_source=docker_config_source or None,
        env=env,
    )


def current_plan() -> ApplyPlan:
    """Inspect this container (and the llm service) and plan the update."""
    self_info = inspect_ref(self_container_id())
    if self_info is None:
        raise UpdateNotApplicableError(
            "Not running in a container managed through the docker socket. "
            "Update this install the way it was started."
        )
    defaults: dict[str, str] = {}
    with contextlib.suppress(OSError):
        defaults = compose_defaults(BAKED_COMPOSE_PATH.read_text(encoding="utf-8"))
    project = _labels(self_info).get(f"{_COMPOSE_LABEL_PREFIX}project", "")
    llm_info = find_project_container(project, "llm") if project else None
    image = str(cast("JsonDict", self_info.get("Config") or {}).get("Image") or "")
    image_info = inspect_ref(image) if image else None
    if image_info is not None:
        self_info["Image_RepoDigests"] = image_info.get("RepoDigests") or []
    return plan_apply(self_info, llm_info, defaults=defaults)


def updater_command(
    plan: ApplyPlan, target_tag: str, target_digest: str | None = None
) -> list[str]:
    """The ``docker run`` that starts the helper from the *new* image.

    The helper is the new release's ``medmcp-update apply``; this command line
    is the whole contract between the two versions, so it carries values, not
    decisions: which container to replace, which project, and the compose
    variables the operator set. The image entrypoint is bypassed because it
    would start a second workspace server instead.
    """
    args = [
        "run",
        "-d",
        "--name",
        UPDATER_CONTAINER,
        "--label",
        "org.medmcp.updater=1",
        "-v",
        f"{plan.socket_source}:{_SOCKET_TARGET}",
        "-v",
        f"{plan.state_volume}:{_STATE_TARGET}",
    ]
    if plan.docker_config_source:
        args += ["-v", f"{plan.docker_config_source}:{_DOCKER_CONFIG_TARGET}:ro"]
    args += [
        "-e",
        f"DOCKER_HOST=unix://{_SOCKET_TARGET}",
        "-e",
        f"MEDMCP_UPDATE_REPLACE={plan.container_id}",
        "-e",
        f"MEDMCP_UPDATE_PROJECT={plan.project}",
        "-e",
        f"MEDMCP_UPDATE_PROJECT_DIR={plan.project_dir}",
        "-e",
        f"MEDMCP_UPDATE_PREVIOUS_TAG={plan.current_tag}",
        "-e",
        f"MEDMCP_UPDATE_PREVIOUS_DIGEST={plan.current_digest or ''}",
        "-e",
        f"MEDMCP_UPDATE_DIGEST={target_digest or ''}",
        "-e",
        f"MEDMCP_UPDATE_ENV={json.dumps(plan.env, sort_keys=True)}",
        "--entrypoint",
        "medmcp-update",
        # Started by digest when one was verified: the tag could be moved
        # between verification and here, the digest cannot.
        f"{plan.image_repo}@{target_digest}" if target_digest else plan.image(target_tag),
        "apply",
    ]
    return args


def verify_pulled_image(image: str, tag: str) -> None:
    """Refuse to start the helper from an image that does not carry *tag*.

    Every released core is labelled with its tag (``org.opencontainers.image.version``).
    A registry answering ``:v0.3.0`` with something else — a mis-tagged mirror,
    a moved tag — is caught here rather than run.
    """
    try:
        out = _run(["image", "inspect", image], timeout=30)
        items = cast("list[JsonDict]", json.loads(out))
    except (RuntimeError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not inspect the pulled image {image}: {exc}") from exc
    labelled = _labels(items[0]).get("org.opencontainers.image.version", "") if items else ""
    if labelled != tag:
        raise RuntimeError(
            f"the image pulled for {tag} is labelled {labelled or 'nothing'!r}; not starting it."
        )


def remove_stale_updater() -> None:
    """Remove a finished helper container so a new one can take its name."""
    info = inspect_ref(UPDATER_CONTAINER)
    if info is None:
        return
    state = cast("JsonDict", info.get("State") or {})
    if state.get("Running"):
        return
    try:
        _run(["rm", "-f", UPDATER_CONTAINER], timeout=30)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        log.warning("could not remove %s: %s", UPDATER_CONTAINER, exc)


def start_updater(plan: ApplyPlan, target_tag: str, target_digest: str | None = None) -> str:
    """Start the helper; returns its container id. The image must already be pulled."""
    remove_stale_updater()
    args = updater_command(plan, target_tag, target_digest)
    shown = [
        f"{a.partition('=')[0]}=<redacted>" if a.startswith("MEDMCP_UPDATE_ENV=") else a
        for a in args
    ]
    log.info("starting updater: docker %s", shlex.join(shown))
    return _run(args, timeout=120).strip()


# ── Status for the API ───────────────────────────────────────────────────────


def host_commands(plan_or_tag: ApplyPlan | str, target_tag: str) -> JsonDict:
    """The commands to run on the host: the update itself and the way back."""
    previous = plan_or_tag.current_tag if isinstance(plan_or_tag, ApplyPlan) else plan_or_tag
    base = "docker compose -f oci://ghcr.io/medmcp/compose:{tag} up -d --pull always"
    return {"update": base.format(tag=target_tag), "rollback": base.format(tag=previous)}


def status(
    *,
    current_version: str,
    build: str,
    plan_error: PlanError | None,
    state: JsonDict | None = None,
) -> JsonDict:
    """The ``GET /api/update`` payload."""
    state = state if state is not None else load_state()
    current_version = effective_version(current_version)
    latest = cast("JsonDict | None", state.get("latest"))
    rehearsal = rehearsal_target(state, current_version)
    if rehearsal:
        latest = rehearsal_release(rehearsal)
        plan_error = None
    available = bool(latest and is_newer(str(latest.get("version", "")), current_version))
    target_tag = str(latest.get("tag", "")) if latest else ""
    previous = build if parse_version(build) is not None else "latest"
    commands = host_commands(previous, target_tag) if available else None
    if commands and plan_error and plan_error.command:
        commands["update"] = plan_error.command
    return {
        "current": {"version": current_version, "build": build},
        "enabled": check_enabled(),
        "auto_check": auto_check_enabled(state),
        "checked_at": state.get("checked_at"),
        "error": state.get("error"),
        "latest": latest,
        "available": available,
        "dismissed": state.get("dismissed") == (latest or {}).get("version"),
        "can_apply": available and plan_error is None,
        "apply_reason": plan_error.reason if plan_error else None,
        "host_commands": commands,
        "last_result": state.get("last_result"),
        "rehearsal": bool(rehearsal),
    }


@dataclass(frozen=True)
class PlanError:
    """Why the UI cannot apply an update here, and the host command if there is a specific one."""

    reason: str
    command: str | None = None


def plan_error() -> PlanError | None:
    """Why the UI cannot apply an update here, or ``None`` if it can."""
    try:
        current_plan()
    except UpdateNotApplicableError as exc:
        return PlanError(str(exc), exc.command)
    except Exception as exc:  # docker CLI trouble: report, never raise into the UI
        log.warning("update plan failed: %s", exc)
        return PlanError(f"Could not inspect this container: {exc}")
    return None


__all__ = [
    "ApplyPlan",
    "PlanError",
    "ReleaseInfo",
    "UpdateNotApplicableError",
    "ack_result",
    "auto_check_enabled",
    "check_due",
    "check_enabled",
    "collect_updater_result",
    "compose_defaults",
    "current_plan",
    "dismiss",
    "effective_version",
    "fetch_latest_release",
    "host_commands",
    "is_newer",
    "load_state",
    "parse_release",
    "parse_version",
    "plan_apply",
    "plan_error",
    "rehearsal_release",
    "rehearsal_target",
    "run_check",
    "run_rehearsal",
    "save_state",
    "set_auto_check",
    "start_updater",
    "status",
    "updater_command",
    "updater_pending",
    "verify_pulled_image",
]
