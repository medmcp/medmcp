"""``medmcp-docker-guard``: the core's Docker socket, filtered.

The core starts containers — stacks, the isolated agent, the update helper — and
for that it held the Docker socket itself, which is the host user (or host root,
on a rootful daemon): a bug anywhere in the core would have been a way onto the
host. This guard runs as its own small container holding the real socket and
hands the core a filtered one. It forwards only what the core does:

- reads: ping, version, info, image and network inspection, container lists;
  a container's own details only for containers of this compose project or that
  the guard created;
- pulls by name (``fromImage``) with no body, and registry logins — never an
  image import, which would let the caller name its own image anything;
- ``POST /containers/create`` after checking the body: no ``--privileged``, no
  host or container namespaces, no capabilities beyond ``DAC_OVERRIDE``, no
  security option beyond ``no-new-privileges``, no devices but NVIDIA GPUs, and
  bind mounts only of the workspace and the core's own volumes. The real socket
  may be mounted only into the update helper, and only in exactly the shape
  :func:`medmcp.update.updater_command` creates it: the core image by a digest
  the daemon recorded from a registry pull, ``medmcp-update apply``, its fixed
  mounts, and an environment whose every value is what this install already
  runs with. Every container created gets a label marking it as the guard's;
- start, attach, wait, resize, kill, stop, remove and copy-out only for
  containers carrying that label.

Everything else — exec, build, image push or import, swarm, plugins, volume and
network creation, container update — is refused. One request per connection:
the head and its declared body are forwarded with ``Connection: close`` and only
the response is relayed back. The one exception is ``attach`` on a container the
guard created, whose upgrade is honoured, and piped both ways, only once the
daemon has answered ``101``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from medmcp.update import CORE_ENV_VARS

log: logging.Logger = logging.getLogger("medmcp.dockerguard")

JsonDict = dict[str, Any]

GUARD_LABEL = "org.medmcp.guard"
_HEAD_LIMIT = 64 * 1024
_BODY_LIMIT = 4 * 1024 * 1024
_VERSION_PREFIX = re.compile(r"^/v\d+(\.\d+)?(?=/)")
_SAFE_SECURITY_OPTS = frozenset(
    {"no-new-privileges", "no-new-privileges:true", "no-new-privileges=true"}
)
_SAFE_CAPS = frozenset({"DAC_OVERRIDE", "CAP_DAC_OVERRIDE"})
_SAFE_RUNTIMES = frozenset({"", "runc", "nvidia"})
_SAFE_NETWORK_MODES = frozenset({"", "default", "bridge", "none"})
_NAMESPACE_KEYS = ("PidMode", "IpcMode", "UTSMode", "UsernsMode", "CgroupnsMode")
_MUST_BE_EMPTY = ("Devices", "DeviceCgroupRules", "VolumesFrom", "Sysctls", "CgroupParent")
UPDATER_ENTRYPOINT = "medmcp-update"
_SOCKET_TARGET = "/var/run/docker.sock"
_STATE_TARGET = "/app/.vibe/state"
_DOCKER_CONFIG_TARGET = "/run/host-docker-config.json"
_OLLAMA_MODELS_TARGET = "/root/.ollama"
_UPDATER_ENV_KEYS = frozenset(
    {
        "DOCKER_HOST",
        "MEDMCP_UPDATE_REPLACE",
        "MEDMCP_UPDATE_PROJECT",
        "MEDMCP_UPDATE_PROJECT_DIR",
        "MEDMCP_UPDATE_PREVIOUS_TAG",
        "MEDMCP_UPDATE_PREVIOUS_DIGEST",
        "MEDMCP_UPDATE_DIGEST",
        "MEDMCP_UPDATE_ENV",
    }
)
_PULL_QUERY_KEYS = frozenset({"fromImage", "tag", "platform"})

# (method, path pattern) → rule. Container ids and image names may contain "/".
_READS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (m, re.compile(p))
    for m, p in (
        ("GET", r"/_ping"),
        ("HEAD", r"/_ping"),
        ("GET", r"/version"),
        ("GET", r"/info"),
        ("GET", r"/containers/json"),
        ("GET", r"/images/json"),
        ("GET", r"/images/.+/json"),
        ("GET", r"/distribution/.+/json"),
        ("GET", r"/networks"),
        ("GET", r"/networks/[^/]+"),
        # `docker inspect <name>` falls through containers, images, networks and
        # volumes in turn; reading a volume's record is harmless.
        ("GET", r"/volumes"),
        ("GET", r"/volumes/[^/]+"),
        ("GET", r"/plugins/[^/]+/json"),
        ("POST", r"/auth"),
    )
)
_CONTAINER_OP = re.compile(r"/containers/([^/]+)(/[a-z]+)?")
_LABELLED_OPS: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/start"),
        ("POST", "/attach"),
        ("POST", "/wait"),
        ("POST", "/resize"),
        ("POST", "/kill"),
        ("POST", "/stop"),
        ("DELETE", ""),
        ("GET", "/archive"),
        ("HEAD", "/archive"),
        ("GET", "/logs"),
    }
)


@dataclass
class GuardContext:
    """What the checks compare against, read off the daemon."""

    project: str
    workspace: str
    socket_source: str
    core_image_repo: str
    mount_sources: set[str] = field(default_factory=set[str])
    networks: set[str] = field(default_factory=set[str])
    # What the update helper must match (see check_updater).
    core_id: str = ""
    core_env: dict[str, str] = field(default_factory=dict[str, str])
    core_digests: set[str] = field(default_factory=set[str])
    project_dir: str = ""
    state_volume: str = ""
    docker_config_source: str = ""
    llm_image: str = ""
    llm_env: dict[str, str] = field(default_factory=dict[str, str])
    llm_models_source: str = ""


@dataclass
class Decision:
    """Forward (with *body* replacing the request's when set), or refuse with *reason*."""

    allowed: bool
    reason: str = ""
    body: bytes | None = None
    upgrade: bool = False


def _repo(image: str) -> str:
    """``ghcr.io/medmcp/core:v1@sha256:…`` → ``ghcr.io/medmcp/core``."""
    name = image.split("@", 1)[0]
    head, _, tail = name.rpartition("/")
    return f"{head}/{tail.split(':', 1)[0]}" if head else tail.split(":", 1)[0]


def _bind_source(bind: str) -> str:
    return bind.split(":", 1)[0]


def check_create(body: JsonDict, ctx: GuardContext, repo_digests: list[str]) -> str | None:
    """Why a ``POST /containers/create`` body must be refused, or ``None``.

    *repo_digests* are the ``repo@digest`` references the daemon recorded for the
    body's image when it was pulled (none for an image that was not), which is
    how the update helper's image is recognised.
    """
    host = cast("JsonDict", body.get("HostConfig") or {})
    if host.get("Privileged"):
        return "privileged containers are not allowed"
    for key in _NAMESPACE_KEYS:
        value = str(host.get(key) or "")
        if value == "host" or value.startswith("container:"):
            return f"{key} {value!r} is not allowed"
    network = str(host.get("NetworkMode") or "")
    if network not in _SAFE_NETWORK_MODES and network not in ctx.networks:
        return f"network {network!r} is not allowed"
    caps = {str(c).upper() for c in cast("list[Any]", host.get("CapAdd") or [])}
    if not caps <= _SAFE_CAPS:
        return f"capabilities {sorted(caps - _SAFE_CAPS)} are not allowed"
    opts = {str(o) for o in cast("list[Any]", host.get("SecurityOpt") or [])}
    if not opts <= _SAFE_SECURITY_OPTS:
        return f"security options {sorted(opts - _SAFE_SECURITY_OPTS)} are not allowed"
    for key in _MUST_BE_EMPTY:
        if host.get(key):
            return f"{key} is not allowed"
    for key in ("MaskedPaths", "ReadonlyPaths"):
        if host.get(key) is not None:
            return f"{key} is not allowed"
    if str(host.get("Runtime") or "") not in _SAFE_RUNTIMES:
        return f"runtime {host.get('Runtime')!r} is not allowed"
    for request in cast("list[JsonDict]", host.get("DeviceRequests") or []):
        driver = str(request.get("Driver") or "")
        ids = [str(i) for i in cast("list[Any]", request.get("DeviceIDs") or [])]
        if driver == "cdi" and all(i.startswith("nvidia.com/gpu=") for i in ids):
            continue
        if driver == "nvidia" and not ids:
            continue
        return f"device request {request!r} is not allowed"

    sources = [_bind_source(str(b)) for b in cast("list[Any]", host.get("Binds") or [])]
    for mount in cast("list[JsonDict]", host.get("Mounts") or []):
        if str(mount.get("Type") or "") in {"bind", "volume"}:
            sources.append(str(mount.get("Source") or ""))
    if ctx.socket_source and ctx.socket_source in sources:
        return check_updater(body, ctx, repo_digests)
    allowed = {ctx.workspace, *ctx.mount_sources}
    for source in sources:
        if source not in allowed:
            return f"mount source {source!r} is not the workspace or one of the core's volumes"
    return None


def _env_map(body: JsonDict) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in cast("list[Any]", body.get("Env") or []):
        key, _, value = str(item).partition("=")
        out[key] = value
    return out


def _carried_value_ok(key: str, value: str, ctx: GuardContext) -> bool:
    """Whether a compose variable the helper would redeploy with is this install's own."""
    if key in CORE_ENV_VARS:
        return value == ctx.core_env.get(key)
    if key == "XDG_RUNTIME_DIR":
        return value == os.path.dirname(ctx.socket_source)
    if key == "HOME":
        home, sep, _ = ctx.docker_config_source.rpartition("/.docker/")
        return bool(sep) and value == home
    if key == "OLLAMA_IMAGE":
        return value == ctx.llm_image
    if key == "OLLAMA_KEEP_ALIVE":
        return value == ctx.llm_env.get(key)
    if key == "OLLAMA_MODELS_DIR":
        return value == ctx.llm_models_source
    return False


def check_updater(body: JsonDict, ctx: GuardContext, repo_digests: list[str]) -> str | None:
    """Why a create that mounts the real socket is not the update helper, or ``None``.

    The helper is the one container that holds the real socket, so it must be
    exactly what :func:`medmcp.update.updater_command` asks for — the signed core
    image (by a digest the daemon recorded from a registry pull, which an import
    cannot produce), ``medmcp-update apply``, its fixed mounts — and may carry
    only values this install already runs with, so it can redeploy the stack but
    not change what it mounts or runs.
    """
    host = cast("JsonDict", body.get("HostConfig") or {})
    image = str(body.get("Image") or "")
    digest = re.fullmatch(rf"{re.escape(ctx.core_image_repo)}@(sha256:[0-9a-f]{{64}})", image)
    if not ctx.core_image_repo or digest is None:
        return "the update helper must be the core image, by digest"
    if image not in repo_digests:
        return "the update helper's image was not pulled from the core's registry"
    if body.get("Entrypoint") != [UPDATER_ENTRYPOINT] or body.get("Cmd") != ["apply"]:
        return "the update helper must run 'medmcp-update apply'"
    if body.get("User") or body.get("WorkingDir") or host.get("Mounts"):
        return "the update helper may not set a user, working directory or extra mounts"
    expected = {f"{ctx.socket_source}:{_SOCKET_TARGET}", f"{ctx.state_volume}:{_STATE_TARGET}"}
    optional = (
        {f"{ctx.docker_config_source}:{_DOCKER_CONFIG_TARGET}:ro"}
        if ctx.docker_config_source
        else set[str]()
    )
    binds = {str(b) for b in cast("list[Any]", host.get("Binds") or [])}
    if not ctx.state_volume or not expected <= binds or not binds <= expected | optional:
        return "the update helper's mounts are not the ones it is given"
    env = _env_map(body)
    if not set(env) <= _UPDATER_ENV_KEYS:
        return f"the update helper may not set {sorted(set(env) - _UPDATER_ENV_KEYS)}"
    fixed = {
        "DOCKER_HOST": f"unix://{_SOCKET_TARGET}",
        "MEDMCP_UPDATE_REPLACE": ctx.core_id,
        "MEDMCP_UPDATE_PROJECT": ctx.project,
        "MEDMCP_UPDATE_PROJECT_DIR": ctx.project_dir,
        "MEDMCP_UPDATE_DIGEST": digest.group(1),
    }
    for key, value in fixed.items():
        if not value or env.get(key) != value:
            return f"the update helper's {key} is not this install's"
    if env.get("MEDMCP_UPDATE_PREVIOUS_DIGEST", "") not in {"", *ctx.core_digests}:
        return "the update helper may roll back only to the running core"
    if not re.fullmatch(r"[\w.-]*", env.get("MEDMCP_UPDATE_PREVIOUS_TAG", "")):
        return "the update helper's previous tag is malformed"
    try:
        loaded: object = json.loads(env.get("MEDMCP_UPDATE_ENV", ""))
    except json.JSONDecodeError:
        return "MEDMCP_UPDATE_ENV is not JSON"
    if not isinstance(loaded, dict):
        return "MEDMCP_UPDATE_ENV is not an object"
    carried = cast("JsonDict", loaded)
    if carried.get("MEDMCP_WORKSPACE") != ctx.workspace:
        return "the update helper may not change the workspace"
    for key, value in carried.items():
        if not isinstance(value, str) or not _carried_value_ok(key, value, ctx):
            return f"the update helper may not carry {key}={value!r}"
    return None


def with_guard_label(body: JsonDict) -> JsonDict:
    """*body* with the label that marks the container as the guard's."""
    labels = dict(cast("JsonDict", body.get("Labels") or {}))
    labels[GUARD_LABEL] = "1"
    return {**body, "Labels": labels}


# ── the daemon side ──────────────────────────────────────────────────────────


class Daemon:
    """Reads the guard needs from the real daemon (blocking; called off-loop)."""

    def __init__(self, socket_path: str, project: str, workspace: str) -> None:
        """Talk to the daemon at *socket_path*; guard compose project *project*."""
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path), base_url="http://docker", timeout=30
        )
        self._socket_path = socket_path
        self._project = project
        self._workspace = workspace
        self._ctx: GuardContext | None = None
        self._ctx_at = 0.0

    def _get(self, path: str) -> JsonDict | list[Any] | None:
        resp = self._client.get(path)
        return cast("JsonDict | list[Any]", resp.json()) if resp.status_code == 200 else None

    def container(self, ref: str) -> JsonDict | None:
        """A container's inspect record, or ``None``."""
        found = self._get(f"/containers/{ref}/json")
        return found if isinstance(found, dict) else None

    def repo_digests(self, ref: str) -> list[str]:
        """``repo@digest`` references the daemon recorded for an image's pulls."""
        found = self._get(f"/images/{ref}/json") if ref else None
        if not isinstance(found, dict):
            return []
        return [str(d) for d in cast("list[Any]", found.get("RepoDigests") or [])]

    def _service(self, service: str) -> JsonDict | None:
        flt = json.dumps(
            {
                "label": [
                    f"com.docker.compose.project={self._project}",
                    f"com.docker.compose.service={service}",
                ]
            }
        )
        listed = self._client.get("/containers/json", params={"filters": flt}).json()
        return self.container(str(listed[0]["Id"])) if listed else None

    def context(self) -> GuardContext:
        """The core's mounts, networks and image, re-read every few seconds."""
        if self._ctx is not None and time.monotonic() - self._ctx_at < 10:
            return self._ctx
        core = self._service("medmcp")
        llm = self._service("llm")
        ctx = GuardContext(
            project=self._project,
            workspace=self._workspace,
            socket_source=os.environ.get("MEDMCP_DOCKER_SOCKET_SOURCE", ""),
            core_image_repo="",
        )
        if core is not None:
            config = cast("JsonDict", core.get("Config") or {})
            for mount in cast("list[JsonDict]", core.get("Mounts") or []):
                dest = str(mount.get("Destination", ""))
                if dest.startswith("/run/medmcp-guard"):
                    continue  # the filtered socket stays with the core
                ctx.mount_sources.add(str(mount.get("Name") or mount.get("Source") or ""))
                if dest == _STATE_TARGET and mount.get("Type") == "volume":
                    ctx.state_volume = str(mount.get("Name") or "")
                if dest == _DOCKER_CONFIG_TARGET:
                    ctx.docker_config_source = str(mount.get("Source") or "")
            ctx.mount_sources.discard("")
            net_settings = cast("JsonDict", core.get("NetworkSettings") or {})
            ctx.networks = set(cast("JsonDict", net_settings.get("Networks") or {}))
            ctx.core_image_repo = _repo(str(config.get("Image") or ""))
            ctx.core_id = str(core.get("Id") or "")
            ctx.core_env = _env_map(config)
            labels = cast("JsonDict", config.get("Labels") or {})
            ctx.project_dir = str(labels.get("com.docker.compose.project.working_dir") or "")
            ctx.core_digests = {
                d.partition("@")[2] for d in self.repo_digests(str(core.get("Image") or ""))
            }
        if llm is not None:
            llm_config = cast("JsonDict", llm.get("Config") or {})
            ctx.llm_image = str(llm_config.get("Image") or "")
            ctx.llm_env = _env_map(llm_config)
            for mount in cast("list[JsonDict]", llm.get("Mounts") or []):
                if (
                    mount.get("Destination") == _OLLAMA_MODELS_TARGET
                    and mount.get("Type") == "bind"
                ):
                    ctx.llm_models_source = str(mount.get("Source") or "")
        self._ctx = ctx
        self._ctx_at = time.monotonic()
        return self._ctx


def check_request(method: str, target: str, body: bytes, daemon: Daemon) -> Decision:
    """Whether to forward one request (see the module docstring)."""
    parts = urlsplit(target)
    path = _VERSION_PREFIX.sub("", unquote(parts.path))
    for allowed_method, pattern in _READS:
        if method == allowed_method and pattern.fullmatch(path):
            return Decision(True)
    if method == "POST" and path == "/images/create":
        # A pull, never an import: ``fromSrc`` with a body or URL would let the
        # caller name an image of its own making after the core's repository.
        query = parse_qs(parts.query, keep_blank_values=True)
        if not query.get("fromImage") or not set(query) <= _PULL_QUERY_KEYS or body:
            return Decision(False, "only pulls by name are allowed")
        return Decision(True)
    if method == "POST" and path == "/containers/create":
        try:
            parsed = cast("JsonDict", json.loads(body or b"{}"))
        except json.JSONDecodeError:
            return Decision(False, "create body is not JSON")
        ctx = daemon.context()
        reason = check_create(parsed, ctx, daemon.repo_digests(str(parsed.get("Image") or "")))
        if reason is not None:
            return Decision(False, reason)
        return Decision(True, body=json.dumps(with_guard_label(parsed)).encode())
    match = _CONTAINER_OP.fullmatch(path)
    if match is None:
        return Decision(False, f"{method} {path} is not something the core does")
    ref, op = match.group(1), match.group(2) or ""
    record = daemon.container(ref)
    if record is None:
        # The daemon answers 404 itself; `docker inspect` relies on that to try
        # the name as an image next.
        return Decision(True)
    labels = cast("JsonDict", cast("JsonDict", record.get("Config") or {}).get("Labels") or {})
    ours = labels.get(GUARD_LABEL) == "1"
    if (method, op) == ("GET", "/json"):
        same_project = labels.get("com.docker.compose.project") == daemon.context().project
        return Decision(ours or same_project, "" if ours or same_project else "not this project's")
    if (method, op) in _LABELLED_OPS and ours:
        return Decision(True, upgrade=(method, op) == ("POST", "/attach"))
    return Decision(False, f"{method} {path} is not allowed on this container")


# ── the proxy ────────────────────────────────────────────────────────────────


async def _read_head(reader: asyncio.StreamReader) -> bytes:
    return await reader.readuntil(b"\r\n\r\n")


def _parse_head(head: bytes) -> tuple[str, str, list[tuple[str, str]]]:
    lines = head.decode("latin-1").split("\r\n")
    method, target, _ = lines[0].split(" ", 2)
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if line:
            name, _, value = line.partition(":")
            headers.append((name.strip(), value.strip()))
    return method, target, headers


def _render_head(method: str, target: str, headers: list[tuple[str, str]]) -> bytes:
    lines = [f"{method} {target} HTTP/1.1", *(f"{k}: {v}" for k, v in headers), "", ""]
    return "\r\n".join(lines).encode("latin-1")


def _refusal(reason: str) -> bytes:
    body = json.dumps({"message": f"refused by medmcp-docker-guard: {reason}"}).encode()
    head = (
        "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    )
    return head.encode() + body


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(Exception):
            if writer.can_write_eof():
                writer.write_eof()


class Guard:
    """The filtering socket server."""

    def __init__(self, listen: str, upstream: str, daemon: Daemon) -> None:
        """Serve on *listen*, forwarding allowed requests to *upstream*."""
        self._listen = listen
        self._upstream = upstream
        self._daemon = daemon

    async def serve(self) -> None:
        """Run until cancelled."""
        Path(self._listen).parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._listen)
        server = await asyncio.start_unix_server(self._handle, path=self._listen, limit=_HEAD_LIMIT)
        os.chmod(self._listen, 0o600)
        log.warning("docker guard listening at %s", self._listen)
        async with server:
            await server.serve_forever()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._handle_one(reader, writer)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            pass
        except Exception:
            log.exception("docker guard request failed")
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    async def _handle_one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        method, target, headers = _parse_head(await _read_head(reader))
        names = {k.lower(): v for k, v in headers}
        path = _VERSION_PREFIX.sub("", unquote(urlsplit(target).path))
        if "transfer-encoding" in names:
            writer.write(_refusal("request bodies must have a length"))
            await writer.drain()
            return
        length = int(names.get("content-length", "0") or 0)
        if length > _BODY_LIMIT:
            writer.write(_refusal("request body too large"))
            await writer.drain()
            return
        body = await reader.readexactly(length) if length else b""
        decision = await asyncio.to_thread(check_request, method, target, body, self._daemon)
        if not decision.allowed:
            log.warning("refused %s %s: %s", method, path, decision.reason)
            writer.write(_refusal(decision.reason))
            await writer.drain()
            return
        if decision.body is not None:
            body = decision.body
        # Exactly one request goes upstream: the connection is closed after its
        # response, unless this is an attach the guard allowed to upgrade.
        hop = {"connection", "upgrade", "keep-alive", "content-length", "transfer-encoding"}
        headers = [(k, v) for k, v in headers if k.lower() not in hop]
        if body:
            headers.append(("Content-Length", str(len(body))))
        if decision.upgrade and names.get("upgrade"):
            headers += [("Connection", "Upgrade"), ("Upgrade", names["upgrade"])]
        else:
            headers.append(("Connection", "close"))
        up_reader, up_writer = await asyncio.open_unix_connection(self._upstream)
        try:
            up_writer.write(_render_head(method, target, headers) + body)
            await up_writer.drain()
            response_head = await up_reader.readuntil(b"\r\n\r\n")
            writer.write(response_head)
            await writer.drain()
            status = response_head.split(b" ", 2)[1] if b" " in response_head else b""
            if decision.upgrade and status == b"101":
                await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
            else:
                await _pipe(up_reader, writer)
        finally:
            with contextlib.suppress(Exception):
                up_writer.close()


def _own_project(upstream: str) -> str:
    """The compose project this guard was started in (its container's label)."""
    ref = os.environ.get("HOSTNAME", "")
    with httpx.Client(transport=httpx.HTTPTransport(uds=upstream), base_url="http://docker") as c:
        resp = c.get(f"/containers/{ref}/json")
    labels = cast("JsonDict", cast("JsonDict", resp.json().get("Config") or {}).get("Labels") or {})
    project = str(labels.get("com.docker.compose.project") or "")
    if resp.status_code != 200 or not project:
        sys.exit("medmcp-docker-guard: cannot tell which compose project to guard")
    return project


def main() -> None:
    """Entry point: ``medmcp-docker-guard`` (configured by environment)."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    upstream = os.environ.get("MEDMCP_GUARD_UPSTREAM", "/var/run/docker.sock")
    listen = os.environ.get("MEDMCP_GUARD_SOCKET", "/run/medmcp-guard/docker.sock")
    project = os.environ.get("MEDMCP_GUARD_PROJECT", "") or _own_project(upstream)
    workspace = os.environ.get("MEDMCP_WORKSPACE", "")
    if not workspace:
        sys.exit("medmcp-docker-guard: MEDMCP_WORKSPACE is required")
    daemon = Daemon(upstream, project, workspace)
    asyncio.run(Guard(listen, upstream, daemon).serve())
