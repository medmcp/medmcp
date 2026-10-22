"""``medmcp-docker-guard``: the core's Docker socket, filtered.

The core starts containers — stacks, the isolated agent, the update helper — and
for that it held the Docker socket itself, which is the host user (or host root,
on a rootful daemon): a bug anywhere in the core would have been a way onto the
host. This guard runs as its own small container holding the real socket and
hands the core a filtered one. It forwards only what the core does:

- reads: ping, version, info, image and network inspection, container lists;
  a container's own details only for containers of this compose project or that
  the guard created;
- pulls and registry logins;
- ``POST /containers/create`` after checking the body: no ``--privileged``, no
  host or container namespaces, no capabilities beyond ``DAC_OVERRIDE``, no
  security option beyond ``no-new-privileges``, no devices but NVIDIA GPUs, and
  bind mounts only of the workspace and the core's own volumes. The real socket
  may be mounted only into the update helper (``medmcp-update`` from the core's
  own image repository), and only with the workspace unchanged. Every container
  created gets a label marking it as the guard's;
- start, attach, wait, resize, kill, stop, remove and copy-out only for
  containers carrying that label.

Everything else — exec, build, image push, swarm, plugins, volume and network
creation, container update — is refused. One request per connection: each is
forwarded with ``Connection: close`` (or as the upgrade it asks for, for
``attach``) and the bytes are then piped both ways untouched.
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
from urllib.parse import unquote, urlsplit

import httpx

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
        ("POST", r"/images/create"),
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


@dataclass
class Decision:
    """Forward (with *body* replacing the request's when set), or refuse with *reason*."""

    allowed: bool
    reason: str = ""
    body: bytes | None = None


def _repo(image: str) -> str:
    """``ghcr.io/medmcp/core:v1@sha256:…`` → ``ghcr.io/medmcp/core``."""
    name = image.split("@", 1)[0]
    head, _, tail = name.rpartition("/")
    return f"{head}/{tail.split(':', 1)[0]}" if head else tail.split(":", 1)[0]


def _bind_source(bind: str) -> str:
    return bind.split(":", 1)[0]


def check_create(body: JsonDict, ctx: GuardContext, image_repo_of: dict[str, str]) -> str | None:
    """Why a ``POST /containers/create`` body must be refused, or ``None``.

    *image_repo_of* resolves an image reference (an id, for the agent) to its
    repository, so the update helper is recognised by what it is, not by name.
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
    image = str(body.get("Image") or "")
    entrypoint = body.get("Entrypoint")
    is_updater = image_repo_of.get(image, _repo(image)) == ctx.core_image_repo and entrypoint in (
        [UPDATER_ENTRYPOINT],
        UPDATER_ENTRYPOINT,
    )
    allowed = {ctx.workspace, *ctx.mount_sources}
    for source in sources:
        if source == ctx.socket_source:
            if not is_updater:
                return "the docker socket may only be mounted into the update helper"
            continue
        if source not in allowed:
            return f"mount source {source!r} is not the workspace or one of the core's volumes"
    if is_updater:
        return _check_updater_env(body, ctx)
    return None


def _check_updater_env(body: JsonDict, ctx: GuardContext) -> str | None:
    """The helper may redeploy the core, but not onto another workspace."""
    for item in cast("list[Any]", body.get("Env") or []):
        key, _, value = str(item).partition("=")
        if key != "MEDMCP_UPDATE_ENV":
            continue
        try:
            carried = cast("JsonDict", json.loads(value))
        except json.JSONDecodeError:
            return "MEDMCP_UPDATE_ENV is not JSON"
        workspace = carried.get("MEDMCP_WORKSPACE")
        if workspace is not None and workspace != ctx.workspace:
            return "the update helper may not change the workspace"
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

    def image_repo(self, ref: str) -> str:
        """The repository an image reference (tag or id) belongs to."""
        found = self._get(f"/images/{ref}/json")
        if isinstance(found, dict):
            for tag in cast("list[Any]", found.get("RepoTags") or []) + cast(
                "list[Any]", found.get("RepoDigests") or []
            ):
                return _repo(str(tag))
        return _repo(ref)

    def context(self) -> GuardContext:
        """The core's mounts, networks and image, re-read every few seconds."""
        if self._ctx is not None and time.monotonic() - self._ctx_at < 10:
            return self._ctx
        flt = json.dumps(
            {
                "label": [
                    f"com.docker.compose.project={self._project}",
                    "com.docker.compose.service=medmcp",
                ]
            }
        )
        listed = self._client.get("/containers/json", params={"filters": flt}).json()
        core = self.container(str(listed[0]["Id"])) if listed else None
        mounts: set[str] = set()
        networks: set[str] = set()
        repo = ""
        if core is not None:
            for mount in cast("list[JsonDict]", core.get("Mounts") or []):
                if str(mount.get("Destination", "")).startswith("/run/medmcp-guard"):
                    continue  # the filtered socket stays with the core
                mounts.add(str(mount.get("Name") or mount.get("Source") or ""))
            networks = set(
                cast(
                    "JsonDict",
                    cast("JsonDict", core.get("NetworkSettings") or {}).get("Networks") or {},
                )
            )
            repo = _repo(str(cast("JsonDict", core.get("Config") or {}).get("Image") or ""))
        self._ctx = GuardContext(
            project=self._project,
            workspace=self._workspace,
            socket_source=os.environ.get("MEDMCP_DOCKER_SOCKET_SOURCE", ""),
            core_image_repo=repo,
            mount_sources=mounts - {""},
            networks=networks,
        )
        self._ctx_at = time.monotonic()
        return self._ctx


def check_request(method: str, target: str, body: bytes, daemon: Daemon) -> Decision:
    """Whether to forward one request (see the module docstring)."""
    parts = urlsplit(target)
    path = _VERSION_PREFIX.sub("", unquote(parts.path))
    for allowed_method, pattern in _READS:
        if method == allowed_method and pattern.fullmatch(path):
            return Decision(True)
    if method == "POST" and path == "/containers/create":
        try:
            parsed = cast("JsonDict", json.loads(body or b"{}"))
        except json.JSONDecodeError:
            return Decision(False, "create body is not JSON")
        ctx = daemon.context()
        image = str(parsed.get("Image") or "")
        reason = check_create(parsed, ctx, {image: daemon.image_repo(image)})
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
        return Decision(True)
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
        chunked = "chunked" in names.get("transfer-encoding", "").lower()
        length = int(names.get("content-length", "0") or 0)
        if length > _BODY_LIMIT:
            writer.write(_refusal("request body too large"))
            await writer.drain()
            return
        body = await reader.readexactly(length) if length else b""
        path = _VERSION_PREFIX.sub("", unquote(urlsplit(target).path))
        if chunked and path == "/containers/create":
            writer.write(_refusal("a create body must have a length"))
            await writer.drain()
            return
        decision = await asyncio.to_thread(check_request, method, target, body, self._daemon)
        if not decision.allowed:
            log.warning("refused %s %s: %s", method, path, decision.reason)
            writer.write(_refusal(decision.reason))
            await writer.drain()
            return
        if decision.body is not None:
            body = decision.body
            headers = [(k, v) for k, v in headers if k.lower() != "content-length"]
            headers.append(("Content-Length", str(len(body))))
        upgrade = names.get("connection", "").lower() == "upgrade"
        if not upgrade:
            headers = [(k, v) for k, v in headers if k.lower() != "connection"]
            headers.append(("Connection", "close"))
        up_reader, up_writer = await asyncio.open_unix_connection(self._upstream)
        try:
            up_writer.write(_render_head(method, target, headers) + body)
            await up_writer.drain()
            await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))
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
