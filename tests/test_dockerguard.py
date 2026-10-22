"""The docker guard: what the core may ask the daemon for, and nothing more."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from medmcp import dockerguard
from medmcp.dockerguard import GuardContext, check_create, check_request

JsonDict = dict[str, Any]
WS = "/srv/data"
SOCK = "/run/user/1000/docker.sock"
CTX = GuardContext(
    project="medmcp",
    workspace=WS,
    socket_source=SOCK,
    core_image_repo="ghcr.io/medmcp/core",
    mount_sources={"medmcp_vibe-state", "medmcp_vibe-agent", "medmcp_vibe-logs"},
    networks={"medmcp_default", "medmcp_agent"},
)


def _stack(**host: object) -> JsonDict:
    """A stack's create body as the docker CLI sends it, with *host* overrides."""
    base: JsonDict = {
        "NetworkMode": "none",
        "CapDrop": ["ALL"],
        "CapAdd": ["CAP_DAC_OVERRIDE"],
        "SecurityOpt": ["no-new-privileges"],
        "Binds": [f"{WS}:{WS}"],
        "DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=0"]}],
    }
    return {"Image": "ghcr.io/medmcp/neuro-core:main", "HostConfig": {**base, **host}}


class TestCreate:
    """``POST /containers/create`` bodies."""

    def test_stacks_the_agent_and_skill_extraction_pass(self) -> None:
        """Everything the core really creates."""
        agent = _stack(
            NetworkMode="medmcp_agent",
            Binds=[f"{WS}:{WS}", "medmcp_vibe-agent:/app/.vibe/agent-home:ro"],
            DeviceRequests=None,
        )
        assert check_create(_stack(), CTX, {}) is None
        assert check_create(agent, CTX, {}) is None
        assert check_create({"Image": "img", "HostConfig": {}}, CTX, {}) is None

    @pytest.mark.parametrize(
        ("host", "reason"),
        [
            ({"Privileged": True}, "privileged"),
            ({"PidMode": "host"}, "PidMode"),
            ({"IpcMode": "container:core"}, "IpcMode"),
            ({"NetworkMode": "host"}, "network"),
            ({"NetworkMode": "container:medmcp-medmcp-1"}, "network"),
            ({"CapAdd": ["SYS_ADMIN"]}, "capabilities"),
            ({"SecurityOpt": ["seccomp=unconfined"]}, "security options"),
            ({"Devices": [{"PathOnHost": "/dev/sda"}]}, "Devices"),
            ({"VolumesFrom": ["medmcp-medmcp-1"]}, "VolumesFrom"),
            ({"MaskedPaths": []}, "MaskedPaths"),
            ({"Binds": ["/:/host"]}, "mount source"),
            ({"Binds": [f"{SOCK}:/var/run/docker.sock"]}, "docker socket"),
            ({"Mounts": [{"Type": "bind", "Source": "/etc", "Target": "/e"}]}, "mount source"),
            ({"Binds": ["ollama-models:/m"]}, "mount source"),
            ({"DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["vendor.com/x=1"]}]}, "device"),
        ],
    )
    def test_escapes_are_refused(self, host: JsonDict, reason: str) -> None:
        """Each of these is a way out of the sandbox, or onto the host."""
        refused = check_create(_stack(**host), CTX, {})
        assert refused is not None and reason in refused

    def test_only_the_update_helper_gets_the_socket_and_not_a_new_workspace(self) -> None:
        """The helper is recognised by its image repository and entry point."""
        helper: JsonDict = {
            "Image": "sha256:abc",
            "Entrypoint": ["medmcp-update"],
            "Env": [f"MEDMCP_UPDATE_ENV={json.dumps({'MEDMCP_WORKSPACE': WS})}"],
            "HostConfig": {"Binds": [f"{SOCK}:/var/run/docker.sock", "medmcp_vibe-state:/s"]},
        }
        repos = {"sha256:abc": "ghcr.io/medmcp/core"}
        assert check_create(helper, CTX, repos) is None
        moved = {**helper, "Env": ['MEDMCP_UPDATE_ENV={"MEDMCP_WORKSPACE": "/"}']}
        assert "workspace" in (check_create(moved, CTX, repos) or "")
        impostor = {**helper, "Entrypoint": ["sh"]}
        assert "socket" in (check_create(impostor, CTX, repos) or "")
        foreign = {**helper, "Image": "evil/core"}
        assert "socket" in (check_create(foreign, CTX, {}) or "")


class _Daemon:
    """A daemon stand-in: two containers, one the guard's, one another project's."""

    def __init__(self) -> None:
        self.containers: dict[str, JsonDict] = {
            "ours": {"Config": {"Labels": {dockerguard.GUARD_LABEL: "1"}}},
            "core": {"Config": {"Labels": {"com.docker.compose.project": "medmcp"}}},
            "other": {"Config": {"Labels": {"com.docker.compose.project": "bench"}}},
        }

    def container(self, ref: str) -> JsonDict | None:
        return self.containers.get(ref)

    def image_repo(self, ref: str) -> str:
        return ref

    def context(self) -> GuardContext:
        return CTX


def _check(method: str, target: str, body: bytes = b"") -> dockerguard.Decision:
    return check_request(method, target, body, cast("dockerguard.Daemon", _Daemon()))


class TestRequests:
    """Endpoints and container ownership."""

    @pytest.mark.parametrize(
        ("method", "target"),
        [
            ("GET", "/v1.47/_ping"),
            ("GET", "/v1.47/images/ghcr.io/medmcp/neuro-core:main/json"),
            ("POST", "/v1.47/images/create?fromImage=ghcr.io/medmcp/x&tag=main"),
            ("GET", "/v1.47/containers/json?all=1"),
            ("GET", "/v1.47/networks/medmcp_agent"),
            ("POST", "/v1.47/containers/ours/attach?stream=1&stdin=1"),
            ("POST", "/v1.47/containers/ours/start"),
            ("DELETE", "/v1.47/containers/ours?force=1"),
            ("GET", "/v1.47/containers/ours/archive?path=/skills"),
            ("GET", "/v1.47/containers/core/json"),
            ("GET", "/v1.47/containers/missing/json"),
        ],
    )
    def test_what_the_core_does_is_forwarded(self, method: str, target: str) -> None:
        """Pulls, inspection, and the lifecycle of containers it created."""
        assert _check(method, target).allowed

    @pytest.mark.parametrize(
        ("method", "target"),
        [
            ("POST", "/v1.47/containers/ours/exec"),
            ("POST", "/v1.47/build"),
            ("POST", "/v1.47/images/x/push"),
            ("POST", "/v1.47/networks/create"),
            ("POST", "/v1.47/volumes/create"),
            ("POST", "/v1.47/containers/ours/update"),
            ("PUT", "/v1.47/containers/ours/archive?path=/"),
            ("POST", "/v1.47/containers/core/kill"),
            ("DELETE", "/v1.47/containers/other"),
            ("GET", "/v1.47/containers/other/json"),
            ("POST", "/v1.47/plugins/pull"),
        ],
    )
    def test_everything_else_is_refused(self, method: str, target: str) -> None:
        """Exec, build, push, and touching containers that are not the guard's."""
        assert not _check(method, target).allowed

    def test_a_created_container_is_labelled_as_the_guards(self) -> None:
        """The label is what later lifecycle calls are checked against."""
        decision = _check("POST", "/v1.47/containers/create?name=x", json.dumps(_stack()).encode())
        assert decision.allowed and decision.body is not None
        assert json.loads(decision.body)["Labels"][dockerguard.GUARD_LABEL] == "1"


async def test_the_proxy_forwards_one_request_and_refuses_another(tmp_path: Path) -> None:
    """Over real sockets: an allowed request reaches the daemon, a refused one never does."""
    seen: list[bytes] = []

    async def fake_daemon(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
        await writer.drain()
        writer.close()

    upstream, listen = str(tmp_path / "d.sock"), str(tmp_path / "g.sock")
    daemon_server = await asyncio.start_unix_server(fake_daemon, path=upstream)
    guard = dockerguard.Guard(listen, upstream, cast("dockerguard.Daemon", _Daemon()))
    task = asyncio.create_task(guard.serve())
    for _ in range(50):
        if Path(listen).exists():
            break
        await asyncio.sleep(0.02)

    async def request(line: str) -> bytes:
        reader, writer = await asyncio.open_unix_connection(listen)
        writer.write(f"{line} HTTP/1.1\r\nHost: docker\r\n\r\n".encode())
        await writer.drain()
        data = await reader.read()
        writer.close()
        return data

    try:
        assert (await request("GET /v1.47/_ping")).endswith(b"OK")
        assert b"403" in (await request("POST /v1.47/containers/ours/exec"))
        assert len(seen) == 1
        assert b"Connection: close" in seen[0]
        assert Path(listen).stat().st_mode & 0o077 == 0
    finally:
        task.cancel()
        daemon_server.close()
