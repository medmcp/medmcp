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
DIGEST = "sha256:" + "a" * 64
OLD_DIGEST = "sha256:" + "b" * 64
HELPER_IMAGE = f"ghcr.io/medmcp/core@{DIGEST}"
CTX = GuardContext(
    project="medmcp",
    workspace=WS,
    socket_source=SOCK,
    core_image_repo="ghcr.io/medmcp/core",
    mount_sources={"medmcp_vibe-state", "medmcp_vibe-agent", "medmcp_vibe-logs"},
    networks={"medmcp_default", "medmcp_agent"},
    core_id="c0ffee",
    core_env={"MEDMCP_WORKSPACE": WS, "MEDMCP_GPU": "0"},
    core_digests={OLD_DIGEST},
    project_dir="/home/u/.cache/docker-compose/x",
    state_volume="medmcp_vibe-state",
    docker_config_source="/home/u/.docker/config.json",
    llm_image="ollama/ollama:0.40.7",
)


def _helper(**overrides: object) -> JsonDict:
    """The helper create body exactly as ``update.updater_command`` produces it."""
    carried = {
        "MEDMCP_WORKSPACE": WS,
        "MEDMCP_GPU": "0",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "HOME": "/home/u",
    }
    env = {
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "MEDMCP_UPDATE_REPLACE": "c0ffee",
        "MEDMCP_UPDATE_PROJECT": "medmcp",
        "MEDMCP_UPDATE_PROJECT_DIR": "/home/u/.cache/docker-compose/x",
        "MEDMCP_UPDATE_PREVIOUS_TAG": "v0.7.1",
        "MEDMCP_UPDATE_PREVIOUS_DIGEST": OLD_DIGEST,
        "MEDMCP_UPDATE_DIGEST": DIGEST,
        "MEDMCP_UPDATE_ENV": json.dumps(carried),
    }
    body: JsonDict = {
        "Image": HELPER_IMAGE,
        "Entrypoint": ["medmcp-update"],
        "Cmd": ["apply"],
        "Env": [f"{k}={v}" for k, v in env.items()],
        "HostConfig": {
            "Binds": [
                f"{SOCK}:/var/run/docker.sock",
                "medmcp_vibe-state:/app/.vibe/state",
                "/home/u/.docker/config.json:/run/host-docker-config.json:ro",
            ]
        },
    }
    return {**body, **overrides}


def _with_env(**changes: str) -> JsonDict:
    body = _helper()
    env = dict(str(e).split("=", 1) for e in body["Env"])
    env.update(changes)
    return {**body, "Env": [f"{k}={v}" for k, v in env.items()]}


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
        assert check_create(_stack(), CTX, []) is None
        assert check_create(agent, CTX, []) is None
        assert check_create({"Image": "img", "HostConfig": {}}, CTX, []) is None

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
            ({"Binds": [f"{SOCK}:/var/run/docker.sock"]}, "update helper"),
            ({"Mounts": [{"Type": "bind", "Source": "/etc", "Target": "/e"}]}, "mount source"),
            ({"Binds": ["ollama-models:/m"]}, "mount source"),
            ({"DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["vendor.com/x=1"]}]}, "device"),
        ],
    )
    def test_escapes_are_refused(self, host: JsonDict, reason: str) -> None:
        """Each of these is a way out of the sandbox, or onto the host."""
        refused = check_create(_stack(**host), CTX, [])
        assert refused is not None and reason in refused

    def test_the_update_helper_exactly_as_the_core_creates_it_passes(self) -> None:
        """The one container that may hold the real socket."""
        assert check_create(_helper(), CTX, [HELPER_IMAGE]) is None

    @pytest.mark.parametrize(
        ("body", "reason"),
        [
            # Code of the caller's choosing inside the helper.
            (_with_env(PYTHONPATH="/w/x"), "may not set"),
            (_with_env(LD_PRELOAD="/w/x.so"), "may not set"),
            (_helper(Cmd=["sh", "-c", "id"]), "medmcp-update apply"),
            (_helper(Entrypoint=["sh"]), "medmcp-update apply"),
            (_helper(User="0", WorkingDir="/w"), "user"),
            # A workspace bind, which would carry that code in.
            (
                _helper(
                    HostConfig={
                        "Binds": [
                            f"{SOCK}:/var/run/docker.sock",
                            "medmcp_vibe-state:/app/.vibe/state",
                            f"{WS}:/w",
                        ]
                    }
                ),
                "mounts",
            ),
            # Redeploying the stack onto other host paths or images.
            (_with_env(MEDMCP_WORKSPACE="/home/u"), "may not set"),
            (
                _with_env(MEDMCP_UPDATE_ENV=json.dumps({"MEDMCP_WORKSPACE": "/home/u"})),
                "workspace",
            ),
            (
                _with_env(
                    MEDMCP_UPDATE_ENV=json.dumps(
                        {"MEDMCP_WORKSPACE": WS, "OLLAMA_IMAGE": "evil/ollama"}
                    )
                ),
                "OLLAMA_IMAGE",
            ),
            (
                _with_env(MEDMCP_UPDATE_ENV=json.dumps({"MEDMCP_WORKSPACE": WS, "HOME": "/"})),
                "HOME",
            ),
            (_with_env(MEDMCP_UPDATE_PROJECT_DIR="/"), "PROJECT_DIR"),
            (_with_env(MEDMCP_UPDATE_PREVIOUS_DIGEST="sha256:" + "c" * 64), "roll back"),
            # An image named like the core's but not pulled as one.
            (_helper(Image="ghcr.io/medmcp/core:evil"), "by digest"),
        ],
    )
    def test_anything_else_holding_the_socket_is_refused(self, body: JsonDict, reason: str) -> None:
        """Each of these would turn the helper into a way onto the host."""
        refused = check_create(body, CTX, [HELPER_IMAGE])
        assert refused is not None and reason in refused

    def test_an_imported_image_cannot_pose_as_the_helper(self) -> None:
        """An import records no repo digest, whatever it is tagged as."""
        refused = check_create(_helper(), CTX, [])
        assert refused is not None and "registry" in refused


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

    def repo_digests(self, ref: str) -> list[str]:
        return [HELPER_IMAGE] if ref == HELPER_IMAGE else []

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
            ("POST", "/v1.47/images/create?fromImage=x&tag=1&platform=linux/amd64"),
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
            ("POST", "/v1.47/images/create?fromSrc=-&repo=ghcr.io/medmcp/core&tag=evil"),
            ("POST", "/v1.47/images/create?fromSrc=http://x/i.tar&repo=ghcr.io/medmcp/core"),
            ("POST", "/v1.47/images/create?fromImage=x&changes=ENV%20A=1"),
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


class _FakeDaemon:
    """A keep-alive HTTP server, like dockerd: it closes only when told to."""

    def __init__(self) -> None:
        self.seen: list[bytes] = []

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                self.seen.append(head)
                if b"/attach" in head.split(b"\r\n", 1)[0]:
                    writer.write(b"HTTP/1.1 101 UPGRADED\r\n\r\n")
                    await writer.drain()
                    while data := await reader.read(1024):
                        writer.write(b"echo:" + data)
                        await writer.drain()
                    break
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
                await writer.drain()
                if b"connection: close" in head.lower():
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        writer.close()


@pytest.fixture
async def proxy(tmp_path: Path) -> Any:  # noqa: ANN401 - an async generator fixture
    """A guard in front of a fake daemon; yields ``(guard socket path, fake daemon)``."""
    fake = _FakeDaemon()
    upstream, listen = str(tmp_path / "d.sock"), str(tmp_path / "g.sock")
    daemon_server = await asyncio.start_unix_server(fake.handle, path=upstream)
    guard = dockerguard.Guard(listen, upstream, cast("dockerguard.Daemon", _Daemon()))
    task = asyncio.create_task(guard.serve())
    for _ in range(50):
        if Path(listen).exists():
            break
        await asyncio.sleep(0.02)
    yield listen, fake
    task.cancel()
    daemon_server.close()


async def _exchange(listen: str, raw: bytes, then: bytes = b"") -> bytes:
    reader, writer = await asyncio.open_unix_connection(listen)
    writer.write(raw)
    await writer.drain()
    if then:
        await asyncio.sleep(0.1)
        writer.write(then)
        await writer.drain()
        writer.write_eof()
    data = await asyncio.wait_for(reader.read(), timeout=5)
    writer.close()
    return data


async def test_the_proxy_forwards_one_request_and_refuses_another(proxy: Any) -> None:  # noqa: ANN401
    """An allowed request reaches the daemon with Connection: close; a refused one never does."""
    listen, fake = proxy
    ok = await _exchange(listen, b"GET /v1.47/_ping HTTP/1.1\r\nHost: docker\r\n\r\n")
    assert ok.endswith(b"OK")
    refused = await _exchange(
        listen, b"POST /v1.47/containers/ours/exec HTTP/1.1\r\nHost: docker\r\n\r\n"
    )
    assert b"403" in refused
    assert len(fake.seen) == 1
    assert b"Connection: close" in fake.seen[0]
    assert Path(listen).stat().st_mode & 0o077 == 0


async def test_a_smuggled_request_never_reaches_the_daemon(proxy: Any) -> None:  # noqa: ANN401
    """An Upgrade header on an allowed request cannot keep the connection open for more."""
    listen, fake = proxy
    smuggled = (
        b"GET /v1.47/_ping HTTP/1.1\r\nHost: docker\r\nConnection: Upgrade\r\nUpgrade: tcp\r\n\r\n"
        b"POST /v1.47/containers/create HTTP/1.1\r\nHost: docker\r\nContent-Length: 2\r\n\r\n{}"
    )
    await _exchange(listen, smuggled)
    assert len(fake.seen) == 1
    assert b"Connection: close" in fake.seen[0]
    assert b"Upgrade" not in fake.seen[0]


async def test_a_chunked_body_is_refused(proxy: Any) -> None:  # noqa: ANN401
    """A body without a length could carry anything past the check."""
    listen, fake = proxy
    out = await _exchange(
        listen,
        b"POST /v1.47/auth HTTP/1.1\r\nHost: docker\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
    )
    assert b"403" in out
    assert fake.seen == []


async def test_attach_to_a_guard_container_upgrades(proxy: Any) -> None:  # noqa: ANN401
    """The one upgrade honoured: stdio for a container the guard created, after a 101."""
    listen, fake = proxy
    head = (
        b"POST /v1.47/containers/ours/attach?stream=1&stdin=1 HTTP/1.1\r\nHost: docker\r\n"
        b"Connection: Upgrade\r\nUpgrade: tcp\r\n\r\n"
    )
    out = await _exchange(listen, head, then=b"hello")
    assert out.startswith(b"HTTP/1.1 101")
    assert b"echo:hello" in out
    assert b"Upgrade: tcp" in fake.seen[0]
