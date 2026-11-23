"""End-to-end check of the containerized install's isolation (``just e2e-isolation``).

Runs the shipped ``docker-compose.ghcr.yml`` — with a scripted model in place of
GPU Ollama and the image under test in place of the published one — and checks
from outside and inside that:

- the agent runs in its own container: read-only root, every capability dropped
  but ``DAC_OVERRIDE``, no docker socket, no state volume, no credentials;
- the agent has no route out: neither the internet nor the model server, only the
  core's egress gateway, which its config names;
- a stack tool runs only when approved: the approved call succeeds, the same tool
  called straight at the broker from the agent's shell is refused;
- the agent cannot rewrite its own config;
- the core's docker access goes through the guard, which refuses a privileged
  container, a host mount and exec into another container;
- the API refuses a request without the password, and the agent's network even
  with it.

Needs docker with compose ≥ 2.24 and the image to test (built, e.g. by
``just docker-build``). Exit status 0 when every check holds.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import websockets
from websockets.typing import Origin

JsonDict = dict[str, Any]

REPO = Path(__file__).resolve().parents[2]
PROJECT = "medmcp-e2e"
PORT = 18100
PASSWORD = "e2e-password-0123456789abcdef"
STACK_IMAGE = "medmcp-e2e-stack:latest"
BASE = f"http://127.0.0.1:{PORT}"
AUTH = {"Authorization": f"Bearer {PASSWORD}"}

OVERRIDE = """\
services:
  llm:
    image: {image}
    pull_policy: never
    entrypoint: ["python3", "-u", "/e2e/fake_llm.py"]
    devices: !reset []
    volumes: !override
      - "{fake_llm}:/e2e/fake_llm.py:ro"
    healthcheck: !override
      test:
        - CMD
        - python3
        - -c
        - import socket; socket.create_connection(('127.0.0.1', 11434))
      interval: 2s
      retries: 30
  llm-init:
    profiles: [never]
  llm-warmup:
    profiles: [never]
  docker-guard:
    image: {image}
    pull_policy: never
  medmcp:
    image: {image}
    pull_policy: never
    depends_on: !override
      llm:
        condition: service_healthy
      docker-guard:
        condition: service_started
    environment:
      MEDMCP_LLM_SHIM: "0"
      MEDMCP_UPDATE_CHECK: "0"
      MEDMCP_AUTO_TITLES: "0"
      MEDMCP_PASSWORD: "{password}"
      MEDMCP_CATALOG_URL: "/app/catalog.ghcr.json"
      GHCR_TOKEN: "e2e-registry-token-must-not-reach-the-agent"
    ports: !override
      - "127.0.0.1:{port}:8100"
"""

STACK_DOCKERFILE = """\
FROM {image}
COPY server.py /e2e/server.py
COPY skills /e2e/skills
LABEL org.medmcp.stack='{{"name": "medmcp-e2e", "skills_path": "/e2e/skills"}}'
ENTRYPOINT ["python3", "/e2e/server.py"]
"""


class CheckFailedError(Exception):
    """One of the isolation checks did not hold."""


def _check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        raise CheckFailedError(what)


def _run(*args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=900)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


def _wait_healthy(timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(f"{BASE}/healthz", timeout=2).status_code == 200:
                return
        time.sleep(1)
    raise RuntimeError("the core did not come up")


async def _chat() -> list[str]:
    """Run one chat, approving each call once; return the tool outputs in order."""
    outputs: list[str] = []
    async with websockets.connect(
        f"ws://127.0.0.1:{PORT}/ws/chat",
        origin=Origin(f"http://127.0.0.1:{PORT}"),
        additional_headers=AUTH,
        max_size=None,
    ) as ws:
        sent = False
        while True:
            frame: JsonDict = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
            kind = frame.get("type")
            if kind == "ready" and not sent:
                await ws.send(json.dumps({"type": "prompt", "text": "run the checks"}))
                sent = True
            elif kind == "permission_request":
                opts: list[JsonDict] = frame["options"]
                once = next((o for o in opts if o["kind"] == "allow_once"), opts[0])
                await ws.send(
                    json.dumps(
                        {
                            "type": "permission",
                            "requestId": frame["requestId"],
                            "optionId": once["optionId"],
                        }
                    )
                )
            elif kind == "tool_call_update" and frame.get("status") in ("completed", "failed"):
                outputs.append(json.dumps(frame.get("output") or frame.get("content") or ""))
            elif kind == "error":
                raise RuntimeError(f"chat failed: {frame}")
            elif kind == "done":
                return outputs


def _checks(env: dict[str, str]) -> None:
    core = f"{PROJECT}-medmcp-1"
    _check(
        httpx.get(f"{BASE}/api/workspace", timeout=10).status_code == 401, "API needs the password"
    )

    resp = httpx.post(
        f"{BASE}/api/stacks/install", json={"image": STACK_IMAGE}, headers=AUTH, timeout=300
    )
    _check(resp.status_code == 200, f"stack installs through the guard ({resp.status_code})")

    outputs = asyncio.run(_chat())
    _check(len(outputs) == 3, f"three tool calls ran ({len(outputs)})")
    probe, stack_call, rewrite = outputs
    for marker, what in (
        ("NET-BLOCKED", "agent has no internet"),
        ("LLM-BLOCKED", "agent cannot reach the model server directly"),
        ("SOCKET-ABSENT", "agent has no docker socket"),
        ("STATE-ABSENT", "agent has no state volume"),
        ("CapEff:\\t0000000000000002", "agent keeps only DAC_OVERRIDE"),
        ("ROOT-READONLY", "agent's root file system is read-only"),
        ("WORKSPACE-WRITABLE", "agent can write the workspace"),
        ("not approved", "unapproved broker call is refused"),
        ("GATEWAY-403", "gateway refuses an encoded path to Ollama's admin API"),
    ):
        _check(marker in probe, what)
    _check("\\n0\\n" in probe, "agent environment holds no credential-shaped variable")
    _check("approved" in stack_call and "isError" not in stack_call, "approved stack call runs")
    _check("Read-only file system" in rewrite, "agent cannot rewrite its config")

    agents = _run(
        "docker", "ps", "-q", "--filter", f"label=org.medmcp.agent-of={core}", env=env
    ).split()
    _check(len(agents) == 1, "one agent container")
    info: JsonDict = json.loads(_run("docker", "inspect", agents[0], env=env))[0]
    host = info["HostConfig"]
    _check(host["ReadonlyRootfs"] is True, "agent container: read-only root (inspect)")
    _check(host["CapDrop"] == ["ALL"], "agent container: capabilities dropped (inspect)")
    mounts = [m["Destination"] for m in info["Mounts"]]
    _check(not any("docker" in m for m in mounts), "agent container: no socket mount (inspect)")

    networks: JsonDict = json.loads(_run("docker", "inspect", core, env=env))[0]["NetworkSettings"][
        "Networks"
    ]
    core_ip = networks[f"{PROJECT}_agent"]["IPAddress"]
    # A loopback Host and the password, so only the agent-network refusal can stop it.
    probe_ui = (
        "import urllib.request, urllib.error\n"
        f"req = urllib.request.Request('http://{core_ip}:8100/api/access', headers={{"
        f"'Host': 'localhost:8100', 'Authorization': 'Bearer {PASSWORD}'}})\n"
        "try:\n    print(urllib.request.urlopen(req, timeout=5).status)\n"
        "except urllib.error.HTTPError as e:\n    print(e.code, e.read().decode())\n"
    )
    status = _run("docker", "exec", agents[0], "python3", "-c", probe_ui, env=env, check=False)
    _check(
        status.startswith("403") and "agent network" in status,
        f"agent cannot use the workspace API ({status.strip()})",
    )

    for args, what in (
        (["run", "--rm", "--privileged", STACK_IMAGE], "a privileged container"),
        (["run", "--rm", "-v", "/:/host", STACK_IMAGE], "a host root mount"),
        (["exec", f"{PROJECT}-llm-1", "true"], "exec into another container"),
        (["import", "/dev/null", "ghcr.io/medmcp/core:e2e"], "an image import"),
    ):
        out = _run("docker", "exec", core, "docker", *args, env=env, check=False)
        _check("refused by medmcp-docker-guard" in out, f"guard refuses {what}")


def main() -> int:
    """Set up, check, tear down."""
    parser = argparse.ArgumentParser(description="End-to-end isolation check")
    parser.add_argument("--image", default="medmcp-core:dev", help="core image to test")
    parser.add_argument("--keep", action="store_true", help="leave the project running")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="medmcp-e2e-")).resolve()
    workspace = tmp / "ws"
    workspace.mkdir()
    shutil.copy(Path(__file__).with_name("fake_llm.py"), tmp / "fake_llm.py")
    stack = tmp / "stack"
    (stack / "skills" / "echo").mkdir(parents=True)
    shutil.copy(REPO / "tests" / "fake_stack_server.py", stack / "server.py")
    (stack / "skills" / "echo" / "SKILL.md").write_text(
        "---\nname: echo\ndescription: e2e skill\n---\nEcho.\n"
    )
    (stack / "Dockerfile").write_text(STACK_DOCKERFILE.format(image=args.image))
    override = tmp / "override.yml"
    override.write_text(
        OVERRIDE.format(
            image=args.image, fake_llm=tmp / "fake_llm.py", password=PASSWORD, port=PORT
        )
    )
    env = {
        **os.environ,
        "MEDMCP_WORKSPACE": str(workspace),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR") or "/var/run",
    }
    compose = [
        "docker", "compose", "-p", PROJECT,
        "-f", str(REPO / "docker-compose.ghcr.yml"), "-f", str(override),
    ]  # fmt: skip
    status = 1
    try:
        _run("docker", "build", "-q", "-t", STACK_IMAGE, str(stack), env=env)
        _run(*compose, "up", "-d", env=env)
        _wait_healthy()
        _checks(env)
        status = 0
    except (CheckFailedError, RuntimeError) as exc:
        print(f"isolation check failed: {exc}", file=sys.stderr)
        print(_run(*compose, "logs", "--tail", "80", env=env, check=False), file=sys.stderr)
    finally:
        if not args.keep:
            _run(*compose, "down", "-v", env=env, check=False)
            leftovers = _run(
                "docker", "ps", "-aq", "--filter", f"label=org.medmcp.agent-of={PROJECT}-medmcp-1",
                env=env, check=False,
            ).split()  # fmt: skip
            if leftovers:
                _run("docker", "rm", "-f", *leftovers, env=env, check=False)
            _run("docker", "rmi", STACK_IMAGE, env=env, check=False)
            shutil.rmtree(tmp, ignore_errors=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
