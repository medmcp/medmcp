"""Reproduce: a stack call its tool's schema rejects ends the whole chat turn.

``just e2e-invalid-args IMAGE`` runs the shipped ``docker-compose.ghcr.yml``
with a scripted model (``invalid_args_llm.py``) and a one-tool stack
(``args_stack.py``) whose schema nests a model as ``$defs``/``$ref``. The model
calls the tool from ``run_typescript`` with a list inside the list the schema
allows (``labels: [["a", "b"]]`` where it takes a string or a list of strings).

Under vibe 2.26.1 the path guard's ``pre_tool`` hook let that call continue;
vibe then validated the arguments once more, rejected them as an invalid
command, and the turn died with ``Core rejected durable input``. With the fix
the guard denies the call with what is wrong, and the turn goes on.

Prints the chat's errors and what the model was told, then ``FIXED`` (exit 0)
or ``BROKEN`` (exit 1). Needs docker with compose >= 2.24.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROJECT = "medmcp-e2e-args"
PORT = 18110
PASSWORD = "e2e-password-0123456789abcdef"
STACK_IMAGE = "medmcp-e2e-args-stack:latest"
BASE = f"http://127.0.0.1:{PORT}"
AUTH = {"Authorization": f"Bearer {PASSWORD}"}

OVERRIDE = """\
services:
  llm:
    image: {image}
    pull_policy: never
    entrypoint: ["python3", "-u", "/e2e/llm.py"]
    devices: !reset []
    volumes: !override
      - "{llm}:/e2e/llm.py:ro"
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


def _docker_runtime_dir() -> str:
    """Where the compose file finds ``docker.sock``: the rootless socket, else the system one."""
    rootless = os.environ.get("XDG_RUNTIME_DIR", "")
    return rootless if rootless and Path(rootless, "docker.sock").exists() else "/var/run"


def _run(*args: str, env: dict[str, str], check: bool = True) -> str:
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


async def _chat() -> tuple[list[str], bool]:
    """Run one chat, approving each call once; return its error frames and whether it ended."""
    errors: list[str] = []
    async with websockets.connect(
        f"ws://127.0.0.1:{PORT}/ws/chat",
        origin=Origin(f"http://127.0.0.1:{PORT}"),
        additional_headers=AUTH,
        max_size=None,
    ) as ws:
        sent = False
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=120)
            except TimeoutError:
                return errors, False
            frame: JsonDict = json.loads(raw)
            kind = frame.get("type")
            if kind == "ready" and not sent:
                await ws.send(json.dumps({"type": "prompt", "text": "pick the labels"}))
                sent = True
            elif kind == "permission_request":
                opts: list[JsonDict] = frame["options"]
                once = next((o for o in opts if o["kind"] == "allow_once"), opts[0])
                reply = {"requestId": frame["requestId"], "optionId": once["optionId"]}
                await ws.send(json.dumps({"type": "permission", **reply}))
            elif kind == "error":
                errors.append(str(frame.get("message") or frame))
            elif kind == "done":
                return errors, True


def _scenario(env: dict[str, str], compose: list[str]) -> bool:
    resp = httpx.post(
        f"{BASE}/api/stacks/install", json={"image": STACK_IMAGE}, headers=AUTH, timeout=300
    )
    if resp.status_code != 200:
        raise RuntimeError(f"stack install failed: {resp.status_code} {resp.text[:300]}")
    errors, ended = asyncio.run(_chat())
    told = [
        json.loads(line.split("TOOL-RESULT ", 1)[1])
        for line in _run(*compose, "logs", "--no-log-prefix", "llm", env=env).splitlines()
        if "TOOL-RESULT " in line
    ]
    print("chat errors:", json.dumps(errors, indent=2))
    print("the model was told:", json.dumps(told, indent=2))
    rejected = any("Core rejected durable input" in e for e in errors)
    return ended and not rejected and bool(told)


def main() -> int:
    """Set up, run the chat, tear down."""
    parser = argparse.ArgumentParser(description="Reproduce: an invalid stack call ends the turn")
    parser.add_argument("--image", default="medmcp-core:dev", help="core image to test")
    parser.add_argument("--keep", action="store_true", help="leave the project running")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="medmcp-e2e-args-")).resolve()
    workspace = tmp / "ws"
    workspace.mkdir()
    shutil.copy(HERE / "invalid_args_llm.py", tmp / "llm.py")
    stack = tmp / "stack"
    (stack / "skills" / "pick").mkdir(parents=True)
    shutil.copy(HERE / "args_stack.py", stack / "server.py")
    (stack / "skills" / "pick" / "SKILL.md").write_text(
        "---\nname: pick\ndescription: e2e skill\n---\nPick.\n"
    )
    (stack / "Dockerfile").write_text(STACK_DOCKERFILE.format(image=args.image))
    override = tmp / "override.yml"
    override.write_text(
        OVERRIDE.format(image=args.image, llm=tmp / "llm.py", password=PASSWORD, port=PORT)
    )
    env = {
        **os.environ,
        "MEDMCP_WORKSPACE": str(workspace),
        "XDG_RUNTIME_DIR": _docker_runtime_dir(),
    }
    compose = [
        "docker", "compose", "-p", PROJECT,
        "-f", str(REPO / "docker-compose.ghcr.yml"), "-f", str(override),
    ]  # fmt: skip
    fixed = False
    try:
        _run("docker", "build", "-q", "-t", STACK_IMAGE, str(stack), env=env)
        _run(*compose, "up", "-d", env=env)
        _wait_healthy()
        fixed = _scenario(env, compose)
    except RuntimeError as exc:
        print(f"could not run the scenario: {exc}", file=sys.stderr)
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
    print("FIXED: the call was turned back and the turn went on" if fixed else "BROKEN")
    return 0 if fixed else 1


if __name__ == "__main__":
    sys.exit(main())
