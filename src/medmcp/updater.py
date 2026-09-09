"""``medmcp-update`` — the one-shot helper that swaps a running install to this image.

Started by the *previous* release's workspace server as a detached container
from the *new* image (see :func:`medmcp.update.updater_command`), with the
docker socket and the ``vibe-state`` volume mounted. It runs ``docker compose
up -d`` for the operator's project against the compose file baked into this
image, pinned to this image's own tag — so the topology that comes up is the
one this release was published with, and any compose change the release
carries (a model-server pin, a new volume) is applied along with the image.

The old core dies when compose recreates it, which is why this is a separate
container and why the outcome goes to a file on the shared volume rather than
back to whoever started it: the next core reads it on boot and shows it.

Rollback: if ``up`` fails, the same command is re-run with the previous tag, so
a release whose container does not start leaves the install where it was.

Everything it needs arrives in the environment:

``MEDMCP_UPDATE_REPLACE``
    Id of the core container being replaced (to recover its ``env_file``).
``MEDMCP_UPDATE_PROJECT`` / ``MEDMCP_UPDATE_PROJECT_DIR``
    The compose project and its recorded working directory.
``MEDMCP_UPDATE_PREVIOUS_TAG`` / ``MEDMCP_UPDATE_PREVIOUS_DIGEST``
    The tag to roll back to, and the digest it ran as (empty if unknown).
``MEDMCP_UPDATE_DIGEST``
    The verified digest of this image; compose pins the core to it.
``MEDMCP_UPDATE_ENV``
    JSON object of the compose variables the operator set at install time.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

log: logging.Logger = logging.getLogger("medmcp.updater")

COMPOSE_PATH: Path = Path("/app/docker-compose.ghcr.yml")
RESULT_PATH: Path = Path("/app/.vibe/state/update_result.json")
HOST_DOCKER_CONFIG: Path = Path("/run/host-docker-config.json")
ENV_FILE_PATH: Path = Path("/tmp/medmcp.env")
COMPOSE_TIMEOUT_SEC: float = 30 * 60
DOCKER_WAIT_SEC: float = 60.0

# Never re-exported through the env file: docker/image housekeeping and the
# variables compose itself sets on the service.
_ENV_FILE_SKIP: frozenset[str] = frozenset({"PATH", "HOME", "HOSTNAME", "TERM"})


def _docker(args: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def wait_for_docker(deadline_sec: float = DOCKER_WAIT_SEC) -> bool:
    """Wait until the mounted daemon answers."""
    end = time.monotonic() + deadline_sec
    while time.monotonic() < end:
        try:
            if _docker(["version", "--format", "{{.Server.Version}}"], timeout=10).returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(1)
    return False


def sanitize_docker_config(src: Path, dst: Path) -> None:
    """Copy only ``auths`` from the host docker config (mirrors the entrypoint)."""
    try:
        data = cast("dict[str, Any]", json.loads(src.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"auths": data.get("auths", {})}), encoding="utf-8")


def _inspect(ref: str, *, image: bool = False) -> dict[str, Any] | None:
    args = ["image", "inspect", ref] if image else ["inspect", ref]
    try:
        result = _docker(args, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        items = cast("list[dict[str, Any]]", json.loads(result.stdout))
    except json.JSONDecodeError:
        return None
    return items[0] if items else None


def _env_of(info: dict[str, Any] | None) -> dict[str, str]:
    if info is None:
        return {}
    config = cast("dict[str, Any]", info.get("Config") or {})
    out: dict[str, str] = {}
    for item in cast("list[str]", config.get("Env") or []):
        key, sep, value = item.partition("=")
        if sep:
            out[key] = value
    return out


def compose_service_env_keys(compose_text: str, service: str = "medmcp") -> set[str]:
    """The keys a compose file sets on *service* through ``environment:``."""
    try:
        doc = cast("dict[str, Any]", yaml.safe_load(compose_text) or {})
    except yaml.YAMLError:
        return set()
    services = cast("dict[str, Any]", doc.get("services") or {})
    svc = cast("dict[str, Any]", services.get(service) or {})
    env = svc.get("environment")
    if isinstance(env, dict):
        return set(cast("dict[str, Any]", env))
    if isinstance(env, list):
        return {str(item).partition("=")[0] for item in cast("list[Any]", env)}
    return set()


def env_file_entries(
    container_env: dict[str, str], image_env: dict[str, str], declared: set[str]
) -> dict[str, str]:
    """The variables that reached the old container through its ``env_file``.

    Those are the operator's secrets for external MCP servers (the compose file
    forwards nothing else that way). They are not on any label, so they are
    recovered by elimination: whatever is in the container's environment that
    neither the image nor the compose ``environment:`` block put there.
    """
    return {
        key: value
        for key, value in container_env.items()
        if key not in image_env and key not in declared and key not in _ENV_FILE_SKIP
    }


def write_env_file(path: Path, entries: dict[str, str]) -> None:
    """Write ``KEY=VALUE`` lines the way compose reads them (values verbatim)."""
    lines = [f"{key}={value}" for key, value in sorted(entries.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    path.chmod(0o600)


def compose_up(
    *,
    project: str,
    project_dir: str,
    compose_file: Path,
    env: dict[str, str],
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """``docker compose up -d`` for the project with *env* as the interpolation env.

    With *dry_run* it prints the resolved configuration (``compose config``)
    instead — what the update *would* bring up, for checking a plan by hand.
    """
    args = [
        "compose",
        "--project-name",
        project,
        "--project-directory",
        project_dir,
        "-f",
        str(compose_file),
    ]
    args += ["config"] if dry_run else ["up", "-d"]
    log.info("docker %s (MEDMCP_TAG=%s)", " ".join(args), env.get("MEDMCP_TAG", ""))
    return subprocess.run(
        ["docker", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=COMPOSE_TIMEOUT_SEC,
    )


def write_result(path: Path, result: dict[str, Any]) -> None:
    """Record the outcome on the shared volume for the next core to report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:].strip()


def apply(env: dict[str, str] | None = None, *, dry_run: bool = False) -> int:
    """Run the update; returns the process exit code.

    *dry_run* stops after printing the resolved compose configuration: nothing
    is started, rolled back, or recorded.
    """
    env = dict(os.environ if env is None else env)
    replace = env.get("MEDMCP_UPDATE_REPLACE", "")
    project = env.get("MEDMCP_UPDATE_PROJECT", "")
    project_dir = env.get("MEDMCP_UPDATE_PROJECT_DIR", "")
    previous = env.get("MEDMCP_UPDATE_PREVIOUS_TAG", "")
    previous_digest = env.get("MEDMCP_UPDATE_PREVIOUS_DIGEST", "")
    target = env.get("MEDMCP_BUILD", "")
    target_digest = env.get("MEDMCP_UPDATE_DIGEST", "")
    try:
        plan_env = cast("dict[str, str]", json.loads(env.get("MEDMCP_UPDATE_ENV") or "{}"))
    except json.JSONDecodeError:
        plan_env = {}
    result: dict[str, Any] = {
        "from": previous,
        "to": target,
        "status": "failed",
        "detail": "",
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if not (project and project_dir and target and previous):
        result["detail"] = "updater started without its project, tags, or build id"
        write_result(RESULT_PATH, result)
        log.error(result["detail"])
        return 2
    if not wait_for_docker():
        result["detail"] = "the docker socket did not answer"
        write_result(RESULT_PATH, result)
        log.error(result["detail"])
        return 2

    if HOST_DOCKER_CONFIG.exists():
        sanitize_docker_config(HOST_DOCKER_CONFIG, Path.home() / ".docker" / "config.json")

    compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
    run_env = {k: v for k, v in env.items() if not k.startswith("MEDMCP_UPDATE_")}
    run_env.update(plan_env)
    run_env["MEDMCP_TAG"] = target
    # The compose file appends "@<digest>" to the core image when this is set,
    # so what comes up is exactly what was verified, whatever the tag says now.
    run_env["MEDMCP_CORE_DIGEST"] = target_digest

    old = _inspect(replace) if replace else None
    if old is not None:
        old_image = str(cast("dict[str, Any]", old.get("Config") or {}).get("Image") or "")
        extras = env_file_entries(
            _env_of(old),
            _env_of(_inspect(old_image, image=True)) if old_image else {},
            compose_service_env_keys(compose_text),
        )
        write_env_file(ENV_FILE_PATH, extras)
        run_env["MEDMCP_ENV_FILE"] = str(ENV_FILE_PATH)
        if extras:
            log.info("carrying %d env_file variable(s) forward", len(extras))
    else:
        log.warning("container %s not found; env_file variables are not carried", replace)

    try:
        if dry_run:
            proc = compose_up(
                project=project,
                project_dir=project_dir,
                compose_file=COMPOSE_PATH,
                env=run_env,
                dry_run=True,
            )
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            return proc.returncode

        proc = compose_up(
            project=project, project_dir=project_dir, compose_file=COMPOSE_PATH, env=run_env
        )
        if proc.returncode == 0:
            result["status"] = "ok"
            result["detail"] = _tail(proc.stderr)
            write_result(RESULT_PATH, result)
            log.info("updated %s → %s", previous, target)
            return 0

        failure = _tail(proc.stderr) or f"docker compose exited {proc.returncode}"
        log.error("update to %s failed: %s", target, failure)
        run_env["MEDMCP_TAG"] = previous
        run_env["MEDMCP_CORE_DIGEST"] = previous_digest
        back = compose_up(
            project=project, project_dir=project_dir, compose_file=COMPOSE_PATH, env=run_env
        )
        if back.returncode == 0:
            result["status"] = "rolled_back"
            result["detail"] = failure
            log.info("rolled back to %s", previous)
            code = 1
        else:
            result["status"] = "failed"
            result["detail"] = f"{failure}\n\nrollback also failed: {_tail(back.stderr)}"
            log.error("rollback to %s failed: %s", previous, _tail(back.stderr))
            code = 3
        write_result(RESULT_PATH, result)
        return code
    finally:
        # The recovered env_file secrets have served their purpose: compose has
        # read them into the new container. Nothing of them stays in this one.
        with contextlib.suppress(OSError):
            ENV_FILE_PATH.unlink()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(prog="medmcp-update")
    sub = parser.add_subparsers(dest="command", required=True)
    apply_parser = sub.add_parser("apply", help="replace the running install with this image")
    apply_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved compose configuration instead of applying it",
    )
    args = parser.parse_args(argv)
    if args.command == "apply":
        return apply(dry_run=bool(args.dry_run))
    return 2


if __name__ == "__main__":
    sys.exit(main())
