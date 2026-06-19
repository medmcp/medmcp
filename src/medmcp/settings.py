"""Shared MedMCP configuration: stack discovery, user toggles, vibe config sync.

Everything here is UI-agnostic (no FastAPI dependency) so the workspace
server (``server.py``) and the ``medmcp`` CLI can both drive the
same state:

- **Stack discovery** (:func:`load_mcp_servers`) — uv tool environments with a
  ``[medmcp.stacks]`` entry point, plus manual ``[[mcp_servers]]`` entries in
  ``.vibe/config.toml``.
- **Active sets** — which stacks/workflows are enabled, persisted as JSON under
  ``.vibe/`` (all active when the file is absent).
- **Feature toggles** — provenance capture, personal workflows, tool-call
  explanations; each defaults to on.
- **Config sync** (:func:`sync_servers_to_vibe_config`) — writes the resolved
  server list and workflow skill paths into ``.vibe/config.toml`` before each
  session, because vibe-acp reads that file directly.
"""

from __future__ import annotations

import configparser
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import tomllib
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import httpx
import tomli_w

from medmcp.acp import PROJECT_ROOT, VIBE_HOME, JsonDict

log: logging.Logger = logging.getLogger(__name__)

# Ollama endpoint + model, used for the agent's auxiliary LLM calls
# (tool-call explanations, distillation prose) and the provenance manifest.
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4-medmcp")

# GPU selector (CDI device id) substituted into container-stack manifests' ${MEDMCP_GPU}.
# Default so os.path.expandvars (no ":-" support) never leaves the literal placeholder;
# "all" = every GPU, override with an index/UUID (e.g. "4") to pin. LLM_GPU captures
# the value the LLM container was created with (deploy-time) before a persisted runtime
# selection overrides MEDMCP_GPU for stacks (see load/save_gpu_selection).
os.environ.setdefault("MEDMCP_GPU", "all")
LLM_GPU: str = os.environ["MEDMCP_GPU"]
GPU_SELECTION_PATH: Path = VIBE_HOME / "gpu_selection.json"
if GPU_SELECTION_PATH.exists():
    try:
        _gpu_sel = str(
            cast("dict[str, Any]", json.loads(GPU_SELECTION_PATH.read_text())).get("gpu", "")
        ).strip()
    except (json.JSONDecodeError, OSError):
        _gpu_sel = ""
    if _gpu_sel:
        os.environ["MEDMCP_GPU"] = _gpu_sel

# Context window size used when Ollama hasn't been queried (yet) — the value
# set in Modelfile.gemma4.
DEFAULT_CONTEXT_WINDOW: int = 131_072

# Cached context window size fetched from Ollama /api/show. None = not yet
# fetched; populated lazily by fetch_context_window().
_context_window_tokens: int | None = None


async def fetch_context_window() -> int:
    """Return the active model's num_ctx by querying Ollama ``/api/show``.

    The result is cached for the process lifetime. Falls back to
    :data:`DEFAULT_CONTEXT_WINDOW` if Ollama is unreachable or the parameter
    is absent from the response.
    """
    global _context_window_tokens
    if _context_window_tokens is not None:
        return _context_window_tokens
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/show",
                json={"model": OLLAMA_MODEL},
            )
            resp.raise_for_status()
            data = cast("JsonDict", resp.json())
            params_str = str(data.get("parameters") or "")
            for line in params_str.splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "num_ctx":
                    _context_window_tokens = int(parts[1])
                    return _context_window_tokens
    except Exception:
        log.warning("could not fetch context window size from Ollama; using fallback")
    _context_window_tokens = DEFAULT_CONTEXT_WINDOW
    return _context_window_tokens


def cached_context_window() -> int:
    """Return the cached window size without I/O (fallback when unfetched)."""
    return (
        _context_window_tokens if _context_window_tokens is not None else (DEFAULT_CONTEXT_WINDOW)
    )


# Tracks which discovered stacks are enabled; defaults to all when absent.
ACTIVE_STACKS_PATH: Path = VIBE_HOME / "active_stacks.json"
# Tracks which personal workflows are enabled (loaded as skills); all when absent.
ACTIVE_WORKFLOWS_PATH: Path = VIBE_HOME / "active_workflows.json"
# Persists the provenance-capture on/off preference; defaults to on when absent.
PROVENANCE_ENABLED_PATH: Path = VIBE_HOME / "provenance_enabled.json"
# Master on/off for the personal-workflows feature; defaults to on when absent.
WORKFLOWS_ENABLED_PATH: Path = VIBE_HOME / "workflows_enabled.json"
# Persists the explain-tool-calls on/off preference; defaults to on when absent.
EXPLAIN_ENABLED_PATH: Path = VIBE_HOME / "explain_enabled.json"

# Directory of container-stack manifests (``stacks.d/<name>.toml``). Each manifest
# declares a stack launched as a container (``command = "docker"``, ``args = [...]``)
# rather than a uv-tool-installed binary — the deployment path where stacks ship as
# images. ``${VAR}`` references in ``command``/``args``/``skills_path`` are expanded
# against the environment at load time (e.g. ``${MEDMCP_WORKSPACE}`` for path parity).
STACKS_D_PATH: Path = Path(PROJECT_ROOT) / "stacks.d"


def get_uv_tool_dir() -> Path | None:
    """Return the uv tool installation directory, or ``None`` if unavailable."""
    try:
        result = subprocess.run(["uv", "tool", "dir"], capture_output=True, text=True, timeout=5)
        return Path(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:
        return None


def call_entry_point(python: Path, module: str, attr: str) -> object:
    """Call ``module.attr()`` inside *python*'s environment and return the result.

    Raises ``RuntimeError`` on non-zero exit and ``ValueError`` on JSON-decode
    failure; both are caught by the caller so broken stacks are skipped cleanly.
    """
    result = subprocess.run(
        [str(python), "-c", f"import json,{module};print(json.dumps({module}.{attr}()))"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def _load_stack_manifests() -> list[JsonDict]:
    """Read container-stack manifests from :data:`STACKS_D_PATH`.

    Each ``stacks.d/<name>.toml`` declares a stack that is launched as a container
    (``command = "docker"``, ``args = [...]``) instead of a uv-tool binary — the
    deployment path where stacks ship as images. ``${VAR}`` references in
    ``command``, ``args`` and ``skills_path`` are expanded against the environment
    (notably ``${MEDMCP_WORKSPACE}`` so the bind-mount lands at the host path the
    workspace server already uses). Malformed manifests are skipped with a warning
    so one bad file never breaks discovery.

    Returns a list of server-config dicts in the same shape as the uv-tool scan.
    """
    if not STACKS_D_PATH.is_dir():
        return []
    manifests: list[JsonDict] = []
    for path in sorted(STACKS_D_PATH.glob("*.toml")):
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except Exception as exc:
            log.warning("Could not parse stack manifest %s; skipping: %s", path, exc)
            continue
        name = str(raw.get("name", "")).strip()
        command = str(raw.get("command", "")).strip()
        if not name or not command:
            log.warning("Stack manifest %s missing 'name'/'command'; skipping", path)
            continue
        args = [os.path.expandvars(str(a)) for a in cast("list[Any]", raw.get("args", []))]
        entry: JsonDict = {
            "name": name,
            "command": os.path.expandvars(command),
            "args": args,
            "env": dict(cast("dict[str, str]", raw.get("env", {}))),
        }
        if raw.get("skills_path"):
            entry["skills_path"] = os.path.expandvars(str(raw["skills_path"]))
        if raw.get("tool_timeout_sec") is not None:
            entry["tool_timeout_sec"] = raw["tool_timeout_sec"]
        manifests.append(entry)
    return manifests


@lru_cache(maxsize=1)
def load_mcp_servers() -> list[JsonDict]:
    """Discover MCP servers from uv tool environments and ``.vibe/config.toml``.

    Server configs are collected from two sources:

    1. **Installed uv tools** (authoritative) — any package installed via
       ``uv tool install`` that registers a ``[medmcp.stacks]`` entry point in
       its dist-info is auto-discovered.  The entry point must be a zero-argument
       callable returning a dict with at least ``name`` and ``command`` keys.
       The executable is resolved to its absolute path inside the isolated tool
       env, so PATH ordering never causes the wrong binary to be picked up.

    2. **Container-stack manifests** in ``stacks.d/*.toml`` (:func:`_load_stack_manifests`)
       — stacks shipped as images and launched via ``docker run -i`` (the
       deployment path); accepted only for names not already claimed by a uv tool.

    3. **Manual ``[[mcp_servers]]`` entries in ``.vibe/config.toml``** — only
       entries whose ``name`` is *not* already registered by a uv tool or a
       manifest are accepted.  This covers stacks that have not yet been installed
       via ``just install-stack``.

    Returns a list of server-config dicts ready for ``sync_servers_to_vibe_config``.
    """
    servers: dict[str, JsonDict] = {}

    # ── 1. Scan uv tool environments ─────────────────────────────────────────
    tool_dir = get_uv_tool_dir()
    if tool_dir is not None:
        cp = configparser.ConfigParser()
        cp.optionxform = lambda optionstr: optionstr  # preserve case
        for ep_file in tool_dir.glob("*/lib/python*/site-packages/*.dist-info/entry_points.txt"):
            cp.clear()
            cp.read(ep_file)
            if not cp.has_section("medmcp.stacks"):
                continue
            # tool env root: entry_points.txt → dist-info → site-packages →
            #                python3.x → lib → <tool-env>
            tool_env = ep_file.parents[4]
            python = tool_env / "bin" / "python"
            if not python.exists():
                continue
            for ep_name, ep_value in cp.items("medmcp.stacks"):
                parts = ep_value.strip().split(":")
                if len(parts) != 2:
                    continue
                module, attr = parts
                # Reject non-identifier strings before passing to a subprocess.
                if not re.fullmatch(r"[\w.]+", module) or not re.fullmatch(r"\w+", attr):
                    log.warning("Skipping malformed entry point %r = %r", ep_name, ep_value)
                    continue
                try:
                    raw = call_entry_point(python, module, attr)
                except Exception as exc:
                    log.warning("medmcp.stacks entry point %r failed: %s", ep_name, exc)
                    continue
                if not isinstance(raw, dict) or "name" not in raw:
                    log.warning(
                        "medmcp.stacks entry point %r returned an invalid config "
                        "(expected dict with 'name' key); skipping",
                        ep_name,
                    )
                    continue
                srv = cast("JsonDict", raw)
                name = str(srv.get("name", ""))
                if not name:
                    continue
                # Resolve command to the absolute executable inside the tool env
                # so vibe-acp always uses the fully-dep-installed binary.
                # Guard: an empty command must not be resolved — Path / "" collapses
                # to the parent dir, which exists, corrupting command to a dir path.
                command = str(srv.get("command", ""))
                if command:
                    candidate = tool_env / "bin" / command
                    if candidate.exists():
                        command = str(candidate)
                dist_info = ep_file.parent.name  # e.g. "medmcp_dicom-0.1.0.dist-info"
                version = dist_info.removesuffix(".dist-info").rsplit("-", 1)[-1]
                entry: JsonDict = {
                    "name": name,
                    "command": command,
                    "args": list(cast("list[Any]", srv.get("args", []))),
                    "env": dict(cast("dict[str, str]", srv.get("env", {}))),
                    "version": version,
                }
                for key in ("skills_path", "tool_timeout_sec", "startup_timeout_sec"):
                    if srv.get(key) is not None:
                        entry[key] = srv[key]
                servers[name] = entry

    # ── 2. Container-stack manifests (stacks.d/*.toml) ───────────────────────
    # Stacks shipped as images, launched via `docker run -i`. A uv-tool install
    # of the same name wins (local dev against an installed stack overrides the
    # container manifest); manifests fill in everything else.
    for entry in _load_stack_manifests():
        if entry["name"] not in servers:
            servers[entry["name"]] = entry

    # ── 3. Manual config.toml entries ────────────────────────────────────────
    # Only accepted for names NOT already claimed by a uv tool or a manifest.
    # This prevents a feedback loop where servers written to config.toml by
    # sync_servers_to_vibe_config shadow the live tool-env definitions.
    config_path = VIBE_HOME / "config.toml"
    if config_path.exists():
        try:
            with config_path.open("rb") as f:
                cfg = tomllib.load(f)
        except Exception as exc:
            log.warning("Could not parse %s; skipping manual entries: %s", config_path, exc)
            return list(servers.values())

        tool_names = set(servers)
        for srv in cfg.get("mcp_servers", []):
            name = cast("str", srv.get("name", ""))
            if not name or name in tool_names:
                continue
            command = cast("str", srv.get("command", ""))
            # Skip stale entries written by sync_servers_to_vibe_config for
            # tools that have since been uninstalled: absolute paths that no
            # longer exist on disk indicate a removed uv tool environment.
            if command and Path(command).is_absolute() and not Path(command).exists():
                log.debug(
                    "Skipping stale config.toml entry %r (command not found: %s)", name, command
                )
                continue
            # A container-stack entry (command "docker") reaching here is an orphan:
            # a present stacks.d manifest would have claimed its name in source #2, so
            # this is a leftover written into config.toml by a prior sync before the
            # stack was uninstalled. Drop it so uninstalls actually take effect.
            if command and Path(command).name == "docker":
                log.debug("Skipping orphaned container-stack config.toml entry %r", name)
                continue
            servers[name] = {
                "name": name,
                "command": command,
                "args": cast("list[Any]", srv.get("args", [])),
                "env": {},
            }

    return list(servers.values())


def load_active_server_names() -> set[str]:
    """Return the set of server names currently marked active.

    When ``.vibe/active_stacks.json`` is absent (first run) every discovered
    server is considered active so behaviour is identical to the previous
    all-servers-always-on model.
    """
    all_names = {s["name"] for s in load_mcp_servers()}
    if not ACTIVE_STACKS_PATH.exists():
        return all_names
    try:
        data = cast("dict[str, Any]", json.loads(ACTIVE_STACKS_PATH.read_text()))
        return set(cast("list[str]", data.get("active", list(all_names))))
    except (json.JSONDecodeError, OSError):
        return all_names


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a small JSON state file atomically, safe across processes.

    A unique temp name (not a fixed ``.tmp`` path) so a concurrent writer in
    another process (a second server instance, the CLI) can never consume or
    interleave into this writer's temp file; ``os.replace`` makes the
    last-complete-writer win.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def save_active_server_names(names: set[str]) -> None:
    """Persist the active server set to ``.vibe/active_stacks.json``."""
    _atomic_write_json(ACTIVE_STACKS_PATH, {"active": sorted(names)})


def active_servers() -> list[JsonDict]:
    """Return only the active subset of all discovered servers."""
    active_names = load_active_server_names()
    return [s for s in load_mcp_servers() if s["name"] in active_names]


# ── Container-stack install / uninstall (UI-driven) ──────────────────────────
# An installed container stack is just a stacks.d/<name>.toml manifest plus its
# extracted skills. Metadata comes from the image's OCI label (read via
# `docker inspect` — the image is never executed to introspect it).

# Label carrying a stack's launch metadata. JSON value, e.g.:
#   {"name": "medmcp-neuro", "gpu": true, "tool_timeout_sec": 7200,
#    "skills_path": "/app/src/medmcp_neuro/skills"}
STACK_LABEL: str = "org.medmcp.stack"
_STACK_NAME_RE: re.Pattern[str] = re.compile(r"[a-z0-9][a-z0-9-]*")


def _run_docker(args: list[str], *, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    """Run a ``docker`` command (list args, no shell); raise RuntimeError on failure."""
    try:
        result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError("docker CLI not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"docker {args[0]} timed out") from exc
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"docker {args[0]} failed")
    return result


def read_stack_label(image: str) -> JsonDict:
    """Return the parsed ``org.medmcp.stack`` label of *image*, pulling it if absent.

    Raises:
        ValueError: invalid image ref, or malformed/incomplete label.
        FileNotFoundError: the image carries no such label (not a medmcp stack).
        RuntimeError: a docker command failed (e.g. image not found in any registry).
    """
    image = image.strip()
    if not image or any(c.isspace() for c in image):
        raise ValueError(f"invalid image reference: {image!r}")
    # Ensure the image is present locally; pull from the registry only if not.
    try:
        _run_docker(["image", "inspect", image], timeout=30)
    except RuntimeError:
        _run_docker(["pull", image])
    fmt = '{{ index .Config.Labels "' + STACK_LABEL + '" }}'
    raw = _run_docker(["inspect", "--format", fmt, image], timeout=30).stdout.strip()
    if not raw or raw == "<no value>":
        raise FileNotFoundError(f"{image} has no {STACK_LABEL} label (not a medmcp stack)")
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed {STACK_LABEL} label on {image}: {exc}") from exc
    if not isinstance(meta, dict) or not str(cast("JsonDict", meta).get("name", "")).strip():
        raise ValueError(f"{STACK_LABEL} label on {image} missing 'name'")
    return cast("JsonDict", meta)


def _extract_image_skills(image: str, in_image_path: str, into_dir: Path) -> None:
    """Copy ``<image>:<in_image_path>`` into *into_dir* via a throwaway container.

    The result is ``into_dir/<basename of in_image_path>`` (docker cp of a dir
    drops it as a subdir of the destination).
    """
    into_dir.mkdir(parents=True, exist_ok=True)
    cid = _run_docker(["create", image], timeout=60).stdout.strip()
    try:
        _run_docker(["cp", f"{cid}:{in_image_path}", str(into_dir)], timeout=120)
    finally:
        with contextlib.suppress(RuntimeError):
            _run_docker(["rm", "-f", cid], timeout=30)


# Callback for streaming install progress (one human-readable line at a time).
ProgressFn = Callable[[str], None]


def _image_present(image: str) -> bool:
    """Return whether *image* is already present locally."""
    try:
        _run_docker(["image", "inspect", image], timeout=30)
        return True
    except RuntimeError:
        return False


def _pull_streaming(image: str, on_progress: ProgressFn | None) -> None:
    """Run ``docker pull`` streaming its status lines to *on_progress*.

    Output is a non-TTY pipe, so docker emits line-oriented status updates (not
    in-place bars); carriage-return segments are collapsed to the last token.
    Raises RuntimeError on a non-zero exit.
    """
    try:
        proc = subprocess.Popen(
            ["docker", "pull", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("docker CLI not found") from exc
    assert proc.stdout is not None
    tail: list[str] = []
    for raw in proc.stdout:
        line = raw.split("\r")[-1].strip()
        if line:
            tail = [*tail[-4:], line]
            if on_progress:
                on_progress(line)
    if proc.wait() != 0:
        blob = " ".join(tail).lower()
        auth_markers = ("denied", "unauthorized", "authentication required", "forbidden")
        if any(k in blob for k in auth_markers):
            raise RuntimeError(
                f"not authorized to pull {image} — log in to the registry on the host "
                "(`docker login ghcr.io`) or set GHCR_USER/GHCR_TOKEN."
            )
        raise RuntimeError(f"docker pull {image} failed")


def _write_stack_manifest(name: str, entry: JsonDict) -> None:
    """Atomically write ``stacks.d/<name>.toml``."""
    STACKS_D_PATH.mkdir(parents=True, exist_ok=True)
    path = STACKS_D_PATH / f"{name}.toml"
    fd, tmp_name = tempfile.mkstemp(dir=STACKS_D_PATH, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(entry, fh)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def install_stack_image(image: str, on_progress: ProgressFn | None = None) -> str:
    """Install a container stack from *image* and return its name.

    Pulls the image if absent (streaming progress to *on_progress*), reads its
    ``org.medmcp.stack`` label, extracts its skills next to the manifest, writes
    ``stacks.d/<name>.toml`` (the launch recipe), and marks the stack active.
    Idempotent — re-installing overwrites. Raises as :func:`read_stack_label`
    plus ValueError for a bad name in the label.
    """

    def report(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    image = image.strip()
    if not image or any(c.isspace() for c in image):
        raise ValueError(f"invalid image reference: {image!r}")

    report(f"Checking {image}…")
    if not _image_present(image):
        report(f"Pulling {image}…")
        _pull_streaming(image, on_progress)

    report("Reading stack metadata…")
    meta = read_stack_label(image)
    name = str(meta["name"]).strip()
    if not _STACK_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid stack name in label: {name!r}")

    args: list[str] = ["run", "--rm", "-i"]
    if meta.get("gpu"):
        # ${MEDMCP_GPU} is expanded at load time (defaults to "all"), so the GPU
        # can be re-pinned via env without reinstalling the stack.
        args += ["--device", "nvidia.com/gpu=${MEDMCP_GPU}"]
    args += ["-v", "${MEDMCP_WORKSPACE}:${MEDMCP_WORKSPACE}", image]
    entry: JsonDict = {"name": name, "command": "docker", "args": args}

    timeout = meta.get("tool_timeout_sec")
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
        entry["tool_timeout_sec"] = float(timeout)

    # Skills live inside the image but the agent loads them from the core fs, so
    # extract them next to the manifest and point skills_path there.
    in_image_skills = meta.get("skills_path")
    if isinstance(in_image_skills, str) and in_image_skills.strip():
        report("Extracting skills…")
        skills_root = STACKS_D_PATH / name
        shutil.rmtree(skills_root, ignore_errors=True)  # clean prior extraction
        _extract_image_skills(image, in_image_skills.strip(), skills_root)
        extracted = skills_root / Path(in_image_skills.strip()).name
        if extracted.is_dir():
            entry["skills_path"] = str(extracted)

    report("Registering stack…")
    _write_stack_manifest(name, entry)

    # Make it active immediately. Only touch an explicit set if one exists;
    # an absent active_stacks.json already means "all discovered are active".
    if ACTIVE_STACKS_PATH.exists():
        save_active_server_names(load_active_server_names() | {name})
    return name


def uninstall_stack(name: str) -> None:
    """Remove an installed container stack's manifest and extracted skills.

    Raises ValueError for a bad name, FileNotFoundError if not installed.
    """
    if not _STACK_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid stack name: {name!r}")
    manifest = STACKS_D_PATH / f"{name}.toml"
    if not manifest.exists():
        raise FileNotFoundError(f"no installed stack named {name!r}")
    manifest.unlink()
    shutil.rmtree(STACKS_D_PATH / name, ignore_errors=True)
    if ACTIVE_STACKS_PATH.exists():
        save_active_server_names(load_active_server_names() - {name})


def list_installed_stacks() -> list[JsonDict]:
    """Return installed container stacks (from stacks.d manifests): name, image, gpu."""
    out: list[JsonDict] = []
    for m in _load_stack_manifests():
        args = cast("list[str]", m.get("args", []))
        out.append(
            {
                "name": m["name"],
                "image": args[-1] if args else "",
                "gpu": "--device" in args,
            }
        )
    return out


# Curated catalog of installable stacks shown in the UI (browse → install). Default
# is the bundled catalog.json; override with MEDMCP_CATALOG_URL (an http(s) URL or
# a file path) to point at a published or air-gapped-mirror catalog.
CATALOG_PATH: Path = Path(PROJECT_ROOT) / "catalog.json"
CATALOG_URL: str = os.environ.get("MEDMCP_CATALOG_URL", "")


def load_catalog() -> list[JsonDict]:
    """Return the curated catalog of installable stacks.

    Each entry is ``{name, image, description, gpu}``. Source is
    :data:`CATALOG_URL` (http(s) URL or file path) when set, else the bundled
    :data:`CATALOG_PATH`. Best-effort: returns ``[]`` (and logs) on any error so
    the UI degrades gracefully.
    """
    src = CATALOG_URL.strip()
    try:
        if src.startswith(("http://", "https://")):
            resp = httpx.get(src, timeout=10.0)
            resp.raise_for_status()
            data: object = resp.json()
        else:
            path = Path(src) if src else CATALOG_PATH
            if not path.is_file():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("could not load stack catalog from %r: %s", src or str(CATALOG_PATH), exc)
        return []
    if not isinstance(data, dict):
        return []
    raw = cast("list[Any]", cast("JsonDict", data).get("stacks", []))
    out: list[JsonDict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        e = cast("JsonDict", entry)
        name = str(e.get("name", "")).strip()
        image = str(e.get("image", "")).strip()
        if not name or not image:
            continue
        out.append(
            {
                "name": name,
                "image": image,
                "description": str(e.get("description", "")),
                "gpu": bool(e.get("gpu", False)),
            }
        )
    return out


def read_skill_description(skill_md: Path) -> str:
    """Return the ``description:`` frontmatter value from a SKILL.md, or ``''``."""
    try:
        lines = skill_md.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines[:15]:
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


def discover_workflows() -> list[JsonDict]:
    """Discover personal workflows from ``draft/`` and ``active/``.

    Returns ``{name, description, kind}`` dicts, deduplicated by name (an active
    workflow shadows a draft of the same name). ``kind`` is ``"active"`` for
    promoted workflows and ``"draft"`` for unpromoted ones.
    """
    found: dict[str, JsonDict] = {}
    for kind in ("active", "draft"):
        base = VIBE_HOME / "workflows" / kind
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            skill = d / "SKILL.md"
            if not d.is_dir() or not skill.is_file() or d.name in found:
                continue
            found[d.name] = {
                "name": d.name,
                "description": read_skill_description(skill),
                "kind": kind,
            }
    return list(found.values())


def load_active_workflow_names() -> set[str]:
    """Return the set of workflow names currently enabled (loaded as skills).

    When ``.vibe/active_workflows.json`` is absent every discovered workflow is
    active, so a freshly distilled/promoted workflow is on by default.
    """
    all_names = {w["name"] for w in discover_workflows()}
    if not ACTIVE_WORKFLOWS_PATH.exists():
        return all_names
    try:
        data = cast("dict[str, Any]", json.loads(ACTIVE_WORKFLOWS_PATH.read_text()))
        return set(cast("list[str]", data.get("active", list(all_names))))
    except (json.JSONDecodeError, OSError):
        return all_names


def save_active_workflow_names(names: set[str]) -> None:
    """Persist the active workflow set to ``.vibe/active_workflows.json``."""
    _atomic_write_json(ACTIVE_WORKFLOWS_PATH, {"active": sorted(names)})


def _load_flag(path: Path) -> bool:
    """Read an ``{"enabled": bool}`` preference file (default ``True`` when unset)."""
    if not path.exists():
        return True
    try:
        data = cast("dict[str, Any]", json.loads(path.read_text()))
        return bool(data.get("enabled", True))
    except (json.JSONDecodeError, OSError):
        return True


def _save_flag(path: Path, enabled: bool) -> None:
    """Persist an ``{"enabled": bool}`` preference file atomically."""
    _atomic_write_json(path, {"enabled": enabled})


def load_provenance_enabled() -> bool:
    """Return whether provenance capture is enabled (default ``True`` when unset)."""
    return _load_flag(PROVENANCE_ENABLED_PATH)


def save_provenance_enabled(enabled: bool) -> None:
    """Persist the provenance-capture on/off preference to disk."""
    _save_flag(PROVENANCE_ENABLED_PATH, enabled)


def load_workflows_enabled() -> bool:
    """Return whether the personal-workflows feature is enabled (default ``True``).

    The master switch: when off, no personal workflow is loaded as a skill and
    the UIs hide their workflow controls (see ``sync_servers_to_vibe_config``).
    """
    return _load_flag(WORKFLOWS_ENABLED_PATH)


def save_workflows_enabled(enabled: bool) -> None:
    """Persist the personal-workflows master on/off preference to disk."""
    _save_flag(WORKFLOWS_ENABLED_PATH, enabled)


def load_explain_enabled() -> bool:
    """Return whether tool-call explanations are enabled (default ``True``)."""
    return _load_flag(EXPLAIN_ENABLED_PATH)


def save_explain_enabled(enabled: bool) -> None:
    """Persist the explain-tool-calls on/off preference to disk."""
    _save_flag(EXPLAIN_ENABLED_PATH, enabled)


def load_gpu_selection() -> str:
    """Effective GPU (CDI device id) for container stacks (persisted, else boot default)."""
    return os.environ.get("MEDMCP_GPU", "all")


def save_gpu_selection(gpu: str) -> None:
    """Persist the stack GPU selection and apply it to this process.

    Clears the discovery cache so the next :func:`load_mcp_servers` re-expands the
    ``${MEDMCP_GPU}`` ``--device`` arg; the caller should re-sync vibe-acp config and
    restart it so newly spawned stacks pick up the device. Does not move the LLM —
    its GPU is fixed at container creation (see :data:`LLM_GPU`).
    """
    value = gpu.strip() or "all"
    _atomic_write_json(GPU_SELECTION_PATH, {"gpu": value})
    os.environ["MEDMCP_GPU"] = value
    load_mcp_servers.cache_clear()


def _llm_image() -> str:
    """Image of the running LLM container (a present CUDA image), or "" if unknown."""
    override = os.environ.get("OLLAMA_IMAGE", "").strip()
    if override:
        return override
    try:
        ps = _run_docker(
            [
                "ps",
                "--filter",
                "label=com.docker.compose.service=llm",
                "--format",
                "{{.Image}}",
            ],
            timeout=10,
        )
    except RuntimeError:
        return ""
    return next((n.strip() for n in ps.stdout.splitlines() if n.strip()), "")


def list_gpus() -> list[JsonDict]:
    """Best-effort list of GPUs as ``{index, uuid, name}`` for the settings picker.

    The CPU-only core can't enumerate GPUs itself, so it runs ``nvidia-smi`` in a
    throwaway container with *every* GPU exposed (``--device nvidia.com/gpu=all``) so
    the real host indices/UUIDs come back — querying the (possibly pinned) LLM
    container would only show its own GPU(s), renumbered. Reuses the LLM image (a
    present CUDA image). Returns ``[]`` when not enumerable — the UI falls back to
    free text.
    """
    image = _llm_image()
    if not image:
        return []
    try:
        smi = _run_docker(
            [
                "run",
                "--rm",
                "--device",
                "nvidia.com/gpu=all",
                "--entrypoint",
                "nvidia-smi",
                image,
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader",
            ],
            timeout=30,
        )
    except RuntimeError:
        return []
    gpus: list[JsonDict] = []
    for line in smi.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            gpus.append({"index": parts[0], "uuid": parts[1], "name": parts[2]})
    return gpus


# Serializes the config.toml read-modify-write within this process: the
# workspace server runs the sync from a worker thread per websocket connect,
# and several connects race after a settings-triggered restart.
_config_write_lock = threading.Lock()


def sync_servers_to_vibe_config(servers: list[JsonDict]) -> None:
    """Overwrite the ``[[mcp_servers]]`` section of ``.vibe/config.toml``.

    This is the mechanism that makes entry-point discovery actually take effect:
    vibe-acp reads its server list directly from the TOML file and ignores the
    ``mcpServers`` field in JSON-RPC calls, so we must write the desired set
    here before each ``session/new`` or ``session/load``.

    Only the ``mcp_servers`` array is replaced.  All other keys (model config,
    tool permissions, etc.) are left untouched.  For servers already present in
    the file, vibe-acp-specific fields such as ``startup_timeout_sec`` are
    preserved; only ``command`` and ``args`` are updated from the discovery
    result.
    """
    with _config_write_lock:
        _sync_servers_to_vibe_config_locked(servers)


def _sync_servers_to_vibe_config_locked(servers: list[JsonDict]) -> None:
    """Body of :func:`sync_servers_to_vibe_config`; caller holds the lock."""
    config_path = VIBE_HOME / "config.toml"
    cfg: dict[str, Any] = {}
    existing_by_name: dict[str, JsonDict] = {}

    if config_path.exists():
        try:
            with config_path.open("rb") as fh:
                cfg = tomllib.load(fh)
        except Exception as exc:
            log.warning("Could not parse %s; skipping mcp_servers sync: %s", config_path, exc)
            return
        for raw in cast("list[JsonDict]", cfg.get("mcp_servers", [])):
            name = cast("str", raw.get("name", ""))
            if name:
                existing_by_name[name] = raw

    new_entries: list[JsonDict] = []
    for srv in servers:
        name = srv["name"]
        if name in existing_by_name:
            # Preserve vibe-acp-specific fields (timeouts, transport, etc.)
            # and overwrite discovery-owned fields (command, args).
            entry = dict(existing_by_name[name])
            entry["command"] = srv["command"]
            if srv.get("args"):
                entry["args"] = srv["args"]
            else:
                entry.pop("args", None)
            new_entries.append(entry)
        else:
            # Brand-new server from an entry point. Copy vibe-acp fields that
            # packages may supply (e.g. tool_timeout_sec for long-running tools),
            # then ensure required fields are set.
            passthrough = {
                "tool_timeout_sec",
                "startup_timeout_sec",
                "transport",
                "env",
                "skills_path",
            }
            entry = {"transport": "stdio"}
            for key in passthrough:
                if key in srv:
                    entry[key] = srv[key]
            entry["name"] = name
            entry["command"] = srv["command"]
            if srv.get("args"):
                entry["args"] = srv["args"]
            else:
                entry.pop("args", None)
            new_entries.append(entry)

    cfg["mcp_servers"] = new_entries

    # Collect skills_path values from discovered servers and write them to
    # skill_paths so vibe-acp loads the bundled skill docs automatically.
    skill_paths = [srv["skills_path"] for srv in servers if srv.get("skills_path")]
    # The personal-workflows feature can be turned off entirely by the master
    # toggle; when off, no workflow dir is added to skill_paths and every workflow
    # is listed in disabled_skills, so nothing personal is loaded.
    workflows_enabled = load_workflows_enabled()
    if workflows_enabled:
        # Promoted, reusable workflows live here as <name>/SKILL.md; include the
        # directory so distilled workflows are discoverable as skills (Tier 3).
        workflows_active = VIBE_HOME / "workflows" / "active"
        if workflows_active.is_dir():
            skill_paths.append(str(workflows_active))
        # Draft workflows are loaded too, so a draft can be tested (invoked as
        # `/<name>`) before it is promoted. Promotion just moves the draft into
        # active/ to keep it permanently; both dirs hold <name>/SKILL.md entries.
        workflows_draft = VIBE_HOME / "workflows" / "draft"
        if workflows_draft.is_dir():
            skill_paths.append(str(workflows_draft))
    cfg["skill_paths"] = skill_paths

    # Personal workflows toggled off in the gear are disabled by name so vibe-acp
    # skips loading them, without removing them from disk. With the master toggle
    # off, all of them are disabled. Non-workflow entries in disabled_skills (if
    # any were set manually) are preserved.
    all_workflows = {w["name"] for w in discover_workflows()}
    deactivated = (
        all_workflows if not workflows_enabled else (all_workflows - load_active_workflow_names())
    )
    existing_disabled = cast("list[str]", cfg.get("disabled_skills", []))
    preserved = [s for s in existing_disabled if s not in all_workflows]
    cfg["disabled_skills"] = sorted(set(preserved) | deactivated)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    # A unique temp name (not a fixed .tmp path) so a concurrent writer in
    # another process can never interleave into the same file; os.replace
    # then makes last-complete-writer win.
    fd, tmp_name = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(cfg, fh)
        os.replace(tmp_name, config_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
