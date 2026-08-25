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
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from string import Formatter
from typing import Any, cast
from urllib.parse import urlparse

import httpx
import tomli_w

from medmcp.acp import PROJECT_ROOT, VIBE_HOME, JsonDict

log: logging.Logger = logging.getLogger(__name__)

# Ollama endpoint + model, used for the agent's auxiliary LLM calls
# (tool-call explanations, distillation prose) and the provenance manifest.
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "muse-medmcp")

# GPU selector (CDI device id) substituted into container-stack manifests' ${MEDMCP_GPU}.
# Default so os.path.expandvars (no ":-" support) never leaves the literal placeholder;
# "all" = every GPU, override with an index/UUID (e.g. "0") to pin. LLM_GPU captures
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
# set in Modelfile.muse.
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


def fetched_context_window() -> int | None:
    """Return the Ollama-fetched window size, or ``None`` when not yet fetched (no I/O).

    Unlike :func:`cached_context_window` this does not substitute the static
    default, so callers with a better fallback (e.g. the size vibe reports on a
    usage frame) can tell "known" from "guessed".
    """
    return _context_window_tokens


# Tracks which discovered stacks are enabled; defaults to all when absent.
ACTIVE_STACKS_PATH: Path = VIBE_HOME / "active_stacks.json"
# Persists the provenance-capture on/off preference; defaults to on when absent.
PROVENANCE_ENABLED_PATH: Path = VIBE_HOME / "provenance_enabled.json"
# Persists the explain-tool-calls on/off preference; defaults to on when absent.
EXPLAIN_ENABLED_PATH: Path = VIBE_HOME / "explain_enabled.json"

# Directory of container-stack manifests (``stacks.d/<name>.toml``). Each manifest
# declares a stack launched as a container (``command = "docker"``, ``args = [...]``)
# rather than a uv-tool-installed binary — the deployment path where stacks ship as
# images. ``${VAR}`` references in ``command``/``args``/``skills_path`` are expanded
# against the environment at load time (e.g. ``${MEDMCP_WORKSPACE}`` for path parity).
STACKS_D_PATH: Path = Path(PROJECT_ROOT) / "stacks.d"

# ── Persistent stack pool / pre-warm proxy (Layer 1) ─────────────────────────
# When MEDMCP_STACK_POOL is enabled, vibe's [[mcp_servers]] are rewritten to spawn
# the lightweight `medmcp-mcp-proxy <stack>` shim, which forwards tool calls to a
# persistent BackendPool over a broker socket (so the spawn/import/CUDA cost is
# paid once, not per call). The real launch specs go to backends.json for the pool
# and the proxy's direct-spawn fallback. Off by default — ships dark.
PROXY_COMMAND: str = "medmcp-mcp-proxy"
DEFAULT_IDLE_TTL_SEC: float = 120.0
DEFAULT_STARTUP_TIMEOUT_SEC: float = 60.0
DEFAULT_TOOL_TIMEOUT_SEC: float = 900.0
# Startup budget for a stack launched as a container. vibe's own default is 10s,
# which a stack image cold-starts past on the first run after a pull: the launch
# has to read the image's Python env off disk before the server answers, and that
# competes with whatever else is warming at the time (the model being loaded into
# VRAM, most of all). Warm, these servers answer in under two seconds; the only
# thing a generous budget costs is how long a genuinely hung server delays
# warm-up, and an image that is missing or unrunnable still fails immediately
# rather than waiting this out. Discovery runs once per agent start and a miss
# silently drops the whole stack, so buy the headroom.
DEFAULT_STACK_STARTUP_TIMEOUT_SEC: float = 120.0
# Env keys the proxy layer injects into a stack's [[mcp_servers]] entry — stripped
# again when the pool is disabled so toggling off leaves no stale routing.
_POOL_ENV_KEYS: tuple[str, ...] = ("MEDMCP_BROKER_SOCK", "MEDMCP_WORKSPACE")


def stack_pool_enabled() -> bool:
    """Whether the persistent stack pool / pre-warm proxy is enabled (default off)."""
    return os.environ.get("MEDMCP_STACK_POOL", "").strip().lower() in {"1", "true", "yes", "on"}


def backend_socket_path() -> Path:
    """Unix socket the broker binds and the proxy connects to."""
    return VIBE_HOME / "backend.sock"


def backends_registry_path() -> Path:
    """Registry of real stack launch specs (pool + proxy direct-spawn fallback)."""
    return VIBE_HOME / "backends.json"


def build_backend_registry(servers: list[JsonDict]) -> dict[str, JsonDict]:
    """Build the name→launch-spec registry the pool and proxy fallback read.

    Captures each stack's *real* command/args/env (the pool spawns these, not vibe)
    plus the pool-policy fields. ``gpu`` is taken from an explicit flag or inferred
    from a CDI ``--device nvidia.com/gpu=…`` arg (host-native stacks without the
    flag are treated as non-GPU — a known gap for the LRU cap in local dev).
    """
    registry: dict[str, JsonDict] = {}
    for srv in servers:
        name = str(srv["name"])
        args = [str(a) for a in cast("list[Any]", srv.get("args", []))]
        raw_env = srv.get("env")
        env = (
            {str(k): str(v) for k, v in cast("JsonDict", raw_env).items()}
            if isinstance(raw_env, dict)
            else {}
        )
        registry[name] = {
            "command": str(srv["command"]),
            "args": args,
            "env": env,
            "gpu": bool(srv.get("gpu")) or any("nvidia.com/gpu" in a for a in args),
            "idle_ttl_sec": float(srv.get("idle_ttl_sec") or DEFAULT_IDLE_TTL_SEC),
            "startup_timeout_sec": float(
                srv.get("startup_timeout_sec") or DEFAULT_STARTUP_TIMEOUT_SEC
            ),
            "tool_timeout_sec": float(srv.get("tool_timeout_sec") or DEFAULT_TOOL_TIMEOUT_SEC),
        }
    return registry


def _write_backend_registry(servers: list[JsonDict]) -> None:
    """Atomically write backends.json from the real (un-proxied) server specs."""
    _atomic_write_json(backends_registry_path(), build_backend_registry(servers))


def _proxied_entry(entry: JsonDict, ws_root: str | None) -> JsonDict:
    """Rewrite a synced ``[[mcp_servers]]`` entry to launch through the proxy shim.

    vibe spawns ``medmcp-mcp-proxy <stack>`` instead of the real server; discovery
    fields (``skills_path``, ``tool_timeout_sec``, ``transport``, ``name``) are kept
    so vibe still loads skills and waits long enough for a possibly-cold backend.
    The stack's real ``env`` lives in backends.json, not here.
    """
    out = dict(entry)
    out["command"] = shutil.which(PROXY_COMMAND) or PROXY_COMMAND
    out["args"] = [str(entry["name"])]
    proxied_env: dict[str, str] = {"MEDMCP_BROKER_SOCK": str(backend_socket_path())}
    if ws_root:
        proxied_env["MEDMCP_WORKSPACE"] = ws_root
    out["env"] = proxied_env
    return out


def _strip_pool_env(entries: list[JsonDict]) -> None:
    """Drop pool-injected env keys preserved from a prior proxied config."""
    for entry in entries:
        env = entry.get("env")
        if isinstance(env, dict):
            for key in _POOL_ENV_KEYS:
                cast("JsonDict", env).pop(key, None)


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


def _stack_pids_limit() -> int:
    """Return the per-stack task limit, scaled to this host's CPU count.

    ``pids.max`` counts *threads*, not just processes, and the imaging stacks are
    deliberately parallel: nnUNet-style inference forks a pool of preprocessing
    workers, and each worker then opens an OpenMP/BLAS pool sized to the core
    count. The peak is therefore roughly quadratic in cores, so any fixed number
    is wrong somewhere — 512 was comfortable on a laptop and throttled a 20-core
    host, where a single LST-AI run peaked at 629 tasks and died with
    ``pthread_create() is 11`` (EAGAIN). nnUNet reports that as "Background
    workers died ... your RAM was full", which sends you looking at memory.

    256 per core keeps ~6x headroom over that measured peak while still bounding
    a runaway fork loop far below the kernel's ``pid_max``; the floor covers hosts
    that report very few cores (containers with a small ``cpuset``) where the
    stack can still be asked to process a full-resolution volume.
    """
    return max(4096, (os.cpu_count() or 8) * 256)


# Flags whose *value* is a floor rather than a fixed setting: an existing lower
# value is raised, an existing higher one is left alone. Everything else in
# _STACK_RUN_HARDENING is presence-checked only, so a deliberate opt-out (a stack
# that sets "network": true and installs with `--network bridge`) survives.
_STACK_RUN_MINIMUMS: tuple[str, ...] = ("--pids-limit",)


def _stack_run_hardening() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Container-isolation flags applied to every container-stack launch.

    Stacks bake their weights at build time and are offline at runtime (verified:
    all shipped stacks initialise and register their tools under ``--network
    none``), so egress is denied by default — the safety model assumes the local
    model can be steered by prompt injection, and a tool call must not become a
    data-exfiltration path.

    The MCP server inside a stack is a plain Python process over stdio: it needs no
    capabilities and no privilege escalation. DAC_OVERRIDE is added back
    deliberately. The workspace is bind-mounted from the host, where its files are
    owned by the invoking user, while the stack runs as root inside the container.
    Root normally bypasses the permission check via CAP_DAC_OVERRIDE; dropping it
    makes every tool fail to write its results into a host-owned directory
    ("cannot open output file ..."), which drops all of ALL's other capabilities
    while keeping the stack functional. The proper fix is to run stacks as the
    invoking uid, which removes the need for this entirely — see the non-root work
    tracked separately.
    """
    return (
        ("--network", ("--network", "none")),
        ("--cap-drop", ("--cap-drop", "ALL")),
        ("--cap-add", ("--cap-add", "DAC_OVERRIDE")),
        ("--security-opt", ("--security-opt", "no-new-privileges")),
        ("--pids-limit", ("--pids-limit", str(_stack_pids_limit()))),
    )


def _harden_stack_run_args(args: list[str]) -> list[str]:
    """Insert container-isolation flags into a ``docker run`` argument list.

    Idempotent — a flag already present (written by a newer install, or by a stack
    that opted into egress) is left untouched, except for the value floors in
    :data:`_STACK_RUN_MINIMUMS`, which are raised in place. Flags are inserted
    directly after ``run`` so they precede the image reference and anything after
    it.

    This runs at manifest *load* time as well as install time, so stacks installed
    before this existed are hardened without requiring a reinstall. ``stacks.d``
    manifests are the launch recipe and are written once at install; without the
    load-time pass, an upgrade would silently leave existing stacks unisolated —
    and, for the floors, would pin them to whatever value was correct on the day
    they were installed.
    """
    if not args or args[0] != "run":
        return args
    args = list(args)
    missing: list[str] = []
    for flag, addition in _stack_run_hardening():
        if flag not in args:
            missing += addition
            continue
        if flag not in _STACK_RUN_MINIMUMS:
            continue
        # Raise a stale or hand-lowered value to the current floor. A malformed
        # value is left alone: docker will reject it with a clearer message than
        # anything guessed here.
        index = args.index(flag) + 1
        if index < len(args):
            with contextlib.suppress(ValueError):
                args[index] = str(max(int(args[index]), int(addition[1])))
    return [args[0], *missing, *args[1:]] if missing else args


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
        expanded_command = os.path.expandvars(command)
        if Path(expanded_command).name == "docker":
            args = _harden_stack_run_args(args)
        entry: JsonDict = {
            "name": name,
            "command": expanded_command,
            "args": args,
            "env": dict(cast("dict[str, str]", raw.get("env", {}))),
        }
        if raw.get("skills_path"):
            entry["skills_path"] = os.path.expandvars(str(raw["skills_path"]))
        if raw.get("tool_timeout_sec") is not None:
            entry["tool_timeout_sec"] = raw["tool_timeout_sec"]
        if raw.get("startup_timeout_sec") is not None:
            entry["startup_timeout_sec"] = raw["startup_timeout_sec"]
        if raw.get("idle_ttl_sec") is not None:
            entry["idle_ttl_sec"] = raw["idle_ttl_sec"]
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
                for key in (
                    "skills_path",
                    "tool_timeout_sec",
                    "startup_timeout_sec",
                    "idle_ttl_sec",
                    "gpu",
                ):
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
            # An HTTP entry here is one this sync wrote for an external server —
            # source #4 owns those. Re-adopting it would strip it back to a stdio
            # entry with an empty command, which is the feedback loop above in a
            # different costume, and would survive the feature being switched off.
            if str(srv.get("transport", "")) in EXTERNAL_MCP_TRANSPORTS or srv.get("url"):
                log.debug("Skipping external-server config.toml entry %r", name)
                continue
            servers[name] = {
                "name": name,
                "command": command,
                "args": cast("list[Any]", srv.get("args", [])),
                "env": {},
            }

    # ── 4. External MCP servers (advanced; gated on an explicit acknowledgement)
    # Last, and non-shadowing: a local stack of the same name always wins, so a
    # remote entry can never displace an audited on-premise tool set.
    for entry in external_servers():
        if entry["name"] not in servers:
            servers[entry["name"]] = entry

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


# ── External MCP servers (advanced, off by default) ──────────────────────────
# Wiring the workspace to a third-party MCP service crosses the on-premise
# boundary the rest of the product guarantees, so this is gated twice: the
# operator must enable the feature *and* have acknowledged what it means. Nothing
# is discovered until both hold, which is why `external_mcp_enabled()` — not the
# stored `enabled` flag — is what discovery consults.
#
# Remote HTTP transports only. A stdio entry here would mean launching an
# arbitrary binary on the host, which is a categorically larger hole than a
# network call the permission flow already gates, so it is not offered.
#
# Credentials are stored as the *name* of an environment variable, never a token:
# `.vibe/config.toml` is rewritten on every sync and readable by anything in the
# container, so it is the wrong place for a secret.
EXTERNAL_MCP_PATH: Path = VIBE_HOME / "external_mcp.json"
EXTERNAL_MCP_TRANSPORTS: tuple[str, ...] = ("streamable-http", "http")
_ENV_VAR_RE: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# RFC 7230 token characters — the same set vibe validates header names against.
_HEADER_NAME_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9!#$%&'*+.^_`|~-]+")


def _default_external_state() -> JsonDict:
    return {"enabled": False, "acknowledged_at": None, "servers": []}


def load_external_mcp() -> JsonDict:
    """Return the external-MCP state (``enabled``/``acknowledged_at``/``servers``)."""
    if not EXTERNAL_MCP_PATH.exists():
        return _default_external_state()
    try:
        data = cast("JsonDict", json.loads(EXTERNAL_MCP_PATH.read_text()))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read %s; treating as disabled: %s", EXTERNAL_MCP_PATH, exc)
        return _default_external_state()
    state = _default_external_state()
    state["enabled"] = bool(data.get("enabled"))
    ack = data.get("acknowledged_at")
    state["acknowledged_at"] = str(ack) if ack else None
    servers = data.get("servers")
    state["servers"] = list(cast("list[JsonDict]", servers)) if isinstance(servers, list) else []
    return state


def _save_external_mcp(state: JsonDict) -> None:
    _atomic_write_json(EXTERNAL_MCP_PATH, state)


def external_mcp_acknowledged() -> bool:
    """Whether the operator has accepted responsibility for external services."""
    return bool(load_external_mcp()["acknowledged_at"])


def external_mcp_enabled() -> bool:
    """Whether external MCP servers may be discovered at all.

    Both the toggle and the acknowledgement are required — a state file that
    somehow carries ``enabled`` without one is treated as off rather than trusted.
    """
    state = load_external_mcp()
    return bool(state["enabled"]) and bool(state["acknowledged_at"])


def acknowledge_external_mcp() -> JsonDict:
    """Record that the operator accepted the risks.

    Idempotent within one activation (re-acknowledging keeps the original
    timestamp). :func:`set_external_mcp_enabled` clears it again on disable, so
    an acknowledgement covers the period it was given for and no longer.
    """
    state = load_external_mcp()
    if not state["acknowledged_at"]:
        state["acknowledged_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        _save_external_mcp(state)
    return state


def set_external_mcp_enabled(enabled: bool) -> JsonDict:
    """Turn the feature on or off; enabling requires a *current* acknowledgement.

    Disabling clears the acknowledgement, so switching the feature back on has to
    pass through the consent dialog again. A once-per-machine consent would mean
    the one control that ends the on-premise guarantee could be re-armed in
    silence — months later, or by someone who never saw what it says — and the
    person turning it on would get no warning at the moment it matters.
    """
    state = load_external_mcp()
    if enabled and not state["acknowledged_at"]:
        raise ValueError("external MCP must be acknowledged before it can be enabled")
    state["enabled"] = bool(enabled)
    if not enabled:
        state["acknowledged_at"] = None
    _save_external_mcp(state)
    return state


def _validate_api_key_format(fmt: str) -> None:
    """Raise ``ValueError`` unless *fmt* is a format string using only ``{token}``.

    Mirrors vibe's own check. Enumerating the replacement fields beats searching
    for the ``{token}`` substring: ``{{token}}`` contains it but escapes to a
    literal, and ``{token:>5}`` is valid without containing it.
    """
    try:
        fields = [name for _, name, _, _ in Formatter().parse(fmt) if name is not None]
    except ValueError as exc:
        raise ValueError(f"invalid token format: {fmt!r} (not a valid format string)") from exc
    if any(name != "token" for name in fields):
        raise ValueError(f"invalid token format: {fmt!r} (may only reference {{token}})")
    if "token" not in fields:
        raise ValueError(f"invalid token format: {fmt!r} (must contain {{token}})")


def _validate_external_server(
    name: str,
    transport: str,
    url: str,
    api_key_env: str,
    api_key_header: str = "",
    api_key_format: str = "",
) -> None:
    """Raise ``ValueError`` if any field of a proposed external server is unusable."""
    if not _STACK_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid server name: {name!r} (lowercase letters, digits and '-')")
    if transport not in EXTERNAL_MCP_TRANSPORTS:
        allowed = ", ".join(EXTERNAL_MCP_TRANSPORTS)
        raise ValueError(f"invalid transport: {transport!r} (allowed: {allowed})")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"invalid url: {url!r} (must be an http(s) URL)")
    if api_key_env and not _ENV_VAR_RE.fullmatch(api_key_env):
        raise ValueError(f"invalid environment variable name: {api_key_env!r}")
    if api_key_header and not _HEADER_NAME_RE.fullmatch(api_key_header):
        raise ValueError(f"invalid header name: {api_key_header!r}")
    if api_key_format:
        _validate_api_key_format(api_key_format)
    # vibe ignores the header/format fields when no token is configured and warns
    # about it; refuse up front rather than storing settings that do nothing.
    if (api_key_header or api_key_format) and not api_key_env:
        raise ValueError("a header or token format needs an environment variable to send")


def add_external_server(
    name: str,
    transport: str,
    url: str,
    api_key_env: str = "",
    active: bool = True,
    api_key_header: str = "",
    api_key_format: str = "",
) -> JsonDict:
    """Register an external MCP server and return the stored entry.

    The name must not collide with a discovered stack: a duplicate would shadow a
    local, audited tool set with a remote one of the operator's choosing.

    ``api_key_header``/``api_key_format`` are optional overrides for services that
    do not take a bearer token (``X-API-Key: <token>``, say); left empty, vibe's
    ``Authorization: Bearer {token}`` default applies.
    """
    name, transport, url = name.strip(), transport.strip(), url.strip()
    api_key_env = api_key_env.strip()
    api_key_header, api_key_format = api_key_header.strip(), api_key_format.strip()
    _validate_external_server(name, transport, url, api_key_env, api_key_header, api_key_format)

    state = load_external_mcp()
    if any(str(s.get("name")) == name for s in cast("list[JsonDict]", state["servers"])):
        raise ValueError(f"an external server named {name!r} already exists")
    local = {s["name"] for s in load_mcp_servers() if not s.get("external")}
    if name in local:
        raise ValueError(f"{name!r} is already the name of an installed stack")

    entry: JsonDict = {
        "name": name,
        "transport": transport,
        "url": url,
        "api_key_env": api_key_env,
        "api_key_header": api_key_header,
        "api_key_format": api_key_format,
        "active": bool(active),
    }
    cast("list[JsonDict]", state["servers"]).append(entry)
    _save_external_mcp(state)

    # A server the operator just added is meant to be on. Once written, the active
    # set is an explicit list rather than "everything discovered", so without this
    # the new entry would be discovered and then silently left out of the sync.
    if active:
        load_mcp_servers.cache_clear()
        save_active_server_names(load_active_server_names() | {name})
    return entry


def remove_external_server(name: str) -> None:
    """Delete an external server by name."""
    state = load_external_mcp()
    servers = cast("list[JsonDict]", state["servers"])
    remaining = [s for s in servers if str(s.get("name")) != name]
    if len(remaining) == len(servers):
        raise FileNotFoundError(f"no external server named {name!r}")
    state["servers"] = remaining
    _save_external_mcp(state)


def set_external_server_active(name: str, active: bool) -> None:
    """Enable or disable a single external server without deleting it."""
    state = load_external_mcp()
    for srv in cast("list[JsonDict]", state["servers"]):
        if str(srv.get("name")) == name:
            srv["active"] = bool(active)
            _save_external_mcp(state)
            return
    raise FileNotFoundError(f"no external server named {name!r}")


def external_servers() -> list[JsonDict]:
    """Return active external servers as server-config dicts, or [] when gated off."""
    if not external_mcp_enabled():
        return []
    out: list[JsonDict] = []
    for srv in cast("list[JsonDict]", load_external_mcp()["servers"]):
        if not srv.get("active"):
            continue
        name, url = str(srv.get("name", "")), str(srv.get("url", ""))
        transport = str(srv.get("transport", ""))
        env_var = str(srv.get("api_key_env") or "")
        header = str(srv.get("api_key_header") or "")
        fmt = str(srv.get("api_key_format") or "")
        try:
            _validate_external_server(name, transport, url, env_var, header, fmt)
        except ValueError as exc:
            log.warning("skipping malformed external server %r: %s", name, exc)
            continue
        entry: JsonDict = {
            "name": name,
            "transport": transport,
            "url": url,
            "external": True,
        }
        if env_var:
            # Shape fixed by vibe's MCPAuth: a discriminated union on ``type``
            # whose members forbid extra keys, so a missing discriminator or a
            # misspelled field is a hard validation error, not a silent no-op —
            # the server would simply fail to load. ``test_generated_entry_*``
            # validates what we write here against vibe's own model.
            #
            # vibe reads the token from the environment itself (defaulting to an
            # ``Authorization: Bearer {token}`` header); the variable's name is
            # all that is ever written to disk.
            auth: JsonDict = {"type": "static", "api_key_env": env_var}
            # Written only when overridden, so a server on the default scheme
            # produces the same config it did before these fields existed.
            if header:
                auth["api_key_header"] = header
            if fmt:
                auth["api_key_format"] = fmt
            entry["auth"] = auth
        out.append(entry)
    return out


# The base prompt tells the agent it runs entirely on-premise and must never
# suggest sending data to external services. With an external server wired up
# that is no longer true, and an agent holding tools it has been told never to
# use behaves erratically — it refuses them, or narrates the contradiction.
#
# vibe resolves `system_prompt_id` to exactly one file with no append hook, so
# the enabled state needs its own prompt. It is *derived* from the base rather
# than checked in beside it: two hand-maintained copies of a 70-line prompt drift,
# and the drift is silent. The anchor below is asserted by a test, so editing it
# out of the base prompt fails CI instead of quietly producing an identical file.
BASE_SYSTEM_PROMPT_ID: str = "medmcp"
EXTERNAL_SYSTEM_PROMPT_ID: str = "medmcp-external"
ONPREM_RULE: str = "- You run entirely on-premise. Never suggest sending data to external services."
EXTERNAL_RULE: str = (
    "- Your tool stacks run on-premise. This workspace also has external MCP servers "
    "configured; the operator enabled them and accepted responsibility for what they "
    "receive, so use their tools when they fit the task."
)


def write_external_prompt_variant() -> bool:
    """Derive the external-enabled system prompt next to the base one.

    Returns ``False`` — leaving the base prompt in force — when the source is
    missing or no longer carries the on-premise rule, so a prompt edit can never
    silently hand the agent a variant identical to the one it was meant to relax.
    """
    prompts = VIBE_HOME / "prompts"
    base = prompts / f"{BASE_SYSTEM_PROMPT_ID}.md"
    try:
        text = base.read_text()
    except OSError as exc:
        log.warning("cannot read %s; keeping the base system prompt: %s", base, exc)
        return False
    if ONPREM_RULE not in text:
        log.warning(
            "%s no longer contains the on-premise rule; keeping the base system prompt", base
        )
        return False
    try:
        (prompts / f"{EXTERNAL_SYSTEM_PROMPT_ID}.md").write_text(
            text.replace(ONPREM_RULE, EXTERNAL_RULE)
        )
    except OSError as exc:
        log.warning("could not write the external system prompt: %s", exc)
        return False
    return True


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


# Docker reports "amd64"/"arm64"; uname reports "x86_64"/"aarch64". Normalise both
# so a host can be compared against an image manifest.
_ARCH_ALIASES = {
    "aarch64": "arm64",
    "arm64": "arm64",
    "x86_64": "amd64",
    "amd64": "amd64",
}


def _normalise_arch(value: str) -> str:
    """Map a uname- or docker-style architecture onto docker's spelling."""
    v = value.strip().lower()
    return _ARCH_ALIASES.get(v, v)


def host_arch() -> str:
    """This host's architecture in docker's spelling ("amd64", "arm64", …)."""
    return _normalise_arch(platform.machine())


def check_image_arch(image: str) -> None:
    """Raise if *image* was built for a different architecture than this host.

    Docker pulls and creates containers from a foreign-architecture image with only
    a warning, then fails at exec time — under compose it reports the stack as *up*
    while the container never starts, which reads as a working install. Checking the
    image manifest at install time converts that into an actionable error.

    The architecture is read from the image itself rather than from the
    ``org.medmcp.stack`` label: the manifest cannot be wrong or forgotten, and a
    multi-arch tag resolves to the host's architecture automatically.

    Raises:
        RuntimeError: the image cannot execute on this host.
    """
    out = _run_docker(["image", "inspect", "--format", "{{.Architecture}}", image], timeout=30)
    image_arch = _normalise_arch(out.stdout)
    host = host_arch()
    if not image_arch or image_arch == host:
        return
    raise RuntimeError(
        f"{image} is built for linux/{image_arch}, but this host is linux/{host}. "
        "It cannot run here. Ask the stack's maintainer to publish a multi-arch "
        "image, or build it locally for this architecture."
    )


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


def resolve_image_digest(image: str) -> str | None:
    """Return the local image's ``sha256:…`` digest, or ``None`` if unresolved.

    Best-effort and offline: reads the ``RepoDigests`` of an **already-present**
    image and never pulls — digest resolution (for a workflow's requirements pin)
    must not block on a registry or drag in a multi-GB image. Prefers the digest
    whose repository matches *image*, falling back to the first available.
    """
    image = image.strip()
    if not image or any(c.isspace() for c in image) or not _image_present(image):
        return None
    try:
        raw = _run_docker(["inspect", "--format", "{{json .RepoDigests}}", image], timeout=30)
    except RuntimeError:
        return None
    try:
        repo_digests = json.loads(raw.stdout.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(repo_digests, list):
        return None
    repo = image.split("@", 1)[0].rsplit(":", 1)[0]
    digests = [
        rd for rd in cast("list[Any]", repo_digests) if isinstance(rd, str) and "@sha256:" in rd
    ]
    for rd in digests:
        rd_repo, _, digest = rd.partition("@")
        if rd_repo == repo:
            return digest
    return digests[0].partition("@")[2] if digests else None


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

    # Refuse a foreign-architecture image here rather than letting it install
    # cleanly and fail at first tool call with "exec format error".
    check_image_arch(image)

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
    if meta.get("network"):
        # Opt-in egress, recorded explicitly so _harden_stack_run_args leaves it
        # alone rather than clamping it to none on every load.
        args += ["--network", "bridge"]
    args = _harden_stack_run_args(args)
    args += ["-v", "${MEDMCP_WORKSPACE}:${MEDMCP_WORKSPACE}", image]
    entry: JsonDict = {"name": name, "command": "docker", "args": args}

    timeout = meta.get("tool_timeout_sec")
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
        entry["tool_timeout_sec"] = float(timeout)

    startup = meta.get("startup_timeout_sec")
    entry["startup_timeout_sec"] = (
        float(startup)
        if isinstance(startup, (int, float)) and not isinstance(startup, bool)
        else DEFAULT_STACK_STARTUP_TIMEOUT_SEC
    )

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


# Enumerating GPUs costs a throwaway container spawn (~1s), and the answer cannot
# change while the process lives — the hardware is fixed. Cached so opening the
# settings drawer does not pay for it every time. Only a non-empty result is kept:
# an empty one may mean docker was briefly unavailable, and caching that would
# require a restart to recover, whereas a host with no GPUs takes the fast path
# above (no LLM image, so no container to spawn).
_gpu_cache: list[JsonDict] | None = None


def list_gpus() -> list[JsonDict]:
    """Best-effort list of GPUs as ``{index, uuid, name}`` for the settings picker.

    The CPU-only core can't enumerate GPUs itself, so it runs ``nvidia-smi`` in a
    throwaway container with *every* GPU exposed (``--device nvidia.com/gpu=all``) so
    the real host indices/UUIDs come back — querying the (possibly pinned) LLM
    container would only show its own GPU(s), renumbered. Reuses the LLM image (a
    present CUDA image). Returns ``[]`` when not enumerable — the UI falls back to
    free text.
    """
    global _gpu_cache
    if _gpu_cache is not None:
        return list(_gpu_cache)
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
    if gpus:
        _gpu_cache = gpus
    return list(gpus)


# Serializes the config.toml read-modify-write within this process: the
# workspace server runs the sync from a worker thread per websocket connect,
# and several connects race after a settings-triggered restart.
_config_write_lock = threading.Lock()


# The pre_tool hook that refuses a tool call whose paths do not resolve, so the
# model corrects itself before the call ever reaches the approval dialog (see
# medmcp.pathguard). Registered here rather than left to hand-editing so it cannot
# silently go missing from an install.
PATHGUARD_HOOK_NAME = "medmcp-pathguard"


def _ensure_pathguard_hook(vibe_home: Path) -> None:
    """Register the path-guard ``pre_tool`` hook in ``<vibe_home>/hooks.toml``.

    Hooks live in their own file: vibe reads ``hooks.toml`` from each project root
    and from ``VIBE_HOME``, and never takes them from ``config.toml``. Since its
    file sources default to ``("user",)``, no project root is consulted at all and
    ``VIBE_HOME`` is the one location that is always read.

    Matched against every tool (no ``match``): ``write_file`` and ``edit`` take
    paths just as the imaging stacks do. ``strict`` stays false so a hook that
    fails to run is a passthrough — a broken guard must not be able to block work.
    Any other hook in the file is preserved.
    """
    path = vibe_home / "hooks.toml"
    others: list[JsonDict] = []
    if path.is_file():
        try:
            with path.open("rb") as fh:
                raw = cast("list[JsonDict]", tomllib.load(fh).get("hooks", []))
            others = [h for h in raw if h.get("name") != PATHGUARD_HOOK_NAME]
        except (OSError, tomllib.TOMLDecodeError, AttributeError):
            log.warning("could not read %s; rewriting it", path)
    entry: JsonDict = {
        "name": PATHGUARD_HOOK_NAME,
        "type": "pre_tool",
        "command": "medmcp-pathguard",
        "timeout": 10.0,
        "strict": False,
        "description": (
            "Refuse a tool call whose file paths do not resolve, handing the "
            "reason back to the model so it can correct them."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump({"hooks": [*others, entry]}, fh)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


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
    result. An entry carrying no ``startup_timeout_sec`` at all is given one —
    vibe's own 10s default is too tight for a stack that cold-starts a container.
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
        if srv.get("external"):
            # An HTTP server has no command/args to preserve or overwrite, and
            # nothing about it is owned by config.toml — the state file is the
            # single source of truth, so it is rebuilt from scratch each sync.
            # That also means disabling the feature removes these entries.
            entry = {
                "transport": srv["transport"],
                "name": name,
                "url": srv["url"],
            }
            if srv.get("auth"):
                entry["auth"] = srv["auth"]
            new_entries.append(entry)
        elif name in existing_by_name:
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

    # vibe defaults startup_timeout_sec to 10s, and a discovery miss drops the whole
    # stack for the session with nothing but a log line, so give every entry that
    # does not carry one an explicit floor (see DEFAULT_STACK_STARTUP_TIMEOUT_SEC).
    # This runs over entries preserved from config.toml too: the preserve-branch
    # above keeps vibe-owned fields as they are, which would otherwise pin an
    # already-installed stack to the 10s default forever. A value that is actually
    # set — by a manifest, a package, or by hand here — is left alone.
    for entry in new_entries:
        if entry.get("startup_timeout_sec") is not None:
            continue
        is_container = Path(str(entry.get("command", ""))).name == "docker"
        entry["startup_timeout_sec"] = (
            DEFAULT_STACK_STARTUP_TIMEOUT_SEC if is_container else DEFAULT_STARTUP_TIMEOUT_SEC
        )

    # Route every stack through the pre-warm proxy when the pool is enabled. vibe
    # then spawns the cheap `medmcp-mcp-proxy <stack>` shim, which forwards to the
    # persistent BackendPool; the real launch specs go to backends.json. Disabled
    # is byte-for-byte the legacy behaviour, minus any stale proxy env keys.
    # External servers are exempt: the pool exists to amortise process spawn and
    # CUDA init for local stacks, and there is no such cost to amortise for an
    # HTTP endpoint — routing one through a spawn-shim would only break it.
    if stack_pool_enabled():
        _write_backend_registry([s for s in servers if not s.get("external")])
        ws_root = os.environ.get("MEDMCP_WORKSPACE") or None
        new_entries = [
            entry if entry.get("url") else _proxied_entry(entry, ws_root) for entry in new_entries
        ]
    else:
        _strip_pool_env(new_entries)

    cfg["mcp_servers"] = new_entries

    # Point vibe at the prompt that matches the posture actually in force. Only a
    # value this function owns is overwritten, so a hand-set custom prompt id is
    # left alone rather than being reset on the next sync.
    use_external = any(s.get("external") for s in servers) and write_external_prompt_variant()
    owned_prompt_ids = ("", BASE_SYSTEM_PROMPT_ID, EXTERNAL_SYSTEM_PROMPT_ID)
    if str(cfg.get("system_prompt_id", "")) in owned_prompt_ids:
        cfg["system_prompt_id"] = (
            EXTERNAL_SYSTEM_PROMPT_ID if use_external else BASE_SYSTEM_PROMPT_ID
        )

    # Collect skills_path values from discovered servers and write them to
    # skill_paths so vibe-acp loads the bundled skill docs automatically.
    #
    # Personal workflows are deliberately NOT here. A distilled workflow is a
    # recorded sequence of tool calls, replayed verbatim by replay.py on new
    # inputs — its value is that it does exactly what it did last time. Exposing
    # it as a skill handed it back to the agent as prose to reinterpret, which is
    # the opposite guarantee, and put a second, unreviewed path to the same tools
    # next to the deterministic one. The replay engine is the only way to run one.
    cfg["skill_paths"] = [srv["skills_path"] for srv in servers if srv.get("skills_path")]

    # Workflow names were listed in disabled_skills to keep deactivated ones from
    # loading. With no workflow dir in skill_paths there is nothing to disable, so
    # drop those entries — but preserve the rest (e.g. vibe's own skill-creator,
    # disabled so distillation stays the single skill-authoring path).
    workflow_names = {w["name"] for w in discover_workflows()}
    existing_disabled = cast("list[str]", cfg.get("disabled_skills", []))
    cfg["disabled_skills"] = sorted(s for s in existing_disabled if s not in workflow_names)

    # Hooks live in their own file next to the config, not inside it.
    _ensure_pathguard_hook(VIBE_HOME)

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
