"""The agent's approval posture, re-imposed on every config sync.

Everything that decides whether a tool call asks first lives in files the agent
can reach: ``config.toml`` (``default_agent``, ``auto_approve``, ``[tools.*]``),
the MedMCP agent profile (whose keys sit above the config), ``hooks.toml`` (a
``pre_tool`` hook may answer ``allow``), ``trusted_folders.toml`` (a trusted
workspace lets its own ``.vibe/`` override all of the above) and the
``stacks.d`` manifests (whose ``command`` the core runs). A single approved
write to one of them would otherwise outlive the chat it was approved in, since
the sync preserved whatever it did not own. These functions put each one back,
and say so in the audit log when they had to.

Stdlib-only apart from :mod:`medmcp.statefile`; :mod:`medmcp.settings` calls
them under its config lock.
"""

from __future__ import annotations

import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Any, cast

from medmcp.statefile import atomic_write_toml

JsonDict = dict[str, Any]

_audit: logging.Logger = logging.getLogger("medmcp.audit")

AGENT_NAME = "medmcp"

# Built-in tools and the permission each must have. Anything that would run
# without asking — a tool missing from here set to "always", an allowlist on a
# gated tool — is a decision the person never made.
PINNED_TOOL_PERMISSIONS: dict[str, str] = {
    "read_file": "always",
    "grep": "always",
    "todo": "always",
    "task": "always",
    "ask_user_question": "always",
    "write_file": "ask",
    "edit": "ask",
    "bash": "ask",
    "web_fetch": "ask",
}

# vibe's one-shot migration unions read-only commands into the bash allowlist
# unless this id is recorded as applied.
_PINNED_MIGRATION = "bash_read_only_defaults_v1"
# The only allowlist entries a gated tool may carry. vibe adds `find` to bash's on
# every start and rewrites config.toml to do it, which a read-only agent home
# refuses; listing it keeps the file as vibe wants it.
_PINNED_ALLOWLISTS: dict[str, list[str]] = {"bash": ["find"]}

# The MedMCP agent profile, minus its prompt id (owned by the prompt sync). A
# profile's keys override config.toml, so any key not listed here is removed.
CANONICAL_AGENT_PROFILE: JsonDict = {
    "display_name": "MedMCP",
    "description": "Asks before every tool call; MedMCP system prompt",
    "safety": "neutral",
    "enable_background_processes": False,
    "enable_subagents": False,
    "disabled_tools": ["exit_plan_mode", "cron", "web_search"],
}

PATHGUARD_HOOK_NAME = "medmcp-pathguard"


def _note(where: str, what: str) -> None:
    _audit.warning("posture: %s: %s (restored)", where, what)


def enforce_config(cfg: JsonDict) -> list[str]:
    """Put the approval keys of a parsed ``config.toml`` back; return what changed."""
    changed: list[str] = []
    if cfg.get("default_agent") != AGENT_NAME:
        changed.append(f"default_agent={cfg.get('default_agent')!r}")
        cfg["default_agent"] = AGENT_NAME
    for key in ("auto_approve", "bypass_tool_permissions"):
        if cfg.get(key):
            changed.append(f"{key}=true")
            cfg[key] = False

    migrations = [str(m) for m in cast("list[Any]", cfg.get("applied_migrations", []))]
    if _PINNED_MIGRATION not in migrations:
        changed.append(f"applied_migrations without {_PINNED_MIGRATION}")
        cfg["applied_migrations"] = [*migrations, _PINNED_MIGRATION]
    disabled = [str(t) for t in cast("list[Any]", cfg.get("disabled_tools", []))]
    if "web_search" not in disabled:
        changed.append("web_search enabled")
        cfg["disabled_tools"] = [*disabled, "web_search"]

    tools = cast("dict[str, JsonDict]", cfg.setdefault("tools", {}))
    for name, permission in PINNED_TOOL_PERMISSIONS.items():
        entry = tools.setdefault(name, {})
        if entry.get("permission") != permission:
            changed.append(f"tools.{name}.permission={entry.get('permission')!r}")
            entry["permission"] = permission
        allowlist = _PINNED_ALLOWLISTS.get(name, [])
        if permission != "always" and entry.get("allowlist", []) != allowlist:
            if not set(entry.get("allowlist", [])) <= set(allowlist):
                changed.append(f"tools.{name}.allowlist={entry['allowlist']!r}")
            entry["allowlist"] = list(allowlist)
    for name, entry in tools.items():
        if name not in PINNED_TOOL_PERMISSIONS and entry.get("permission") == "always":
            changed.append(f"tools.{name}.permission='always'")
            entry["permission"] = "ask"
    for what in changed:
        _note("config.toml", what)
    return changed


def enforce_agent_profile(path: Path) -> list[str]:
    """Reduce the profile at *path* to the canonical keys; return what changed.

    The prompt id is kept as found — :func:`medmcp.settings` owns it. The file is
    only rewritten when something was off, so its comments survive otherwise.
    """
    try:
        with path.open("rb") as fh:
            profile = tomllib.load(fh)
    except FileNotFoundError:
        profile = {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        profile = {}
        _note(str(path), f"unreadable ({exc})")
    expected: JsonDict = {**CANONICAL_AGENT_PROFILE}
    if "system_prompt_id" in profile:
        expected["system_prompt_id"] = profile["system_prompt_id"]
    if profile == expected:
        return []
    changed = sorted(
        f"{k}={profile.get(k)!r}"
        for k in {*profile, *expected}
        if profile.get(k) != expected.get(k) and k != "system_prompt_id"
    )
    for what in changed:
        _note(path.name, what)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_toml(path, expected)
    return changed


def pathguard_hook() -> JsonDict:
    """The one hook this workspace runs: the path guard (see :mod:`medmcp.pathguard`).

    Matched against every tool (no ``match``): ``write_file`` and ``edit`` take
    paths just as the imaging stacks do. ``strict`` stays false so a hook that
    fails to run is a passthrough — a broken guard must not be able to block work.
    """
    return {
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


def enforce_hooks(path: Path) -> list[str]:
    """Make ``hooks.toml`` hold exactly the path guard; return what was removed.

    vibe reads this file with no trust gate, and a ``pre_tool`` hook runs before
    the permission check on every call and may answer ``allow``.
    """
    current: list[JsonDict] = []
    if path.is_file():
        try:
            with path.open("rb") as fh:
                current = cast("list[JsonDict]", tomllib.load(fh).get("hooks", []))
        except (OSError, tomllib.TOMLDecodeError):
            current = []
    wanted = pathguard_hook()
    removed = [str(h.get("name", "?")) for h in current if h.get("name") != PATHGUARD_HOOK_NAME]
    for name in removed:
        _note("hooks.toml", f"hook {name!r}")
    if current != [wanted]:
        atomic_write_toml(path, {"hooks": [wanted]})
    return removed


def enforce_trusted_folders(path: Path) -> list[str]:
    """Empty vibe's trusted-folder list; return the entries removed.

    A trusted workspace would let a ``.vibe/`` inside it — writable by the agent —
    supply its own config, hooks and agent profiles.
    """
    if not path.is_file():
        return []
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        data = {}
    trusted = [str(t) for t in cast("list[Any]", data.get("trusted", []))]
    if not trusted and "trusted" in data:
        return []
    for entry in trusted:
        _note("trusted_folders.toml", f"trusted {entry!r}")
    data["trusted"] = []
    data.setdefault("untrusted", [])
    atomic_write_toml(path, data)
    return trusted


# ── stacks.d manifests ───────────────────────────────────────────────────────

# `docker run` flags a manifest may carry before the image. Anything else —
# --privileged, another -v, --pid/--ipc/--userns, --network host, a capability
# or security option beyond the hardening — is refused, since the core runs the
# manifest as written.
_FLAGS_NO_VALUE: frozenset[str] = frozenset({"--rm", "-i", "--interactive", "--init"})
_FLAG_VALUES: dict[str, re.Pattern[str]] = {
    "--network": re.compile(r"none|bridge"),
    "--cap-drop": re.compile(r"ALL"),
    "--cap-add": re.compile(r"DAC_OVERRIDE"),
    "--security-opt": re.compile(r"no-new-privileges(:true)?"),
    "--pids-limit": re.compile(r"\d+"),
    "--shm-size": re.compile(r"\d+[bkmgBKMG]?"),
    "--memory": re.compile(r"\d+[bkmgBKMG]?"),
    "--cpus": re.compile(r"[\d.]+"),
    "--device": re.compile(r"nvidia\.com/gpu=[\w,.-]+"),
    "-e": re.compile(r"[A-Za-z_][A-Za-z0-9_]*(=.*)?", re.DOTALL),
    "--env": re.compile(r"[A-Za-z_][A-Za-z0-9_]*(=.*)?", re.DOTALL),
    "--name": re.compile(r"[\w.-]+"),
    # Inside the container, like the command after the image.
    "--entrypoint": re.compile(r"[\w./-]+"),
    "-w": re.compile(r"/.*"),
    "--workdir": re.compile(r"/.*"),
}
_VOLUME_FLAGS: frozenset[str] = frozenset({"-v", "--volume"})


def split_flag_values(args: list[str]) -> list[str]:
    """``--flag=value`` written as ``--flag value``, up to the image.

    So the hardening and :func:`unsafe_stack_args` see one spelling: an
    ``--network=host`` would otherwise be missed by a presence check, and the
    hardening would add a second, conflicting ``--network``.
    """
    out: list[str] = args[:1]
    takes_value = False
    for index, arg in enumerate(args[1:], start=1):
        if takes_value:
            out.append(arg)
            takes_value = False
        elif arg.startswith("--") and "=" in arg:
            out += arg.split("=", 1)
        elif arg.startswith("-"):
            out.append(arg)
            takes_value = arg in _FLAG_VALUES or arg in _VOLUME_FLAGS
        else:
            # The image: everything after it belongs to the container's command.
            out += args[index:]
            break
    return out


def unsafe_stack_args(args: list[str], workspace: str | None = None) -> str | None:
    """Why the ``docker run`` *args* of a manifest must not run, or ``None``.

    The only bind mount allowed is the workspace at its own path (``-v W:W``,
    optionally ``:ro``/``:rw``), where *W* is the expanded ``MEDMCP_WORKSPACE``.
    """
    ws = workspace if workspace is not None else os.path.expandvars("${MEDMCP_WORKSPACE}")
    if not args or args[0] != "run":
        return "manifest args must start with 'run'"
    index = 1
    while index < len(args):
        flag = args[index]
        if not flag.startswith("-"):
            return None  # the image; what follows is the container's command
        if flag in _FLAGS_NO_VALUE:
            index += 1
            continue
        if index + 1 >= len(args):
            return f"{flag} has no value"
        value = args[index + 1]
        if flag in _VOLUME_FLAGS:
            if value not in {f"{ws}:{ws}", f"{ws}:{ws}:ro", f"{ws}:{ws}:rw"}:
                return f"bind mount {value!r} is not the workspace"
        elif flag in _FLAG_VALUES:
            if not _FLAG_VALUES[flag].fullmatch(value):
                return f"{flag} {value!r} is not allowed"
        else:
            return f"{flag} is not allowed"
        index += 2
    return "manifest names no image"
