"""Risk labels for a tool call, from the call itself rather than a model's reading of it.

The approval dialog's explanation is written by the local model — the same model
that may have been steered by something it read, and that can therefore call a
harmful command harmless. These labels are worked out by rules from the tool and
its arguments, so they show whatever the model says; its own labels can add to
them, never take one away. They err towards saying more: a label that fires on a
harmless command costs a glance, one that stays quiet on a harmful one costs the
point of the dialog.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

JsonDict = dict[str, Any]

_NETWORK = re.compile(
    r"\b(curl|wget|nc|ncat|netcat|socat|ssh|scp|sftp|rsync|ftp|telnet|aria2c)\b|https?://"
    r"|\b(urllib|requests|httpx|aiohttp|socket|http\.client)\b",
)
_DELETE = re.compile(r"\b(rm|rmdir|unlink|shred|truncate|wipefs)\b|-delete\b|\bmkfs\b")
_WRITE = re.compile(
    r"(?<![0-9&])>>?(?!&)\s*(?!/dev/null)\S|\b(tee|mv|cp|dd|touch|mkdir|ln)\b"
    r"|\bsed\b[^|;&]*\s-i|\bchmod\b|\bchown\b"
)
_SYSTEM = re.compile(
    r"\b(sudo|su|apt|apt-get|dnf|yum|apk|systemctl|crontab|mount|umount|docker|chmod|chown)\b"
    r"|\b(pip|pip3|uv|npm|npx|conda)\s+(install|add|run|x)\b|\bexport\s+PATH="
)
_HIDDEN = re.compile(
    r"\|\s*(ba|z|k)?sh\b|\beval\b|base64\s+(-d|--decode)|\bexec\s*\(|\bpython3?\s+-c\b"
    r"|\bnode\s+-e\b|\bperl\s+-e\b|\$\(|`"
)
_READS_FILES = re.compile(r"\b(cat|tar|zip|gzip|base64|xxd|od|head|tail)\b|@\S|\s-F\s|--data")
_OUTPUT_ARG = re.compile(r"out|dest|save|write", re.IGNORECASE)


def _add(keys: list[str], key: str) -> None:
    if key not in keys:
        keys.append(key)


def _shell_signals(command: str, keys: list[str]) -> None:
    _add(keys, "code_exec")
    network = bool(_NETWORK.search(command))
    if network:
        _add(keys, "network")
        if _READS_FILES.search(command):
            _add(keys, "data_exfil")
    if _DELETE.search(command):
        _add(keys, "file_delete")
    if _WRITE.search(command):
        _add(keys, "file_write")
    if _SYSTEM.search(command):
        _add(keys, "system_config")
    if _HIDDEN.search(command):
        _add(keys, "hidden_code")


def signals(
    tool: str, arguments: JsonDict, path_findings: Iterable[Mapping[str, object]] = ()
) -> list[str]:
    """Risk keys (see :data:`medmcp.explain.RISK_CATEGORIES`) for one tool call.

    *tool* is MedMCP's name for it (``bash``, ``write_file``, ``<stack>_<tool>``);
    *path_findings* are :func:`medmcp.pathcheck.check_tool_call_paths`' results,
    which say whether a path lies outside the workspace or overwrites a file.
    """
    keys: list[str] = []
    if tool == "bash":
        command = arguments.get("command")
        _shell_signals(command if isinstance(command, str) else str(arguments), keys)
    elif tool in {"write_file", "edit"}:
        _add(keys, "file_write")
    elif tool == "read_file":
        _add(keys, "file_read")
    elif tool == "web_fetch":
        _add(keys, "network")
    elif tool == "skill":
        _add(keys, "skill_load")
    elif "_" in tool:
        # A stack tool: it reads its inputs, and writes when it is given somewhere to.
        _add(keys, "file_read")
        if any(_OUTPUT_ARG.search(k) for k in arguments):
            _add(keys, "file_write")
    if any(f.get("status") == "outside_workspace" for f in path_findings):
        _add(keys, "outside_workspace")
    return keys
