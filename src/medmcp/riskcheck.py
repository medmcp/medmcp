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
# Code the assistant wrote itself and is about to run: an interpreter (with a
# script, inline code or a heredoc), a shell started on a string, an awk program,
# or a code file written by the shell. A stack tool is recorded, replayable and
# reviewed; this is not, so it is always put to the user (see medmcp.server).
_ASSISTANT_CODE = re.compile(
    # an interpreter in command position (also behind uv run, env, exec, time,
    # nice, xargs, timeout N, or with a directory in front of it)
    r"(?:^|[;&|(`\n]\s*|\b(?:uv\s+run|env|exec|time|nice|xargs|timeout\s+\S+)\s+)"
    r"(?:\S*/)?(?:python[0-9.]*|Rscript|node|perl|ruby|julia|php|lua|deno|bun|octave|matlab)"
    r"(?=\s|$)"
    r"|\b(?:ba|z|k|da)?sh\s+(?:-\w*c\b|\S+\.sh\b)|\bawk\s+(?:-\S+\s+)*'[^']*[{}]"
    r"|>\s*\S+\.(?:py|r|R|sh|js|jl|pl|rb|ipynb)\b"
)
# File names that hold code, for write_file / edit.
_CODE_FILE = re.compile(r"\.(?:py|r|R|sh|bash|js|mjs|jl|pl|rb|ipynb|m)$")
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
    if _ASSISTANT_CODE.search(command):
        _add(keys, "assistant_code")


def runs_assistant_code(tool: str, arguments: JsonDict) -> bool:
    """Whether this call runs, or writes, code the assistant wrote itself.

    True for a shell command that starts an interpreter, a shell on a string or
    an awk program, or writes a code file, and for ``write_file``/``edit`` of a
    code file (``.py``, ``.R``, ``.sh`` …).
    """
    if tool == "bash":
        command = arguments.get("command")
        return bool(_ASSISTANT_CODE.search(command if isinstance(command, str) else str(arguments)))
    if tool in {"write_file", "edit"}:
        return any(
            isinstance(v, str) and bool(_CODE_FILE.search(v.strip()))
            for k, v in arguments.items()
            if "path" in k.lower() or k.lower() in {"file", "filename"}
        )
    return False


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
        if runs_assistant_code(tool, arguments):
            _add(keys, "assistant_code")
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
