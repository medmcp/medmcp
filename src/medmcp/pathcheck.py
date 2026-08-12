"""Deterministic existence checks for the file paths in a tool call's arguments.

The local model routinely invents plausible-looking paths. Left unchecked, the
first sign is the tool failing after the user already approved it, which costs a
turn and reads like a tool bug rather than a bad argument. This module inspects a
pending call's arguments and reports, per path, whether it is actually there — so
the approval dialog can say so *before* the call runs.

Unlike the risk tags beside it in that dialog, nothing here goes through the LLM:
this is a handful of ``stat`` calls and cannot itself hallucinate.

**Why role matters.** Checking "does this path exist?" is the wrong question for
half the arguments a tool takes. An ``output_dir`` is *supposed* not to exist yet.
Flagging those would put a warning on essentially every call, and a warning that
fires every time is one users learn to ignore — worse than no check at all. So
each path is first classified as an input (must already exist), an output (its
parent must exist; an existing target means an overwrite), or, when the name does
not clearly say, neither — reported without a warning rather than guessed at.

The classification is deliberately conservative: a name is only called an input or
an output when a token in it says so unambiguously. ``mask`` and ``target`` are
left unclassified on purpose — both are an input in some tools and an output in
others, and a confident wrong answer here is worse than an honest neutral one.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

PathRole = Literal["input", "output", "unknown"]
PathStatus = Literal["ok", "missing", "parent_missing", "will_overwrite", "outside_workspace"]
Severity = Literal["error", "warning", "info"]


class PathFinding(TypedDict):
    """One checked path argument, as sent to the browser."""

    param: str
    value: str
    role: PathRole
    status: PathStatus
    severity: Severity
    note: str


# Name tokens that unambiguously mark a parameter as one or the other. Anything
# else stays "unknown" — see the module docstring on why guessing is not worth it.
_OUTPUT_TOKENS = frozenset({"output", "out", "dest", "destination"})
_INPUT_TOKENS = frozenset(
    {"input", "src", "source", "moving", "fixed", "reference", "template", "atlas", "dicom", "root"}
)

# Parameter names that denote a path regardless of what the value looks like, so a
# bare name with no separator or suffix (e.g. output_dir="results") is still checked.
_PATH_NAME_RE = re.compile(r"(^|_)(path|dir|file|folder|prefix|root)s?$", re.IGNORECASE)

# A file suffix, optionally compressed (".nii.gz"). Used to spot a value as a path
# when the parameter name gives nothing away.
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,7}(\.gz|\.bz2|\.xz|\.zst)?$")

# Splits a parameter name into lowercase word tokens: "input_path" / "inputPath"
# / "input-path" all yield {"input", "path"}.
_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

# Cap the number of findings so a pathological argument object can't produce an
# unreadable dialog (or a large frame). Tool calls carry a handful of paths.
_MAX_FINDINGS = 24


def _tokens(name: str) -> frozenset[str]:
    """Return the lowercase word tokens of a parameter *name*."""
    return frozenset(m.group(0).lower() for m in _TOKEN_RE.finditer(name))


def looks_like_path(value: str) -> bool:
    """Heuristic: does *value* look like a filesystem path?

    Whitespace disqualifies it, so a shell command string (``ls -R data/x``) is not
    mistaken for a path; imaging paths in this domain do not contain spaces.
    """
    if not value or any(c.isspace() for c in value):
        return False
    return "/" in value or bool(_EXT_RE.search(value))


def classify_role(param: str) -> PathRole:
    """Classify a parameter *name* as an input path, an output path, or neither."""
    toks = _tokens(param)
    is_out = bool(toks & _OUTPUT_TOKENS)
    is_in = bool(toks & _INPUT_TOKENS)
    # "input" wins a contradictory name (e.g. "input_output_dir" is not a thing, but
    # a stack could ship one); an unclear name is better reported as unknown.
    if is_in and is_out:
        return "unknown"
    if is_out:
        return "output"
    if is_in:
        return "input"
    return "unknown"


def _resolve(value: str, workspace: Path) -> Path:
    """Resolve *value* to an absolute path, relative paths against *workspace*.

    Normalised lexically rather than via ``Path.resolve()``: the point is to judge a
    path that may not exist, and resolve() would additionally follow symlinks.
    """
    raw = Path(value) if Path(value).is_absolute() else workspace / value
    return Path(os.path.normpath(str(raw)))


def _within(path: Path, workspace: Path) -> bool:
    """True if *path* is the workspace root or sits inside it."""
    return path == workspace or path.is_relative_to(workspace)


def _check_one(param: str, value: str, workspace: Path) -> PathFinding:
    """Check a single path argument and describe what was found."""
    role = classify_role(param)
    resolved = _resolve(value, workspace)

    def finding(status: PathStatus, severity: Severity, note: str) -> PathFinding:
        return {
            "param": param,
            "value": value,
            "role": role,
            "status": status,
            "severity": severity,
            "note": note,
        }

    # Outside the workspace is worth saying regardless of role: the agent's tools are
    # meant to operate on the mounted workspace, and a stack container will not see
    # anything else even when the path exists on the host.
    if not _within(resolved, workspace):
        return finding(
            "outside_workspace",
            "warning",
            "outside the workspace — a tool stack will not be able to see it",
        )

    exists = resolved.exists()

    if role == "input":
        if exists:
            return finding("ok", "info", "exists")
        return finding("missing", "error", "does not exist")

    if role == "output":
        parent = resolved.parent
        if not parent.exists():
            return finding(
                "parent_missing",
                "error",
                f"the containing folder {parent.name or parent} does not exist",
            )
        if exists:
            return finding("will_overwrite", "warning", "already exists and would be overwritten")
        return finding("ok", "info", "will be created")

    # Unknown role: report what is there without judging it. Deliberately never an
    # error — the parameter may legitimately be either, and a wrong red flag here
    # would erode trust in the ones that are right.
    if exists:
        return finding("ok", "info", "exists")
    return finding("missing", "info", "does not exist yet (this tool may create it)")


def _coerce_arguments(raw_input: object) -> dict[str, Any]:
    """Return the argument mapping, parsing the JSON-string form if needed.

    vibe-acp ships some tool calls' ``rawInput`` as a JSON-encoded *string* rather
    than an object, so accept both. Anything else yields no arguments.
    """
    value = raw_input
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def check_tool_call_paths(raw_input: object, workspace: Path) -> list[PathFinding]:
    """Check every path-looking argument in *raw_input* against the filesystem.

    Best-effort and non-blocking: this annotates the approval dialog, it never
    decides anything. Any argument that cannot be interpreted is skipped rather
    than reported, so a finding always refers to a real string argument.

    Args:
        raw_input: The tool call's arguments — a mapping, or the JSON-string form.
        workspace: Root that relative paths resolve against and that paths are
            expected to stay inside.

    Returns:
        One finding per checked path, in argument order, capped at a readable
        number. Empty when the call takes no path-like arguments.
    """
    findings: list[PathFinding] = []
    for param, value in _coerce_arguments(raw_input).items():
        # A list of paths (e.g. several inputs) is checked element-wise; everything
        # non-string is skipped, since only a string can name a path.
        values: list[Any] = cast("list[Any]", value) if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, str):
                continue
            if not (looks_like_path(item) or _PATH_NAME_RE.search(param)):
                continue
            findings.append(_check_one(param, item, workspace))
            if len(findings) >= _MAX_FINDINGS:
                return findings
    return findings
