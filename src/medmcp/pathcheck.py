"""Deterministic existence checks for the file paths in a tool call's arguments.

The local model routinely invents plausible-looking paths. Left unchecked, the
first sign is the tool failing after the user already approved it, which costs a
turn and reads like a tool bug rather than a bad argument. This module inspects a
pending call's arguments and reports, per path, whether it is actually there — so
the approval dialog can say so *before* the call runs.

Unlike the risk tags beside it in that dialog, nothing here goes through the LLM:
this is a handful of ``stat`` calls and cannot itself hallucinate.

Being told a file is absent rarely helps on its own, so a missing path also carries
a listing of the nearest folder that *does* exist. That is usually enough to show
the intended path at a glance — the model tends to get the directory right and the
filename wrong, or to invent one subject id in an otherwise correct tree.

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
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

log = logging.getLogger(__name__)

PathRole = Literal["input", "output", "unknown"]
PathStatus = Literal[
    "ok", "missing", "parent_missing", "will_overwrite", "outside_workspace", "unreadable"
]
Severity = Literal["error", "warning", "info"]


class PathFinding(TypedDict):
    """One checked path argument, as sent to the browser.

    When a path is missing, the last three fields carry a look at where it *would*
    have been: knowing a file is absent rarely helps, whereas seeing what sits in
    the nearest folder that does exist usually shows the intended path at a glance.
    They are empty for every other status.
    """

    param: str
    value: str
    role: PathRole
    status: PathStatus
    severity: Severity
    note: str
    # Workspace-relative directory the listing came from ("." is the root).
    nearest: str
    # A ranked, capped sample of what that directory holds; directories end in "/".
    entries: list[str]
    # How many entries it holds (a floor: the scan itself is bounded).
    entry_total: int


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

# How many sibling names to show, and how many directory entries to look at to
# find them. The scan bound keeps a directory holding tens of thousands of files
# from turning one approval dialog into a long walk.
_MAX_ENTRIES = 12
_MAX_SCAN = 2000


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


def _safe_exists(path: Path) -> bool | None:
    """Whether *path* exists, or ``None`` when the filesystem cannot answer.

    ``Path.exists()`` only swallows a specific set of errnos -- ENOENT, ENOTDIR,
    ELOOP, EBADF. Others propagate: EACCES for a path under a directory the server
    cannot read, ENAMETOOLONG for an absurdly long value. Neither says anything
    about whether the file is there, and neither is a reason to fail the check, so
    an unanswerable probe is reported as unknown rather than raised.
    """
    try:
        return path.exists()
    except (OSError, ValueError):
        return None


def _safe_is_dir(path: Path) -> bool:
    """Whether *path* is a directory, treating an unanswerable probe as "no"."""
    try:
        return path.is_dir()
    except (OSError, ValueError):
        return False


# How many trailing components of an outside path to try re-rooting at the
# workspace. Bounded so an absurdly deep value cannot cost a stat per level.
_MAX_TAIL_TRIES = 8


def _did_you_mean(path: Path, workspace: Path) -> Path | None:
    """Longest tail of *path* that names something real inside *workspace*.

    A model that invents a path usually gets the *prefix* wrong and the rest right
    -- a mistyped home directory, a stale mount point -- leaving a tail that still
    matches the real layout. Re-rooting successive tails at the workspace turns
    that into a concrete suggestion instead of a shrug. Longest match wins, since
    the more components agree the less likely the match is a coincidence.
    """
    parts = path.parts
    start = max(1, len(parts) - _MAX_TAIL_TRIES)
    for i in range(start, len(parts)):
        candidate = workspace.joinpath(*parts[i:])
        if _safe_exists(candidate):
            return candidate
    return None


def _shares_top_level_with(path: Path, workspace: Path) -> bool:
    """True if *path* sits under the same first-level directory as *workspace*.

    Separates "meant the workspace and missed" from "somewhere else entirely".
    ``/home/<wrong-user>/scan.nii.gz`` shares ``/home`` with a workspace at
    ``/home/<user>/data`` and is a mistyped path; ``/root/…/MNI152.nii.gz`` shares
    only the root and is plausibly a stack image's own bundled data.
    """
    p, w = path.parts, workspace.parts
    return len(p) > 1 and len(w) > 1 and p[:2] == w[:2]


def _nearest_existing_dir(path: Path, workspace: Path) -> Path | None:
    """Return the closest ancestor of *path* that exists, never leaving *workspace*.

    ``None`` when the walk would step outside the workspace before finding one,
    so a listing is never produced for a directory the workspace does not own.
    """
    for candidate in path.parents:
        if not candidate.is_relative_to(workspace):
            return None
        if _safe_is_dir(candidate):
            return candidate
    return None


def _sample_entries(directory: Path, like: Path) -> tuple[list[str], int]:
    """Return up to :data:`_MAX_ENTRIES` names from *directory*, and how many it holds.

    Entries sharing *like*'s suffix sort first: in a folder of mixed derivatives,
    alphabetical order alone would often truncate away the very file the caller
    meant. Uses ``scandir`` so the directory flag comes from the directory entry
    rather than a ``stat`` per name.
    """
    names: list[str] = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                names.append(entry.name + ("/" if entry.is_dir(follow_symlinks=False) else ""))
                if len(names) >= _MAX_SCAN:
                    break
    except OSError:
        # Unreadable directory: the check is advisory, so degrade to no listing.
        return [], 0
    # ".nii.gz" rather than just ".gz" — the double suffix is what distinguishes
    # imaging volumes from the archives beside them.
    suffix = "".join(like.suffixes[-2:])
    names.sort(key=lambda n: (0 if suffix and n.endswith(suffix) else 1, n.lower()))
    return names[:_MAX_ENTRIES], len(names)


def _check_one(param: str, value: str, workspace: Path) -> PathFinding:
    """Check a single path argument and describe what was found."""
    role = classify_role(param)
    resolved = _resolve(value, workspace)

    def finding(
        status: PathStatus, severity: Severity, note: str, *, show_siblings: bool = False
    ) -> PathFinding:
        nearest, entries, total = "", cast("list[str]", []), 0
        if show_siblings:
            found = _nearest_existing_dir(resolved, workspace)
            if found is not None:
                entries, total = _sample_entries(found, resolved)
                rel = found.relative_to(workspace)
                nearest = str(rel) if str(rel) != "." else "."
        return {
            "param": param,
            "value": value,
            "role": role,
            "status": status,
            "severity": severity,
            "note": note,
            "nearest": nearest,
            "entries": entries,
            "entry_total": total,
        }

    # Outside the workspace splits into three cases that deserve different answers.
    # Collapsing them into one bland warning is what let a hallucinated home
    # directory -- /home/<wrong-user>/… instead of /home/<user>/… -- sail through
    # looking harmless.
    if not _within(resolved, workspace):
        # 1. The same tail names a real file *inside* the workspace. A wrong prefix
        #    on an otherwise correct path is the signature of an invented path, and
        #    the match is the answer, so this is an error and it names the fix.
        suggestion = _did_you_mean(resolved, workspace)
        if suggestion is not None:
            rel = suggestion.relative_to(workspace)
            return finding(
                "outside_workspace",
                "error",
                f"outside the workspace — did you mean {rel}?",
            )
        # 2. It exists on this filesystem. Existence proves it is a host path rather
        #    than something inside a stack image, so the claim is now safe to make.
        if _safe_exists(resolved):
            return finding(
                "outside_workspace",
                "warning",
                "outside the workspace — a tool stack will not be able to see it",
            )
        # 3. Neither -- nothing here and nothing like it in the workspace. Whether
        #    that is a wrong path or a stack's own bundled data turns on where it
        #    points. A path sharing the workspace's top-level directory is aiming
        #    at the workspace's neighbourhood and missing (/home/<wrong-user>/… for
        #    /home/<user>/…), which no image ever does: a stack sees only the
        #    workspace bind-mount and its own image, and image data lives under
        #    /app, /opt, /root and the like. So that is an error; anything further
        #    afield stays unjudged.
        if _shares_top_level_with(resolved, workspace):
            return finding(
                "outside_workspace",
                "error",
                "outside the workspace, and nothing is there",
                show_siblings=True,
            )
        return finding(
            "outside_workspace",
            "warning",
            "outside the workspace — cannot be verified from here",
        )

    exists = _safe_exists(resolved)

    # The filesystem declined to answer (unreadable parent, unusable name). Saying
    # "missing" here would be a guess dressed as a fact, and the file may well be
    # there — so report that it could not be checked and leave it at a warning.
    if exists is None:
        return finding("unreadable", "warning", "could not be checked on this filesystem")

    if role == "input":
        if exists:
            return finding("ok", "info", "exists")
        return finding("missing", "error", "does not exist", show_siblings=True)

    if role == "output":
        parent = resolved.parent
        if not _safe_exists(parent):
            return finding(
                "parent_missing",
                "error",
                f"the containing folder {parent.name or parent} does not exist",
                show_siblings=True,
            )
        if exists:
            return finding("will_overwrite", "warning", "already exists and would be overwritten")
        return finding("ok", "info", "will be created")

    # Unknown role: report what is there without judging it. Deliberately never an
    # error — the parameter may legitimately be either, and a wrong red flag here
    # would erode trust in the ones that are right.
    if exists:
        return finding("ok", "info", "exists")
    return finding(
        "missing", "info", "does not exist yet (this tool may create it)", show_siblings=True
    )


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
            try:
                findings.append(_check_one(param, item, workspace))
            except Exception:
                # Per-argument, so one unusable value costs only its own finding.
                # Checking the whole call inside a single guard would mean a single
                # pathological path silently discards the verdicts on every valid
                # one beside it -- the arguments most worth reporting on.
                log.debug("path check skipped %s=%r", param, item, exc_info=True)
                continue
            if len(findings) >= _MAX_FINDINGS:
                return findings
    return findings
