"""A ``pre_tool`` hook that bounces a tool call whose paths cannot be right.

The workspace UI already annotates the approval dialog with what is and is not on
disk (:mod:`medmcp.pathcheck`). That helps the person, but it still asks them to
adjudicate a mistake the machine can see: the model invents a path, and the human
is handed a doomed call to approve.

This closes that loop one step earlier. vibe runs ``pre_tool`` hooks *before* the
permission check, and a denial short-circuits the call, so a rejection here never
reaches the approval dialog. The denial's reason is handed back to the model as a
tool error, which is the one channel that carries an explanation -- the ACP
permission outcome has no field for one. So the model reads why its path was
wrong, corrects it, and proposes again; the person only ever sees calls whose
paths resolve.

Two deliberate limits:

*Only errors deny.* Warnings -- an overwrite, a real path outside the workspace,
one that cannot be verified -- are judgement calls that belong to the person, and
they still travel to the dialog untouched.

*Denials are capped.* A model that keeps guessing would otherwise loop against
this hook invisibly, burning tokens with nothing on screen. After
:data:`_MAX_DENIALS` refusals of the same call the hook stands aside and lets it
through to the dialog, where a human can see what it is stuck on. Silent infinite
retry would make this feature worse than not having it.

The hook contract is "exit 0 and print a JSON object on stdout", so every failure
path here still prints ``{"decision": "allow"}``: a broken guard must never be
able to block work.
"""

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from medmcp import harness
from medmcp.pathcheck import PathFinding, check_tool_call_paths
from medmcp.statefile import atomic_write_json, read_json

# Refusals of the same call before the hook gives up and lets a human look at it.
_MAX_DENIALS = 2

# Bound on the retry ledger, which is keyed per session and never explicitly
# cleaned: past this many entries the oldest half is dropped (dicts preserve
# insertion order). Entries are tiny and only matter for the few turns a model
# spends correcting itself.
_MAX_LEDGER = 500
_LEDGER_KEEP = 250


def _ledger_path() -> Path:
    """Location of the denial ledger, beside the rest of the run-time state.

    ``MEDMCP_PATHGUARD_LEDGER`` moves it: an isolated agent's home is read-only,
    and a ledger that cannot be written would deny the same call forever.
    """
    override = os.environ.get("MEDMCP_PATHGUARD_LEDGER", "").strip()
    if override:
        return Path(override)
    home = os.environ.get("VIBE_HOME")
    base = Path(home) if home else Path(__file__).resolve().parents[2] / ".vibe"
    return base / "pathguard_retries.json"


def _load_ledger(path: Path) -> dict[str, int]:
    """Read the denial counts, treating any problem as an empty ledger."""
    data = read_json(path) or {}
    return {k: v for k, v in data.items() if isinstance(v, int)}


def _save_ledger(path: Path, ledger: dict[str, int]) -> None:
    """Write the ledger atomically, best-effort."""
    if len(ledger) > _MAX_LEDGER:
        ledger = dict(list(ledger.items())[-_LEDGER_KEEP:])
    # A ledger we cannot persist only costs the retry cap, so carry on.
    with contextlib.suppress(OSError):
        atomic_write_json(path, ledger)


def call_signature(session_id: str, tool_name: str, findings: list[PathFinding]) -> str:
    """Stable key for "this call, with these same bad paths".

    Keyed on the offending values rather than the whole argument object so that a
    model retrying with a *different* wrong path gets its own budget, while one
    resubmitting the identical call is recognised as repeating itself.
    """
    payload = "|".join(sorted(f"{f['param']}={f['value']}" for f in findings))
    digest = hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{session_id}:{tool_name}:{digest}"


def build_reason(findings: list[PathFinding], workspace: Path) -> str:
    """Compose the tool-error text the model receives.

    Written for the model, not the person: it states what is wrong, supplies the
    material needed to fix it (a suggestion, or what the nearest real folder
    holds), and says plainly what to do next.
    """
    # Leads with the instruction. vibe wraps this as "Tool 'X' was denied by hook
    # ...", which reads like a refusal -- the shape of a user saying no, after which
    # retrying would be wrong -- so the correction has to be the first thing said,
    # and the fact that nobody refused anything has to be said outright.
    # Phrased as what to do, not as what to refrain from. An earlier version said
    # "do not ask the user", which the model carried forward for the rest of the
    # turn and stopped asking about anything at all -- including the choices a
    # skill explicitly tells it to put to the user. Prohibitions land in the
    # transcript and outlive the moment; instructions to act do not.
    lines = [
        "RETRY THIS CALL WITH A CORRECTED PATH.",
        "",
        "Nobody refused this call and the tool did not fail. It was not run because "
        "the path below does not exist, which is an automatic check you can satisfy "
        "on your own. Fix only the path, keep every other argument exactly as it "
        "was, and call the same tool again now.",
        "",
    ]
    for f in findings:
        lines.append(f"  {f['param']} = {f['value']!r}")
        lines.append(f"    problem: {f['note']}")
        if f["entries"]:
            where = "the workspace root" if f["nearest"] == "." else f["nearest"]
            listed = ", ".join(f["entries"])
            more = ""
            if f["entry_total"] > len(f["entries"]):
                more = f", and {f['entry_total'] - len(f['entries'])} more"
            lines.append(f"    {where} actually contains: {listed}{more}")
    lines.append("")
    lines.append(
        f"Every path must be absolute and inside {workspace}. If the listing above "
        "does not make the right file obvious, list the directory and choose from "
        "what is actually there — do not guess another path."
    )
    return "\n".join(lines)


def is_containerized_tool(tool_name: str) -> bool:
    """True if *tool_name* belongs to a stack that runs as a container.

    MCP tools arrive as ``<server>_<tool>``, and a container stack is exactly one
    with a ``stacks.d/<server>.toml`` manifest. Read by globbing that directory
    rather than through :mod:`medmcp.settings`, whose stack discovery spawns a
    subprocess per uv-tool stack -- far too much for a hook on every tool call.

    A uv-tool stack (local development) runs on this filesystem like a builtin, and
    correctly reports False.
    """
    names = container_stack_names()
    # Longest first so a stack whose name prefixes another cannot mis-bind.
    return any(tool_name.startswith(f"{n}_") for n in sorted(names, key=len, reverse=True))


def container_stack_names() -> list[str]:
    """The names of the installed container stacks (``stacks.d/<name>.toml``)."""
    root = os.environ.get("MEDMCP_STACKS_D")
    stacks_d = Path(root) if root else Path(__file__).resolve().parents[2] / "stacks.d"
    try:
        return [p.stem for p in stacks_d.glob("*.toml")]
    except OSError:
        return []


def decide(invocation: dict[str, Any]) -> dict[str, Any]:
    """Return the hook response for one ``pre_tool`` invocation."""
    # The harness names a tool by route (``mcp_medmcp_neuro.skull_strip``,
    # ``file_system.read_file``); the checks and the ledger key use MedMCP's name.
    tool_name = harness.canonical_tool_name(
        str(invocation.get("tool_name") or ""), container_stack_names()
    )
    session_id = str(invocation.get("session_id") or "")
    tool_input = invocation.get("tool_input")

    # The agent's cwd is the workspace (session/new is started there); the env var
    # is preferred so the two cannot disagree if that ever changes.
    root = os.environ.get("MEDMCP_WORKSPACE") or str(invocation.get("cwd") or "")
    if not root:
        return {"decision": "allow"}
    workspace = Path(root)

    findings = check_tool_call_paths(
        tool_input, workspace, containerized=is_containerized_tool(tool_name)
    )
    errors = [f for f in findings if f["severity"] == "error"]
    if not errors:
        return {"decision": "allow"}

    ledger_file = _ledger_path()
    ledger = _load_ledger(ledger_file)
    key = call_signature(session_id, tool_name, errors)
    seen = ledger.get(key, 0)
    if seen >= _MAX_DENIALS:
        # Stood aside: the model is not converging, so let a human see the call
        # rather than keep refusing it off-screen.
        return {
            "decision": "allow",
            "system_message": (
                f"path guard: allowing {tool_name} after {seen} refusals — "
                "the agent did not correct the path"
            ),
        }

    ledger[key] = seen + 1
    _save_ledger(ledger_file, ledger)
    return {"decision": "deny", "reason": build_reason(errors, workspace)}


def main() -> int:
    """Read a ``pre_tool`` invocation on stdin, print the decision on stdout."""
    response: dict[str, Any] = {"decision": "allow"}
    try:
        parsed = json.loads(sys.stdin.read() or "{}")
        if isinstance(parsed, dict):
            response = decide(cast("dict[str, Any]", parsed))
    except Exception as exc:
        print(f"medmcp-pathguard: {exc}", file=sys.stderr)
        response = {"decision": "allow"}
    json.dump(response, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
