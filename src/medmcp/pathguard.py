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

It also turns back a call whose arguments the tool's input schema rejects. vibe
2.26 validates the arguments a ``pre_tool`` hook lets through (its
``effective_arguments``) against the tool's schema once more, and an invalid set
there is not a tool error: the core rejects it as an invalid command and the
whole turn dies (the sandbox's own check before the hook lets an invalid call
through when the schema uses the ``$defs``/``$ref`` pydantic writes for a nested
argument, as ``select_series``'s criteria do). A
denial is the one answer that skips that check, so the guard validates the call
against the stack's schema as the pool last listed it (the broker's
``tool_schema`` op) and denies with what is wrong and what the parameter takes.
That denial is not capped: letting an invalid call through ends the turn. No
schema known, or anything going wrong while checking, allows as before.

A stack tool is called from ``run_typescript``, and that program -- so the model
-- gets only ``tool_skipped`` back from a denial under vibe 2.26.1, never the
reason. So every denial of a stack call is also left with the core
(:class:`DenialNotes`), and the LLM shim adds its reason to the result the model
reads (:func:`medmcp.llmshim.note_denied_calls`). Local models only: a cloud
model does not go through the shim.

The hook contract is "exit 0 and print a JSON object on stdout", so every failure
path here still prints ``{"decision": "allow"}``: a broken guard must never be
able to block work.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator, exceptions, validators
from jsonschema.exceptions import ValidationError

from medmcp import harness, plan
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


# Bounds on the "Expected:" hint: nesting rendered, alternatives listed, length.
_HINT_DEPTH = 3
_HINT_ENUM = 8
_HINT_CHARS = 300


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """*schema* with a local ``$ref`` (``#/$defs/X``) followed, else unchanged."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        try:
            node = node[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError):
            return schema
    return cast("dict[str, Any]", node) if isinstance(node, dict) else schema


def schema_hint(schema: object, root: dict[str, Any], depth: int = 0) -> str:
    """A short, readable rendering of what *schema* accepts ("string or array of string")."""
    if not isinstance(schema, dict):
        return "any value"
    node = _resolve_ref(cast("dict[str, Any]", schema), root)
    if depth >= _HINT_DEPTH:
        return "…"
    for key in ("anyOf", "oneOf"):
        options = node.get(key)
        if isinstance(options, list):
            hints = [schema_hint(o, root, depth + 1) for o in cast("list[object]", options)]
            return " or ".join(dict.fromkeys(hints))
    if "const" in node:
        return json.dumps(node["const"])
    enum = node.get("enum")
    if isinstance(enum, list):
        values = [json.dumps(v) for v in cast("list[object]", enum)]
        more = ", …" if len(values) > _HINT_ENUM else ""
        return "one of " + ", ".join(values[:_HINT_ENUM]) + more
    kind = node.get("type")
    if isinstance(kind, list):
        return " or ".join(str(k) for k in cast("list[object]", kind))
    if kind == "array":
        items = node.get("items")
        return f"array of {schema_hint(items, root, depth + 1)}" if items else "array"
    if kind == "object" or "properties" in node:
        props = node.get("properties")
        if isinstance(props, dict) and props:
            names = list(cast("dict[str, Any]", props))
            more = ", …" if len(names) > _HINT_ENUM else ""
            return "object with " + ", ".join(names[:_HINT_ENUM]) + more
        return "object"
    return str(kind) if isinstance(kind, str) else "any value"


def _where(path: list[object]) -> str:
    """A JSON path as the model wrote it: ``criteria.sequence[0]``; the root is "(arguments)"."""
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "(arguments)"


def _holds(error: ValidationError, target: ValidationError) -> bool:
    """True if *target* is *error* or one of the errors under it (``anyOf`` branches)."""
    return error is target or any(_holds(sub, target) for sub in error.context)


def schema_problem(tool: str, tool_input: object, schema: dict[str, Any]) -> str | None:
    """Why *tool_input* does not satisfy *schema*, written for the model; ``None`` if it does.

    Names the most specific error (``best_match``) and, as the hint, what the
    argument it sits under takes — the top-level error's schema, so a value that
    fits none of an ``anyOf`` is shown all the shapes it may take.
    """
    validator = validators.validator_for(schema, default=Draft202012Validator)(schema)
    errors = list(validator.iter_errors(cast("Any", tool_input)))
    if not errors:
        return None
    # best_match is unannotated in the stubs; it returns one of the errors.
    deep = cast("ValidationError", exceptions.best_match(errors))  # pyright: ignore[reportUnknownMemberType]
    top = next((e for e in errors if _holds(e, deep)), deep)
    if top.validator == "required":
        required = ", ".join(repr(r) for r in cast("list[object]", top.validator_value))
        expected = f"{_where(list(top.absolute_path))} requires {required}"
    else:
        expected = f"{_where(list(top.absolute_path))} takes " + schema_hint(top.schema, schema)
    if len(expected) > _HINT_CHARS:
        expected = expected[: _HINT_CHARS - 1] + "…"
    return (
        f"{tool} arguments are invalid: {_where(list(deep.absolute_path))}: {deep.message}. "
        f"Expected: {expected}.\n\n"
        "Nobody refused this call and the tool did not run. Fix only this argument, "
        f"keep every other argument exactly as it was, and call {tool} again now."
    )


def _known_schema(route: str) -> dict[str, Any] | None:
    """The input schema of the stack tool *route* names, from the core; ``None`` if unknown.

    Asked over the broker (``tool_schema``), which answers from the tool lists the
    stack pool keeps (``.vibe/state/tool_schemas.json``) and starts nothing. vibe's
    own tools are never asked about: the harness validates those itself.
    """
    group, dot, _tool = route.partition(".")
    if (
        not route
        or route in harness.BUILTIN_ROUTES
        or route in harness.HARNESS_TOOLS
        or (dot and group == "vibe")
    ):
        return None
    response = asyncio.run(plan.ask_core("tool_schema", {"tool": route}, timeout=3.0))
    schema = response.get("schema") if response is not None else None
    return cast("dict[str, Any]", schema) if isinstance(schema, dict) else None


def check_schema(route: str, tool_input: object) -> dict[str, Any] | None:
    """A denial for a call its tool's schema rejects; ``None`` to go on (also on any error)."""
    try:
        schema = _known_schema(route)
        if schema is None:
            return None
        tool = route.partition(".")[2] or route
        problem = schema_problem(tool, tool_input, schema)
    except Exception as exc:
        print(f"medmcp-pathguard: schema check skipped: {exc}", file=sys.stderr)
        return None
    return {"decision": "deny", "reason": problem} if problem else None


class DenialNotes:
    """The guard's recent denials of stack calls, until the LLM shim takes them.

    A stack tool is called from ``run_typescript``, and under vibe 2.26.1 the
    program -- and so the model -- gets only ``tool_skipped: Tool execution was
    skipped by Runtime policy.`` back from a denial, never its reason. So the guard
    also hands each denial to the core (broker op ``record_denial``), and the shim
    (:mod:`medmcp.llmshim`), seeing that text in the next request to the model,
    takes the oldest live denial of the tool the error names (``take_denial``) and
    adds its reason to the result. The tool is all both sides can see: the shim has
    the program's error, not the arguments the program built. Denials expire after
    *ttl_sec* and at most *limit* are kept, so one nobody took cannot resurface
    much later.
    """

    def __init__(
        self,
        ttl_sec: float = 120.0,
        limit: int = 100,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep up to *limit* denials, each for *ttl_sec*."""
        self._ttl = ttl_sec
        self._limit = limit
        self._clock = clock
        self._denials: list[tuple[float, str, str, str]] = []

    def _live(self) -> list[tuple[float, str, str, str]]:
        now = self._clock()
        self._denials = [d for d in self._denials if now - d[0] < self._ttl]
        return self._denials

    def add(self, session_id: str, tool: str, reason: str) -> None:
        """Record that the call of *tool* (its bare name) was denied for *reason*."""
        live = self._live()
        live.append((self._clock(), session_id, tool, reason))
        del live[: max(0, len(live) - self._limit)]

    def take(self, tool: str) -> str | None:
        """The reason of the oldest live denial of *tool*, removed; ``None`` if none."""
        for i, (_, _session, name, reason) in enumerate(self._live()):
            if name == tool:
                del self._denials[i]
                return reason
        return None


def _record_denial(route: str, session_id: str, response: dict[str, Any]) -> dict[str, Any]:
    """Hand a stack call's denial to the core for the LLM shim; returns *response*.

    Only stack calls (``mcp_<stack>.<tool>``): they are the ones made from
    ``run_typescript``, whose program never sees the reason. vibe's own tools are
    called directly and their denial reason reaches the model as it is.
    """
    group, dot, tool = route.partition(".")
    if dot and group.startswith("mcp_") and response.get("decision") == "deny":
        payload = {"session": session_id, "tool": tool, "reason": str(response["reason"])}
        try:
            asyncio.run(plan.ask_core("record_denial", payload, timeout=2.0))
        except Exception as exc:
            print(f"medmcp-pathguard: denial not recorded: {exc}", file=sys.stderr)
    return response


def decide(invocation: dict[str, Any]) -> dict[str, Any]:
    """Return the hook response for one ``pre_tool`` invocation (recorded if a denial)."""
    return _record_denial(
        str(invocation.get("tool_name") or ""),
        str(invocation.get("session_id") or ""),
        _decide(invocation),
    )


def _decide(invocation: dict[str, Any]) -> dict[str, Any]:
    """The hook response for one ``pre_tool`` invocation."""
    # The harness names a tool by route (``mcp_medmcp_neuro.skull_strip``,
    # ``file_system.read_file``); the checks and the ledger key use MedMCP's name.
    tool_name = harness.canonical_tool_name(
        str(invocation.get("tool_name") or ""), container_stack_names()
    )
    if plan.is_plan_tool(tool_name):
        # A plan names folders that do not exist yet (its output folder); the
        # person reads it, and each call it leads to is checked on its own.
        return {"decision": "allow"}
    session_id = str(invocation.get("session_id") or "")
    tool_input = invocation.get("tool_input")

    # First: an invalid call let through here would end the turn, not fail the call.
    denial = check_schema(str(invocation.get("tool_name") or ""), tool_input)
    if denial is not None:
        return denial

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
