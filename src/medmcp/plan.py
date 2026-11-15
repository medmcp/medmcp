"""Plan first: the agent shows a short plan, the person says Go or Change.

The local model can take minutes over even a simple request, and a request it
misunderstood costs all of them before anyone sees the mistake. So before it does
anything the agent proposes a plan — what it understood, the tool calls it will
make, where the results go — through one tool the core provides,
``medmcp-plan.propose_plan``. The core shows the plan as a card; **Go** approves
exactly the tool calls the plan lists, **Change** sends the person's correction
back to the agent, which proposes again.

This module is the part with no I/O: the plan's schema and its validation
(:func:`parse_plan`), and the matching of a tool call against an approved plan
(:func:`match_call`). The server (``server.py``) wires it
into the permission flow; :func:`main` is the MCP server that offers the tool.

What a Go covers, and what it never covers:

* A **stack tool call** whose tool is named by a step, and every one of whose
  path arguments lies in a path the plan shows: one of the step's inputs or
  outputs, or the plan's ``output_dir`` (or inside one of them). An argument the
  step lists under the same name must hold the value the step shows.
* Never one of vibe's own tools (the shell, file writes) and never **code the
  assistant wrote**: those are put to the person on their own, plan or no plan.
* Never anything else: a call outside the plan goes to the ordinary approval
  dialog, flagged as not in the plan.

The tool says "approved" only on the core's word. The core keeps a verdict for
each plan it decided — Go, or refused with the problems — keyed by the plan
(:func:`plan_key`); the tool takes it over the broker (``plan_verdict``), and a
Go puts the plan in force only when the tool takes it. No verdict — the broker
out of reach, a plan that went to an ordinary dialog, arguments that do not match
the card — is "not approved: propose again", never a Go.

The module names no stack and no stack tool: a plan is checked only for its own
shape, and a step matches a call by the tool name the step gives.

The workspace root and the filesystem root never cover what lies under them — a
plan naming the whole workspace as an input would otherwise approve everything.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from medmcp import harness
from medmcp.pathcheck import classify_role

JsonDict = dict[str, Any]

# The MCP server and tool the core provides. The harness reaches the server as the
# route group ``mcp_medmcp_plan``; MedMCP's own name for the tool is
# ``medmcp-plan_propose_plan``.
PLAN_SERVER = "medmcp-plan"
PLAN_TOOL = "propose_plan"
PLAN_COMMAND = "medmcp-plan-tool"
PLAN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        f"{PLAN_SERVER}_{PLAN_TOOL}",
        f"{harness.normalize_segment(PLAN_SERVER)}_{PLAN_TOOL}",
        f"{harness.mcp_group(PLAN_SERVER)}.{PLAN_TOOL}",
    }
)

# The verdicts the core hands the plan tool (broker op ``plan_verdict``).
VERDICT_GO = "go"
VERDICT_REFUSED = "refused"

# The approval dialog's note on a call outside the approved plan.
OFF_PLAN_REASON = "Not in the plan you approved."

# Bounds that keep a plan a plan: something a person reads in a few seconds.
MAX_STEPS = 12
MAX_TEXT = 600
MAX_QUESTIONS = 5

# The tool's input schema, as the model sees it. Kept small and flat: the local
# model fills it in from TypeScript, and every optional field it can skip is one
# it cannot get wrong.
INPUT_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {
        "understanding": {
            "type": "string",
            "description": "One or two sentences: what you understood the user wants.",
        },
        "steps": {
            "type": "array",
            "description": (
                "The steps, in order (2-5 is typical; one is fine for a lookup). Each "
                "step is one tool call, or a short action without a tool (e.g. "
                "'summarise the results')."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "What the step does, in plain words.",
                    },
                    "tool": {
                        "type": "string",
                        "description": (
                            "The exact name of the tool this step calls, as your "
                            "tools list it; omit for a step that calls none."
                        ),
                    },
                    "inputs": {
                        "type": "object",
                        "description": "The tool's key arguments: input paths, options.",
                    },
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files or folders this step writes.",
                    },
                },
                "required": ["action"],
            },
        },
        "output_dir": {
            "type": "string",
            "description": "The folder results go into (absolute path), if anything is written.",
        },
        "estimated_time": {
            "type": "string",
            "description": "A rough estimate, e.g. 'about 2 minutes'.",
        },
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Anything you need the user to decide before you start.",
        },
    },
    "required": ["understanding", "steps"],
}

TOOL_DESCRIPTION = (
    "Show the user your plan before you do anything, and wait for their answer. "
    "Call this first for every request — a lookup is a one-step plan. The user "
    "answers Go (you then carry out exactly these steps) or Change (their "
    "correction arrives as their next message; propose a revised plan)."
)


class PlanError(ValueError):
    """A plan that cannot be shown: what is wrong with it, one problem per entry."""

    def __init__(self, problems: list[str]) -> None:
        """Keep every *problem* so the model can fix them in one go."""
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class PlanStep:
    """One step: a tool call (``tool`` set) or an action without one."""

    action: str
    tool: str = ""
    inputs: JsonDict = field(default_factory=lambda: cast("JsonDict", {}))
    outputs: tuple[str, ...] = ()

    def to_json(self) -> JsonDict:
        """The step as sent to the browser and recorded in provenance."""
        out: JsonDict = {"action": self.action}
        if self.tool:
            out["tool"] = self.tool
        if self.inputs:
            out["inputs"] = self.inputs
        if self.outputs:
            out["outputs"] = list(self.outputs)
        return out


@dataclass(frozen=True)
class Plan:
    """A validated plan. ``id`` is the tool call that proposed it."""

    id: str
    understanding: str
    steps: tuple[PlanStep, ...]
    output_dir: str = ""
    estimated_time: str = ""
    questions: tuple[str, ...] = ()

    def to_json(self) -> JsonDict:
        """The plan as sent to the browser and recorded in provenance."""
        out: JsonDict = {
            "id": self.id,
            "understanding": self.understanding,
            "steps": [s.to_json() for s in self.steps],
        }
        if self.output_dir:
            out["output_dir"] = self.output_dir
        if self.estimated_time:
            out["estimated_time"] = self.estimated_time
        if self.questions:
            out["questions"] = list(self.questions)
        return out


def is_plan_tool(name: str) -> bool:
    """Whether *name* (canonical or route) is the plan tool."""
    return name in PLAN_TOOL_NAMES


def _text(value: object, where: str, problems: list[str], *, required: bool = False) -> str:
    if value is None or value == "":
        if required:
            problems.append(f"{where} is required")
        return ""
    if not isinstance(value, str):
        problems.append(f"{where} must be a string")
        return ""
    text = value.strip()
    if required and not text:
        problems.append(f"{where} is required")
    if len(text) > MAX_TEXT:
        problems.append(f"{where} is too long (keep it under {MAX_TEXT} characters)")
    return text


def _string_list(value: object, where: str, problems: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        problems.append(f"{where} must be a list of strings")
        return ()
    out: list[str] = []
    for item in cast("list[object]", value):
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif not isinstance(item, str):
            problems.append(f"{where} must be a list of strings")
            return ()
    return tuple(out)


def _coerce(raw: object) -> JsonDict:
    """The argument mapping, accepting the JSON-string form vibe sometimes ships."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return cast("JsonDict", raw) if isinstance(raw, dict) else {}


def parse_plan(raw: object, *, plan_id: str = "") -> Plan:
    """Validate the arguments of a ``propose_plan`` call into a :class:`Plan`.

    Raises :class:`PlanError` listing every problem found, so a model that got
    several things wrong can fix them all at once rather than one per round trip.
    """
    args = _coerce(raw)
    problems: list[str] = []
    if not args:
        raise PlanError(["the plan is empty: give understanding and steps"])
    understanding = _text(args.get("understanding"), "understanding", problems, required=True)
    raw_steps = args.get("steps")
    steps: list[PlanStep] = []
    if isinstance(raw_steps, str):
        # A model that serialised the list itself; accept it once parsed.
        with contextlib.suppress(ValueError):
            raw_steps = json.loads(raw_steps)
    if not isinstance(raw_steps, list) or not raw_steps:
        problems.append("steps must be a non-empty list")
    else:
        step_list = cast("list[object]", raw_steps)
        if len(step_list) > MAX_STEPS:
            problems.append(
                f"too many steps ({len(step_list)}); keep a plan to {MAX_STEPS} or fewer"
            )
        for n, raw_step in enumerate(step_list[:MAX_STEPS], start=1):
            where = f"step {n}"
            if isinstance(raw_step, str):
                raw_step = {"action": raw_step}
            if not isinstance(raw_step, dict):
                problems.append(f"{where} must be an object with an action")
                continue
            step = cast("JsonDict", raw_step)
            tool = _text(step.get("tool"), f"{where} tool", problems)
            action = _text(step.get("action"), f"{where} action", problems) or tool
            if not action:
                problems.append(f"{where} needs an action (what it does, in plain words)")
            inputs_raw = step.get("inputs")
            inputs: JsonDict = {}
            if isinstance(inputs_raw, dict):
                inputs = cast("JsonDict", inputs_raw)
            elif inputs_raw not in (None, "", []):
                problems.append(f"{where} inputs must be an object of argument names to values")
            outputs = _string_list(step.get("outputs"), f"{where} outputs", problems)
            steps.append(PlanStep(action=action, tool=tool, inputs=inputs, outputs=outputs))
    output_dir = _text(args.get("output_dir"), "output_dir", problems)
    if output_dir and not output_dir.startswith("/"):
        problems.append("output_dir must be an absolute path")
    estimated_time = _text(args.get("estimated_time"), "estimated_time", problems)
    questions = _string_list(args.get("questions"), "questions", problems)
    if len(questions) > MAX_QUESTIONS:
        problems.append(f"too many questions; ask at most {MAX_QUESTIONS}")
    if problems:
        raise PlanError(problems)
    return Plan(
        id=plan_id,
        understanding=understanding,
        steps=tuple(steps),
        output_dir=output_dir,
        estimated_time=estimated_time,
        questions=questions,
    )


# ── Matching calls against an approved plan ──────────────────────────────────


def _norm(name: str) -> str:
    return harness.normalize_segment(name.strip()).lower()


def step_names_tool(step_tool: str, stack: str, tool: str) -> bool:
    """Whether a step's ``tool`` names *tool* of *stack*.

    The model may spell a tool several ways — ``<tool>``, ``<stack>_<tool>``,
    ``mcp_<stack>.<tool>``, ``tools.mcp_<stack>.<tool>`` (the stack's name with
    ``-`` as ``_``) — and all of them name the same call. A step that names a
    different stack does not match.
    """
    text = step_tool.strip()
    if not text or not tool:
        return False
    if "." in text:
        group, _, name = text.rpartition(".")
        group = group.removeprefix("tools.")
        if _norm(name) != _norm(tool):
            return False
        return _norm(group) in {_norm(stack), _norm(harness.mcp_group(stack))}
    if _norm(text) == _norm(tool):
        return True
    return _norm(text) == _norm(f"{stack}_{tool}")


def plan_key(raw: object) -> str:
    """A stable key for a ``propose_plan`` call's arguments.

    The core and the plan tool see the same arguments (one at the approval step,
    the other when the call runs), so a verdict the core reached can be looked up
    by the tool with this key. It is taken over the plan as validated — what the
    card shows — so a JSON-string ``steps``, blank optional fields, stray keys or
    surrounding whitespace on either side do not make the two disagree. Should they
    disagree anyway, the tool finds no verdict and says "not approved", never Go.
    """
    try:
        body = parse_plan(raw).to_json()
        body.pop("id", None)
    except PlanError:
        body = _coerce(raw)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resolve(value: str, workspace: Path) -> Path:
    raw = Path(value) if Path(value).is_absolute() else workspace / value
    return Path(os.path.normpath(str(raw)))


def _looks_like_path(key: str, value: str) -> bool:
    if not value or any(c.isspace() for c in value):
        return False
    if value.startswith(("/", "./", "../")):
        return True
    lowered = key.lower()
    return "/" in value and any(t in lowered for t in ("path", "dir", "file", "folder", "root"))


def _path_values(args: JsonDict) -> list[tuple[str, str]]:
    """``(name, value)`` for every path-looking string argument, lists included."""
    out: list[tuple[str, str]] = []
    for key, value in args.items():
        items: list[object] = cast("list[object]", value) if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and _looks_like_path(key, item):
                out.append((key, item))
    return out


def _anchors(plan: Plan, step: PlanStep, workspace: Path, *, output: bool) -> list[Path]:
    """The paths a step's calls may touch.

    Any argument may name one of the step's outputs, the plan's ``output_dir`` or
    an output listed in its inputs; only an argument that is not an output (by its
    name, see :func:`medmcp.pathcheck.classify_role`) may also name an input — so
    a Go never lets a tool write into the data it was shown reading.
    """
    raw: list[str] = [
        v for k, v in _path_values(step.inputs) if not output or classify_role(k) == "output"
    ]
    raw.extend(step.outputs)
    if plan.output_dir:
        raw.append(plan.output_dir)
    return [_resolve(v, workspace) for v in raw if v]


def _covered(path: Path, anchors: list[Path], workspace: Path) -> bool:
    """Whether *path* is one of *anchors* or lies inside one.

    The workspace root and ``/`` only ever cover themselves.
    """
    broad = {Path("/"), Path(os.path.normpath(str(workspace)))}
    for anchor in anchors:
        if path == anchor:
            return True
        if anchor not in broad and path.is_relative_to(anchor):
            return True
    return False


def _same_value(planned: object, actual: object, workspace: Path) -> bool:
    if isinstance(planned, str) and isinstance(actual, str):
        if _looks_like_path("path", planned) or _looks_like_path("path", actual):
            return _covered(_resolve(actual, workspace), [_resolve(planned, workspace)], workspace)
        return planned.strip().lower() == actual.strip().lower()
    if isinstance(planned, (int, float)) and isinstance(actual, (int, float)):
        return float(planned) == float(actual)
    return planned == actual


def call_fits_step(plan: Plan, step: PlanStep, args: JsonDict, workspace: Path) -> bool:
    """Whether a call's arguments stay within what *step* of *plan* showed.

    Every path argument must lie in one of the step's paths or the plan's
    ``output_dir`` (an output argument never in one of its inputs); an argument
    the step lists under the same name must hold the value it shows (a path there
    may be one inside it).
    """
    for key, value in _path_values(args):
        anchors = _anchors(plan, step, workspace, output=classify_role(key) == "output")
        if not _covered(_resolve(value, workspace), anchors, workspace):
            return False
    for key, planned in step.inputs.items():
        if key in args and not _same_value(planned, args[key], workspace):
            return False
    return True


def match_call(plan: Plan, stack: str, tool: str, args: JsonDict, workspace: Path) -> int | None:
    """The index of the first step of *plan* that covers this stack call, else ``None``.

    A step may cover several calls (a retry, or the same tool over the scans it
    names): what bounds it is the tool and the paths, not a count.
    """
    if stack == "builtin":
        return None
    for index, step in enumerate(plan.steps):
        if (
            step.tool
            and step_names_tool(step.tool, stack, tool)
            and call_fits_step(plan, step, args, workspace)
        ):
            return index
    return None


# ── The MCP server ───────────────────────────────────────────────────────────


def approval_text(plan: Plan) -> str:
    """What the tool returns once the person said Go."""
    lines = [
        "The user approved this plan (Go). Carry out exactly these steps, in order, "
        "without asking again; then give a short summary. If anything not listed here "
        "becomes necessary — another tool, any script, another output folder — stop "
        "and call propose_plan with a revised plan first.",
    ]
    for n, step in enumerate(plan.steps, start=1):
        tool = f" [{step.tool}]" if step.tool else ""
        lines.append(f"{n}. {step.action}{tool}")
    if plan.output_dir:
        lines.append(f"Results go into {plan.output_dir}.")
    return "\n".join(lines)


def _not_shown(problems: list[str]) -> str:
    return (
        "The plan could not be shown to the user. Fix these and call propose_plan "
        "again: " + "; ".join(problems)
    )


NOT_APPROVED_TEXT = (
    "The user has not approved this plan: it was not shown to them as a plan card, "
    "or no Go was recorded for it. Nothing is approved — do not carry out any of its "
    "steps. Call propose_plan again with your plan and wait for the user's answer."
)


def tool_result(arguments: object, verdict: Mapping[str, object] | None) -> tuple[str, bool]:
    """The text the tool returns for *arguments*, and whether it is an error.

    *verdict* is the core's word on this plan (broker op ``plan_verdict``):
    ``{"verdict": "go"}`` once the person said Go to exactly this plan's card,
    ``{"verdict": "refused", "problems": [...]}`` when the core would not show
    it. Only a Go gives the approval text. A plan the tool itself cannot read
    (:func:`parse_plan`) says what to fix; anything else — no verdict, no core
    to ask — is :data:`NOT_APPROVED_TEXT`.
    """
    try:
        plan = parse_plan(arguments)
    except PlanError as exc:
        return _not_shown(exc.problems), True
    kind = verdict.get("verdict") if verdict is not None else None
    if kind == VERDICT_GO:
        return approval_text(plan), False
    if kind == VERDICT_REFUSED:
        raw = verdict.get("problems") if verdict is not None else None
        problems = [str(p) for p in cast("list[object]", raw)] if isinstance(raw, list) else []
        return _not_shown(problems or ["the core refused it"]), True
    return NOT_APPROVED_TEXT, True


# The variable naming the broker socket for the plan tool: vibe hands its MCP
# servers only the variables an entry lists, so ``$VIBE_HOME`` is not among them.
BROKER_SOCK_ENV = "MEDMCP_BROKER_SOCK"


def _broker_socket() -> Path:
    """The core's broker socket, as an agent process finds it.

    ``$MEDMCP_BROKER_SOCK`` (set on the plan tool's config entry), else
    ``$VIBE_HOME``, else the checkout's ``.vibe``.
    """
    explicit = os.environ.get(BROKER_SOCK_ENV, "").strip()
    if explicit:
        return Path(explicit)
    home = os.environ.get("VIBE_HOME", "").strip()
    base = Path(home) if home else Path(__file__).resolve().parents[2] / ".vibe"
    return base / "backend.sock"


async def ask_core(op: str, payload: JsonDict, *, timeout: float = 5.0) -> JsonDict | None:
    """One request to the core's broker; ``None`` when it cannot be reached.

    Used by the plan tool (``plan_verdict``: did the person say Go to this plan,
    or did the core refuse to show it, and why); ``None`` is "not approved".
    """
    import asyncio

    path = _broker_socket()
    if not path.exists():
        return None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path), limit=32 * 1024 * 1024), timeout
        )
    except (OSError, TimeoutError):
        return None
    try:
        request = {"op": op, "stack": PLAN_SERVER, "id": 1, **payload}
        writer.write(json.dumps(request, default=str).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
    except (OSError, TimeoutError):
        return None
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    try:
        response = json.loads(line)
    except ValueError:
        return None
    if not isinstance(response, dict) or not cast("JsonDict", response).get("ok"):
        return None
    return cast("JsonDict", response)


async def core_verdict(arguments: object) -> JsonDict | None:
    """The core's verdict on this plan, taken (``None``: no core could be asked)."""
    return await ask_core("plan_verdict", {"key": plan_key(arguments)})


async def _serve() -> None:
    """Offer ``propose_plan`` over stdio until vibe disconnects."""
    import mcp.types as mcp_types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    server: Server[object, object] = Server(PLAN_SERVER)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:  # pyright: ignore[reportUnusedFunction]
        return [
            mcp_types.Tool(name=PLAN_TOOL, description=TOOL_DESCRIPTION, inputSchema=INPUT_SCHEMA)
        ]

    @server.call_tool(validate_input=False)
    async def _call_tool(  # pyright: ignore[reportUnusedFunction]
        name: str, arguments: JsonDict
    ) -> mcp_types.CallToolResult:
        if name != PLAN_TOOL:
            text, is_error = f"unknown tool {name!r}", True
        else:
            text, is_error = tool_result(arguments, await core_verdict(arguments))
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=text)], isError=is_error
        )

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Entry point of ``medmcp-plan-tool``: the MCP server vibe spawns."""
    import anyio

    anyio.run(_serve)
