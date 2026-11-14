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
(:func:`match_call`, :func:`match_mkdir`). The server (``server.py``) wires it
into the permission flow; :func:`main` is the MCP server that offers the tool.

What a Go covers, and what it never covers:

* A **stack tool call** whose tool is named by a step, and every one of whose
  path arguments lies in a path the plan shows: one of the step's inputs or
  outputs, or the plan's ``output_dir`` (or inside one of them). An argument the
  step lists under the same name must hold the value the step shows.
* ``mkdir -p`` of folders inside the plan's ``output_dir`` — the agent is told to
  create its output folder before the first call writes to it.
* Never **code the assistant wrote** (a script, an interpreter, ``write_file`` of
  a ``.py``): that is put to the person on its own, plan or no plan.
* Never ``confirm_selection`` (the cohort stack): confirming a selection is the
  person's decision about a review they have seen, so it always opens its own
  dialog, which shows that review (:mod:`medmcp.reviewsummary`).
* Never anything else: a call outside the plan goes to the ordinary approval
  dialog, flagged as not in the plan. From the third such call in one plan the
  dialog suggests asking for a revised plan, and the call's result reminds the
  agent to propose one (:func:`off_plan_reminder`).

A plan whose steps name a tool no stack offers, or a parameter its tool does not
take, is not shown: :func:`check_tools` lists the problems and the tool hands them
back with the parameters the tool does take.

The workspace root and the filesystem root never cover what lies under them — a
plan naming the whole workspace as an input would otherwise approve everything.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import shlex
import time
from collections.abc import Callable, Mapping
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

# The cohort stack's tools around a selection review. confirm_selection is never
# covered by a Go (see the module doc); a plan never holds both, because the review
# is what the person decides on, and they decide after reading it.
REVIEW_TOOL = "review_selection"
CONFIRM_TOOL = "confirm_selection"

# From this many calls outside one plan on, the dialog suggests a revised plan and
# the allowed call's result reminds the agent to propose one.
OFF_PLAN_NUDGE_FROM = 3

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
                            "The tool this step calls (e.g. 'index_dataset'); "
                            "omit for a step that calls none."
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
    named = [st.tool for st in steps if st.tool]
    if any(_names(t, REVIEW_TOOL) for t in named) and any(_names(t, CONFIRM_TOOL) for t in named):
        problems.append(
            f"{REVIEW_TOOL} and {CONFIRM_TOOL} cannot be in the same plan: end this plan "
            "with the review, show it to the user and stop; confirm only after the user "
            "has answered, in a plan of its own"
        )
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

    The model may spell a tool several ways — ``index_dataset``,
    ``medmcp-cohort_index_dataset``, ``mcp_medmcp_cohort.index_dataset``,
    ``tools.mcp_medmcp_cohort.index_dataset`` — and all of them name the same
    call. A step that names a different stack does not match.
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


def _bare_tool(step_tool: str) -> str:
    """The tool part of a step's ``tool``, whatever the spelling (``a.b`` → ``b``)."""
    text = step_tool.strip()
    if "." in text:
        text = text.rpartition(".")[2]
    return _norm(text)


def _names(step_tool: str, tool: str) -> bool:
    """Whether a step's ``tool`` names *tool* of whichever stack."""
    bare = _bare_tool(step_tool)
    return bare == tool or bare.endswith(f"_{tool}")


def is_confirm_selection(stack: str, tool: str) -> bool:
    """Whether a stack call is ``confirm_selection`` — never covered by a Go."""
    return stack != "builtin" and _norm(tool) == CONFIRM_TOOL


# ── Checking a plan against the tools the stacks offer ───────────────────────

# Names a step may give that are not stack tools: vibe's own tools (a plan may
# list a shell step; it is never covered by the Go) and the harness's.
_BUILTIN_STEP_TOOLS: frozenset[str] = frozenset(
    {
        *harness.BUILTIN_ROUTES,
        *harness.BUILTIN_ROUTES.values(),
        *harness.HARNESS_TOOLS,
        "grep",
        "todo",
        "edit",
        "task",
        "skill",
        "web_fetch",
        "ask_user_question",
        "mkdir",
    }
)

# Stack tool name → its input schema, per stack: ``{stack: {tool: schema}}``.
ToolSchemas = Mapping[str, Mapping[str, JsonDict]]


def _parameters(schema: JsonDict) -> list[str] | None:
    """The parameter names *schema* allows, or ``None`` when it allows any."""
    if schema.get("additionalProperties") is True:
        return None
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    return list(cast("JsonDict", props))


def check_tools(plan: Plan, schemas: ToolSchemas, *, complete: bool) -> list[str]:
    """What is wrong with *plan*'s steps given the tools the stacks offer.

    A step must name a tool one of the stacks offers (or one of vibe's own), and
    every argument it lists must be a parameter of that tool; each problem lists
    the parameters the tool does take, so the model can fix it in one go. An
    unknown tool name is reported only when *complete* — when *schemas* holds
    every stack of the chat; otherwise a tool not found may be one whose list is
    not known yet, and only the steps with a known tool are checked.
    """
    problems: list[str] = []
    for n, step in enumerate(plan.steps, start=1):
        if not step.tool or step.tool.strip().removeprefix("tools.") in _BUILTIN_STEP_TOOLS:
            continue
        found = [
            (stack, name, schema)
            for stack, tools in schemas.items()
            for name, schema in tools.items()
            if step_names_tool(step.tool, stack, name)
        ]
        if not found:
            if complete:
                offered = sorted({name for tools in schemas.values() for name in tools})
                close = difflib.get_close_matches(_bare_tool(step.tool), offered, n=3, cutoff=0.5)
                hint = (
                    f"did you mean {', '.join(close)}?"
                    if close
                    else "name one of the tools your stacks offer"
                )
                problems.append(f"step {n}: no stack offers a tool {step.tool!r}; {hint}")
            continue
        _stack, name, schema = found[0]
        allowed = _parameters(schema)
        if allowed is None:
            continue
        unknown = [key for key in step.inputs if key not in allowed]
        if unknown:
            listed = ", ".join(allowed) if allowed else "none"
            problems.append(
                f"step {n} ({name}): {name} has no parameter "
                + ", ".join(repr(k) for k in unknown)
                + f"; its parameters are: {listed}"
            )
    return problems


def plan_key(raw: object) -> str:
    """A stable key for a ``propose_plan`` call's arguments.

    The core and the plan tool see the same arguments (one at the approval step,
    the other when the call runs), so a verdict the core reached can be looked up
    by the tool with this key.
    """
    canonical = json.dumps(_coerce(raw), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def off_plan_reminder(count: int) -> str:
    """What an allowed call outside the plan carries back to the model, from the third on."""
    return (
        f"[MedMCP] This was call {count} outside the plan the user approved. Before any "
        "further step that is not in the plan, call propose_plan with a revised plan "
        "and wait for the user's Go."
    )


def note_key(name: str, args: JsonDict) -> str:
    """The key a call is known by in :class:`PlanNotes`: its tool and arguments.

    A shell command is known by its command alone, so the hook (which sees the
    call as the harness hands it over) and the core (which saw it at the approval
    step) agree whatever else rides along.
    """
    command = args.get("command")
    if name == "bash" and isinstance(command, str):
        return json.dumps(["bash", command])
    return json.dumps([name, args], sort_keys=True, separators=(",", ":"), default=str)


class PlanNotes:
    """What to add to the results of calls allowed outside the plan, until taken.

    The core adds a note when the person allows such a call; whatever returns the
    call's result to the model takes it — the broker for a stack call, the
    plan-note hook (``medmcp-plan-note``) for one of vibe's own tools. A note
    nobody took expires.
    """

    def __init__(self, ttl_sec: float = 900.0, clock: Callable[[], float] = time.monotonic) -> None:
        """Notes expire *ttl_sec* after they are added."""
        self._ttl = ttl_sec
        self._clock = clock
        self._notes: dict[str, list[tuple[float, str]]] = {}

    def add(self, name: str, args: JsonDict, text: str) -> None:
        """Add *text* to the next result of the call *name* with *args*."""
        self._notes.setdefault(note_key(name, args), []).append((self._clock(), text))

    def take(self, name: str, args: JsonDict) -> str | None:
        """The note for this call, removed; ``None`` when there is none."""
        key = note_key(name, args)
        now = self._clock()
        live = [n for n in self._notes.get(key, []) if now - n[0] < self._ttl]
        if not live:
            self._notes.pop(key, None)
            return None
        _, text = live.pop(0)
        if live:
            self._notes[key] = live
        else:
            self._notes.pop(key, None)
        return text


def off_plan_reason(count: int) -> str:
    """The dialog's note on a call outside the approved plan (*count*: its number)."""
    if count < OFF_PLAN_NUDGE_FROM:
        return "Not in the plan you approved."
    return (
        f"Not in the plan you approved — call {count} outside it. Ask the assistant "
        "to revise the plan: deny this and say so in the chat."
    )


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


_SHELL_META = frozenset(";&|<>`$(){}*?[]~!#\\\"'\n")


def match_mkdir(plan: Plan, command: str, workspace: Path) -> bool:
    """Whether *command* is a plain ``mkdir -p`` of folders inside the plan's output_dir.

    Anything with shell syntax beyond words — a pipe, a redirect, a substitution,
    a glob, a quote — is not matched, and goes to the dialog.
    """
    if not plan.output_dir or not command.strip() or any(c in _SHELL_META for c in command):
        return False
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    if len(words) < 3 or words[0] != "mkdir" or words[1] not in {"-p", "--parents"}:
        return False
    anchor = _resolve(plan.output_dir, workspace)
    root = Path(os.path.normpath(str(workspace)))
    if anchor == root or not anchor.is_relative_to(root):
        return False
    return all(
        not w.startswith("-") and _covered(_resolve(w, workspace), [anchor], workspace)
        for w in words[2:]
    )


# ── The MCP server ───────────────────────────────────────────────────────────


def approval_text(plan: Plan) -> str:
    """What the tool returns once the person said Go."""
    lines = [
        "The user approved this plan (Go). Carry out exactly these steps, in order, "
        "without asking again; then give a short summary. If anything not listed here "
        "becomes necessary — another tool, any script, another output folder — stop "
        "and call propose_plan with a revised plan first. confirm_selection is never "
        "part of a Go: the user approves it on its own, after reading the review.",
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


def tool_result(arguments: object, core_problems: list[str] | None = None) -> tuple[str, bool]:
    """The text the tool returns for *arguments*, and whether it is an error.

    The tool only runs once the person said Go — the core holds the call at the
    approval step until then — or when the plan could not be shown, in which case
    it says what to fix: its own problems (:func:`parse_plan`) or those the core
    found checking the steps against the stacks' tools (*core_problems*).
    """
    try:
        plan = parse_plan(arguments)
    except PlanError as exc:
        return _not_shown(exc.problems), True
    if core_problems:
        return _not_shown(core_problems), True
    return approval_text(plan), False


def _broker_socket() -> Path:
    """The core's broker socket, as an agent process finds it (``$VIBE_HOME``)."""
    home = os.environ.get("VIBE_HOME", "").strip()
    base = Path(home) if home else Path(__file__).resolve().parents[2] / ".vibe"
    return base / "backend.sock"


async def ask_core(op: str, payload: JsonDict, *, timeout: float = 5.0) -> JsonDict | None:
    """One request to the core's broker; ``None`` when it cannot be reached.

    Used by the plan tool (``check_plan``: did the core refuse to show this plan,
    and why) and by the plan-note hook (``plan_note``). A core without a broker
    (no stack pool) checks no schemas and keeps no notes, so ``None`` is the
    answer "nothing to add".
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


async def core_problems(arguments: object) -> list[str]:
    """The problems the core found with this plan (``[]``: none, or no core to ask)."""
    response = await ask_core("check_plan", {"key": plan_key(arguments)})
    if response is None:
        return []
    problems = response.get("problems")
    if not isinstance(problems, list):
        return []
    return [str(p) for p in cast("list[object]", problems)]


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
            text, is_error = tool_result(arguments, await core_problems(arguments))
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=text)], isError=is_error
        )

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Entry point of ``medmcp-plan-tool``: the MCP server vibe spawns."""
    import anyio

    anyio.run(_serve)
