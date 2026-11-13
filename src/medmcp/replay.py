"""Deterministic replay of a distilled workflow on new data (no LLM).

Loads a :class:`~medmcp.workflow.Recipe`, binds new values to its ``{{in_N}}``
inputs, and runs each step by calling the MCP tool directly over stdio — no
vibe-acp, no model reasoning. An output a step produces (its ``produces`` keys)
is captured from the tool's structured result and substituted into later steps'
``{{stepM.<key>}}`` placeholders, so a multi-step pipeline chains exactly as it
did when it was recorded.

This module has no UI/vibe-acp dependency: callers pass in the resolved
MCP server launch configs (``command``/``args``/``env`` as produced by
``settings.load_mcp_servers``), so it can be driven from the UI or a CLI alike.

Replaying runs real tools with real side effects (file writes, etc.). It is
deliberately separate from the chat permission flow: callers are expected to
confirm with the user *before* calling :func:`run` (the UI previews the resolved
steps first). Built-in vibe-acp tools (``builtin:*`` such as ``bash``) are not
MCP tools and cannot be replayed; :func:`validate` reports them rather than
silently skipping a step the pipeline depends on.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, cast

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from medmcp import pathcheck, workflow
from medmcp.toolresult import looks_failed, parse_structured_blob
from medmcp.workflow import Recipe

JsonDict = dict[str, Any]

# A callable that runs one tool and returns (ok, structured_output, error_text).
ToolCaller = Callable[[str, str, JsonDict], Awaitable[tuple[bool, JsonDict, "str | None"]]]

# Notified just before a step's tool is called: ``(step_index, server, tool)``.
StepStartHook = Callable[[int, str, str], Awaitable[None]]

_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")
# A derived reference: ``{{dir(<ref>)}}`` is the directory holding whatever
# ``<ref>`` resolves to. Distillation emits these instead of declaring a second
# input for a folder the caller has already implied — asking for a file and then
# for the directory that file sits in is one piece of information, not two, and
# on replay the outputs should follow the new file rather than the old folder.
_DERIVED_DIR_RE = re.compile(r"dir\((.+)\)")


def _resolve_ref(ref: str, bindings: dict[str, Any]) -> object | None:
    """Resolve one placeholder ref; ``None`` when it cannot be resolved.

    ``None`` rather than a sentinel string so an unresolvable ref can be left
    verbatim by the caller, which is how :func:`unresolved_refs` still sees it.
    """
    derived = _DERIVED_DIR_RE.fullmatch(ref)
    if derived is not None:
        inner = _resolve_ref(derived.group(1).strip(), bindings)
        return None if inner is None else str(PurePosixPath(str(inner)).parent)
    return bindings.get(ref)


# Imaging tools (registration, segmentation) can run for minutes.
DEFAULT_TOOL_TIMEOUT_SEC: float = 900.0


class ReplayError(Exception):
    """Raised for engine-level failures (missing stack, bad recipe)."""


@dataclass
class StepResult:
    """Outcome of replaying a single recipe step."""

    index: int
    server: str
    tool: str
    arguments: JsonDict
    ok: bool
    output: JsonDict | None = None
    error: str | None = None
    produced: dict[str, str] = field(default_factory=dict[str, str])


@dataclass
class ReplayResult:
    """Aggregate outcome of a replay run."""

    steps: list[StepResult] = field(default_factory=list[StepResult])
    ok: bool = True
    error: str | None = None


# ── Placeholder resolution ────────────────────────────────────────────────────


def resolve_value(value: object, bindings: dict[str, Any]) -> object:
    """Substitute ``{{ref}}`` placeholders in *value* using *bindings*.

    A string that is exactly one placeholder is replaced by the bound value with
    its original type preserved (so a path stays a str, a bound number stays a
    number). Placeholders embedded in a larger string, or nested in dict/list
    values, are substituted as text. Unknown refs are left verbatim so the caller
    can detect them via :func:`unresolved_refs`.
    """
    if isinstance(value, str):
        whole = _PLACEHOLDER_RE.fullmatch(value.strip())
        if whole is not None:
            resolved = _resolve_ref(whole.group(1).strip(), bindings)
            return value if resolved is None else resolved

        def _sub(match: re.Match[str]) -> str:
            resolved = _resolve_ref(match.group(1).strip(), bindings)
            return match.group(0) if resolved is None else str(resolved)

        return _PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: resolve_value(v, bindings) for k, v in cast("JsonDict", value).items()}
    if isinstance(value, list):
        return [resolve_value(v, bindings) for v in cast("list[Any]", value)]
    return value


def resolve_arguments(arguments: JsonDict, bindings: dict[str, Any]) -> JsonDict:
    """Resolve every argument value against *bindings*."""
    return {k: resolve_value(v, bindings) for k, v in arguments.items()}


def unresolved_refs(value: object) -> set[str]:
    """Return any ``{{ref}}`` placeholders still present in *value* (recursively)."""
    found: set[str] = set()
    if isinstance(value, str):
        found.update(m.group(1).strip() for m in _PLACEHOLDER_RE.finditer(value))
    elif isinstance(value, dict):
        for v in cast("JsonDict", value).values():
            found |= unresolved_refs(v)
    elif isinstance(value, list):
        for v in cast("list[Any]", value):
            found |= unresolved_refs(v)
    return found


# ── Tool-result interpretation ────────────────────────────────────────────────


def _result_text(result: mcp_types.CallToolResult) -> str:
    """Join the text blocks of a tool result."""
    return "\n".join(
        block.text for block in result.content if isinstance(block, mcp_types.TextContent)
    )


def extract_structured(result: mcp_types.CallToolResult) -> JsonDict:
    """Extract a tool's structured output as a dict.

    Tries, in order:

    1. the protocol-native ``structuredContent`` (newer MCP/stacks with an output
       schema), unwrapping a lone ``{"result": {...}}`` envelope;
    2. parsing the text content as JSON (FastMCP returns a tool's dict as a JSON
       text block);
    3. a ``structured: {...}`` blob in the text — the shape distillation recorded
       ``produces`` keys from — so recipes distilled from vibe's rendered results
       still chain.

    All three surface the same top-level keys, so a recipe's ``produces`` keys
    resolve regardless of which the stack emits.
    """
    structured = result.structuredContent
    if isinstance(structured, dict) and structured:
        inner = structured.get("result")
        if len(structured) == 1 and isinstance(inner, dict):
            return cast("JsonDict", inner)
        return structured

    text = _result_text(result).strip()
    if text:
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            parsed_json = None
        if isinstance(parsed_json, dict):
            return cast("JsonDict", parsed_json)

    return parse_structured_blob(text)


def _result_failed(result: mcp_types.CallToolResult) -> bool:
    """Detect a failed tool call (protocol error flag or known failure markers)."""
    # Defense-in-depth for stacks that report a failure as a non-error result
    # (text only); the markers are shared with distillation so a recipe and its
    # replay agree on failure.
    return result.isError or looks_failed(_result_text(result))


# ── MCP transport ─────────────────────────────────────────────────────────────


def _server_params(cfg: JsonDict, *, cwd: str | None) -> StdioServerParameters:
    """Build stdio launch parameters from a discovered server config."""
    raw_env = cfg.get("env")
    env: dict[str, str] = {**os.environ}
    if isinstance(raw_env, dict):
        env.update({str(k): str(v) for k, v in cast("JsonDict", raw_env).items()})
    return StdioServerParameters(
        command=str(cfg["command"]),
        args=[str(a) for a in cast("list[Any]", cfg.get("args", []))],
        env=env,
        cwd=cwd,
    )


@asynccontextmanager
async def mcp_caller(
    servers: list[JsonDict],
    *,
    cwd: str | None = None,
    tool_timeout_sec: float = DEFAULT_TOOL_TIMEOUT_SEC,
) -> AsyncGenerator[ToolCaller]:
    """Yield a :data:`ToolCaller` backed by lazily-spawned MCP stdio servers.

    Each needed stack's server is started on first use and reused for the rest of
    the run. A transport error (the server process dying) evicts that server, so
    the next call re-spawns a fresh one rather than reusing a broken session — a
    crash on one batch item does not sink the items that follow. Every live server
    is shut down when the context exits (including on cancellation/Stop).
    """
    server_map = {str(s["name"]): s for s in servers}
    timeout = timedelta(seconds=tool_timeout_sec)

    async with AsyncExitStack() as stack:
        # Each server keeps its own exit stack so a dead one can be torn down
        # individually, without disturbing the others.
        sessions: dict[str, ClientSession] = {}
        server_stacks: dict[str, AsyncExitStack] = {}

        async def _evict(server: str) -> None:
            sessions.pop(server, None)
            sstack = server_stacks.pop(server, None)
            if sstack is not None:
                with suppress(Exception):
                    await sstack.aclose()

        async def _evict_all() -> None:
            # Newest first: each stdio client holds an anyio cancel scope, and closing
            # an older one while a newer one is open raises CancelledError.
            for server in reversed(list(server_stacks)):
                await _evict(server)

        # Shut every live server down on exit (normal or cancellation/Stop).
        stack.push_async_callback(_evict_all)

        async def _session(server: str) -> ClientSession:
            if server not in sessions:
                cfg = server_map.get(server)
                if cfg is None:
                    raise ReplayError(f"stack {server!r} is not installed")
                sstack = AsyncExitStack()
                try:
                    read, write = await sstack.enter_async_context(
                        stdio_client(_server_params(cfg, cwd=cwd))
                    )
                    session = await sstack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                except BaseException:
                    await sstack.aclose()  # don't leak a half-started server
                    raise
                server_stacks[server] = sstack
                sessions[server] = session
            return sessions[server]

        async def _call(
            server: str, tool: str, args: JsonDict
        ) -> tuple[bool, JsonDict, str | None]:
            session = await _session(server)
            try:
                result = await session.call_tool(tool, args, read_timeout_seconds=timeout)
            except Exception:
                # Transport-level failure: the server process is probably dead, so
                # drop it — the next call spawns a fresh one instead of reusing a
                # broken session. (A tool that merely *reports* failure returns a
                # result and is handled below; its session stays healthy.)
                await _evict(server)
                raise
            structured = extract_structured(result)
            if _result_failed(result):
                return False, structured, _result_text(result) or "tool reported failure"
            return True, structured, None

        yield _call


# ── Validation & execution ────────────────────────────────────────────────────


def paths_outside(value: object, root: Path) -> list[str]:
    """Strings in a resolved argument tree that name a path outside *root*.

    A string counts as a path when it is absolute, starts with ``~``, or has a
    ``..`` component; anything else is relative to the stack's working
    directory, which replay sets to the workspace. Bound inputs are confined
    before they get here, but a recipe's constants, an input default, a ``..``
    appended to a placeholder and a value a previous step produced are not, and
    a recipe can be imported from anyone.
    """
    found: list[str] = []
    if isinstance(value, dict):
        for item in cast("JsonDict", value).values():
            found += paths_outside(item, root)
    elif isinstance(value, list):
        for item in cast("list[Any]", value):
            found += paths_outside(item, root)
    elif isinstance(value, str) and "://" not in value and "\n" not in value:
        text = value.strip()
        if not (text.startswith(("/", "~")) or ".." in PurePosixPath(text).parts):
            return found
        path = Path(text).expanduser()
        resolved = (path if path.is_absolute() else root / path).resolve()
        if not resolved.is_relative_to(root.resolve()):
            found.append(text)
    return found


# ── Outputs must never land on inputs ─────────────────────────────────────────
#
# Replay has no model in the loop and no per-call approval, so a binding that
# points an output argument at the file it was read from would overwrite source
# data with nothing between the mistake and the write but the tool's own manners.
# These checks are the engine's, not the tool's: a run that would do that is
# refused before any step is called.


@dataclass(frozen=True)
class PathUse:
    """One path a recipe hands to a tool, with what the tool does with it."""

    param: str
    path: str
    role: pathcheck.PathRole
    kind: workflow.OutputKind
    """For an output: a folder results go into, or a file that is written."""


def _normalised(path: str, base: Path | None) -> PurePosixPath:
    """*path* made absolute against *base* and lexically normalised (no disk access)."""
    text = os.path.expanduser(path.strip())
    if not os.path.isabs(text) and base is not None:
        text = os.path.join(str(base), text)
    return PurePosixPath(os.path.normpath(text))


def path_uses(recipe: Recipe, inputs: dict[str, Any]) -> list[PathUse]:
    """Every path the recipe would pass to a tool for *inputs*, in step order.

    Only arguments that are fully known before the run count: one that still
    holds a ``{{stepM.key}}`` reference is a previous step's output, decided at
    run time, and is checked there by the step that produced it.
    """
    bound = apply_input_defaults(recipe, inputs)
    uses: list[PathUse] = []
    for step in recipe.steps:
        for key, raw in step.arguments.items():
            value = resolve_value(raw, bound)
            if unresolved_refs(value):
                continue
            role = pathcheck.classify_role(key)
            for text in _strings(value):
                if pathcheck.looks_like_path(text) and "://" not in text:
                    uses.append(PathUse(key, text, role, workflow.output_kind(key)))
    return uses


def _strings(value: object) -> list[str]:
    """Every string inside a (possibly nested) argument value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in cast("JsonDict", value).values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in cast("list[Any]", value) for s in _strings(v)]
    return []


def _is_file_path(path: PurePosixPath) -> bool:
    """Whether *path* is a file: one on disk, or — when absent — a name with a suffix."""
    real = Path(path)
    try:
        if real.exists():
            return real.is_file()
    except OSError:
        pass
    return bool(path.suffix)


def output_collision(
    recipe: Recipe, inputs: dict[str, Any], *, base: Path | None = None
) -> str | None:
    """Why this binding would write over what it reads, or ``None`` when it would not.

    Refused:

    - an output **file** that is one of the inputs (the binding that prompted
      this check: ``output_path`` bound to the scan itself);
    - an output file that is a folder holding an input, or that lies beneath an
      input file — two paths that cannot both exist as named;
    - an output **folder** that is, or lies beneath, an input *file*.

    Allowed, deliberately: an output folder that holds the input, or is the
    input folder. "Write the result next to the scan" is the ordinary case (it is
    what a distilled ``output_dir`` defaults to), and a path alone cannot tell
    whether a tool writing into a folder reuses a name already in it.
    """
    uses = path_uses(recipe, inputs)
    outputs = [u for u in uses if u.role == "output"]
    sources = [u for u in uses if u.role != "output"]
    for out in outputs:
        out_path = _normalised(out.path, base)
        for src in sources:
            src_path = _normalised(src.path, base)
            src_is_file = _is_file_path(src_path)
            if out_path == src_path:
                if out.kind == "dir" and not src_is_file:
                    continue
                return (
                    f"{out.param} is the same path as the input {src.param} "
                    f"({src.path}): the run would overwrite its own input"
                )
            if src_is_file and src_path in out_path.parents:
                return (
                    f"{out.param} ({out.path}) lies beneath the input file {src.param} ({src.path})"
                )
            if out.kind == "file" and out_path in src_path.parents:
                return (
                    f"{out.param} ({out.path}) is written as a file but is the folder "
                    f"that holds the input {src.param}"
                )
    return None


def batch_output_collisions(
    recipe: Recipe, runs: list[dict[str, Any]], *, base: Path | None = None
) -> dict[int, str]:
    """Items of a batch that would write to the same place as another item.

    Returns ``{item index: reason}`` for **every** item involved, so none of them
    runs until the person has told them apart.

    Two items writing the same output *file* always collide. Two items sharing
    an output *folder* collide when they also read a file of the same name: a
    tool names its results after its input, so ``FLAIR.nii.gz`` of two subjects
    into one folder is one result overwriting the other. Items that share a
    folder but read differently named files are left alone — a flat folder of
    scans with the results beside them is a layout people use.
    """
    files: dict[PurePosixPath, list[int]] = {}
    folders: dict[PurePosixPath, list[int]] = {}
    names: list[set[str]] = []
    for index, inputs in enumerate(runs):
        uses = path_uses(recipe, inputs)
        names.append({_normalised(u.path, base).name for u in uses if u.role != "output"} - {""})
        seen: set[PurePosixPath] = set()
        for use in uses:
            if use.role != "output":
                continue
            path = _normalised(use.path, base)
            if path in seen:
                continue
            seen.add(path)
            (folders if use.kind == "dir" else files).setdefault(path, []).append(index)

    found: dict[int, str] = {}
    for path, items in files.items():
        if len(items) > 1:
            for index in items:
                other = next(i for i in items if i != index)
                found.setdefault(index, f"writes the same output file as item {other + 1}: {path}")
    for path, items in folders.items():
        for index in items:
            clash = next((o for o in items if o != index and names[index] & names[o]), None)
            if clash is not None:
                shared = sorted(names[index] & names[clash])[0]
                found.setdefault(
                    index,
                    f"writes into the same folder as item {clash + 1} ({path}) and both "
                    f"read a file named {shared}: one result would overwrite the other",
                )
    return found


def ensure_output_folders(
    step_arguments: JsonDict, resolved: JsonDict, *, confine_to: Path | None
) -> None:
    """Create the folders a step's output arguments need, before the tool runs.

    In a chat the agent makes the folder first (a ``mkdir`` the recipe cannot
    replay); on a roll-out nobody does, and thirty subjects would mean thirty
    folders made by hand. An output *folder* is created itself, an output *file*
    gets its parent. Only inside *confine_to* when one is given, and best-effort:
    a folder that cannot be made is left for the tool to report in its own words.
    """
    for key in step_arguments:
        if pathcheck.classify_role(key) != "output":
            continue
        for text in _strings(resolved.get(key)):
            if not pathcheck.looks_like_path(text) or "://" in text:
                continue
            path = Path(text).expanduser()
            if not path.is_absolute():
                if confine_to is None:
                    continue
                path = confine_to / path
            folder = path if workflow.output_kind(key) == "dir" else path.parent
            if confine_to is not None and not folder.resolve().is_relative_to(confine_to.resolve()):
                continue
            with suppress(OSError):
                folder.mkdir(parents=True, exist_ok=True)


def _is_unset(value: object) -> bool:
    """Whether *value* counts as not supplied: absent, or whitespace-only text."""
    return value is None or (isinstance(value, str) and not value.strip())


def apply_input_defaults(recipe: Recipe, inputs: dict[str, Any]) -> dict[str, Any]:
    """Return *inputs* with any unbound input filled in from its declared default.

    A default is a placeholder expression that may reference other inputs (see
    :class:`~medmcp.workflow.WorkflowInput`), so it is resolved against the
    bindings accumulated so far, in declaration order. A default whose anchor is
    itself unbound is skipped rather than substituted half-resolved, leaving the
    input genuinely missing so :func:`validate` reports it.

    An explicitly supplied value always wins: the default exists to save typing,
    not to override a caller who pointed the workflow somewhere else.
    """
    # A blank is not a value. The run form posts every field it rendered, so an
    # input the caller deliberately left empty arrives as "" — and treating that
    # as bound skipped the default and sent an empty output_dir to the tool,
    # which then wrote relative to its own working directory, outside the
    # workspace: the run reported success and no files appeared. Dropping blanks
    # here also means a blank *required* input reaches validate() as missing,
    # failing loudly rather than writing somewhere nobody will look.
    bound: dict[str, Any] = {k: v for k, v in inputs.items() if not _is_unset(v)}
    for inp in recipe.inputs:
        if inp.name in bound or not inp.default:
            continue
        resolved = resolve_value(inp.default, bound)
        if unresolved_refs(resolved):
            continue
        bound[inp.name] = resolved
    return bound


def validate(
    recipe: Recipe,
    inputs: dict[str, Any],
    servers: list[JsonDict],
    *,
    base: Path | None = None,
) -> str | None:
    """Return a human-readable error if *recipe* can't be replayed, else ``None``.

    Checks that every declared input has a value, every referenced stack is
    installed, that the recipe has no built-in (non-MCP) tool steps, and that no
    output would be written over an input (:func:`output_collision`; *base* is
    what a relative path is read against, the workspace).
    """
    bound = apply_input_defaults(recipe, inputs)
    missing_inputs = [i.name for i in recipe.inputs if i.name not in bound]
    if missing_inputs:
        return f"missing values for inputs: {', '.join(missing_inputs)}"

    builtin = sorted({f"{s.server}:{s.tool}" for s in recipe.steps if s.server == "builtin"})
    if builtin:
        return (
            "this workflow uses built-in tools that the replay engine can't run "
            f"deterministically: {', '.join(builtin)}"
        )

    installed = {str(s["name"]) for s in servers}
    needed = sorted({s.server for s in recipe.steps} - installed)
    if needed:
        # Make the message actionable: name each missing stack with the image or
        # version it's pinned to in the recipe's requirements, so a colleague who
        # received this workflow knows exactly what to install.
        pins = {
            r.stack: (r.image or (f"v{r.version}" if r.version else "")) for r in recipe.requires
        }
        labels = [f"{s} ({pins[s]})" if pins.get(s) else s for s in needed]
        return f"required stack(s) not installed: {', '.join(labels)}"

    if not recipe.steps:
        return "this workflow has no replayable steps"
    return output_collision(recipe, inputs, base=base)


async def replay_with_caller(
    recipe: Recipe,
    inputs: dict[str, Any],
    *,
    caller: ToolCaller,
    on_step: Callable[[StepResult], Awaitable[None]] | None = None,
    on_step_start: StepStartHook | None = None,
    confine_to: Path | None = None,
) -> ReplayResult:
    """Execute *recipe* step-by-step using *caller*; abort on the first failure.

    *inputs* maps placeholder names (``in_1`` …) to concrete values. Outputs a
    step produces are captured and bound as ``{{stepM.<key>}}`` for later steps.
    ``on_step_start`` fires just before each tool call, so a caller can show
    which step is in flight. With *confine_to*, a step whose arguments name a
    path outside it fails instead of running (see :func:`paths_outside`).
    """
    # Defaults are filled here, the one place every replay path funnels through
    # (single run and each batch item alike), so no caller can forget them.
    bindings: dict[str, Any] = apply_input_defaults(recipe, inputs)
    result = ReplayResult()

    for index, step in enumerate(recipe.steps, start=1):
        args = resolve_arguments(step.arguments, bindings)
        pending = unresolved_refs(args)
        outside = paths_outside(args, confine_to) if confine_to is not None else []
        if pending or outside:
            error = (
                f"unresolved placeholders: {', '.join(sorted(pending))}"
                if pending
                else f"path outside the workspace: {', '.join(outside)}"
            )
            step_result = StepResult(
                index=index,
                server=step.server,
                tool=step.tool,
                arguments=args,
                ok=False,
                error=error,
            )
            result.steps.append(step_result)
            result.ok = False
            result.error = step_result.error
            if on_step is not None:
                await on_step(step_result)
            break

        if on_step_start is not None:
            await on_step_start(index, step.server, step.tool)
        ensure_output_folders(step.arguments, args, confine_to=confine_to)
        try:
            ok, structured, error = await caller(step.server, step.tool, args)
        except Exception as exc:
            # Surface any transport/tool error as a failed step rather than
            # crashing the whole replay.
            ok, structured, error = False, {}, str(exc)

        produced: dict[str, str] = {}
        if ok:
            for out_key, ref in step.produces.items():
                if out_key in structured:
                    bindings[ref] = structured[out_key]
                    produced[ref] = str(structured[out_key])

        step_result = StepResult(
            index=index,
            server=step.server,
            tool=step.tool,
            arguments=args,
            ok=ok,
            output=structured or None,
            error=error,
            produced=produced,
        )
        result.steps.append(step_result)
        if on_step is not None:
            await on_step(step_result)

        if not ok:
            result.ok = False
            result.error = error or f"step {index} ({step.server}:{step.tool}) failed"
            break

    return result


async def run(
    recipe: Recipe,
    inputs: dict[str, Any],
    *,
    servers: list[JsonDict],
    cwd: str | None = None,
    tool_timeout_sec: float = DEFAULT_TOOL_TIMEOUT_SEC,
    on_step: Callable[[StepResult], Awaitable[None]] | None = None,
    confine_to: Path | None = None,
) -> ReplayResult:
    """Validate and replay *recipe* on *inputs*, spawning the needed MCP stacks.

    Returns a :class:`ReplayResult`; an engine-level problem (failed validation,
    a stack that won't start) is reported via ``ReplayResult.error`` with an empty
    or partial ``steps`` list rather than raising.
    """
    error = validate(recipe, inputs, servers, base=confine_to)
    if error is not None:
        return ReplayResult(ok=False, error=error)

    try:
        async with mcp_caller(servers, cwd=cwd, tool_timeout_sec=tool_timeout_sec) as caller:
            return await replay_with_caller(
                recipe, inputs, caller=caller, on_step=on_step, confine_to=confine_to
            )
    except ReplayError as exc:
        return ReplayResult(ok=False, error=str(exc))


async def run_batch(
    recipe: Recipe,
    runs: list[dict[str, Any]],
    *,
    servers: list[JsonDict],
    cwd: str | None = None,
    tool_timeout_sec: float = DEFAULT_TOOL_TIMEOUT_SEC,
    on_step: Callable[[int, StepResult], Awaitable[None]] | None = None,
    on_item: Callable[[int, ReplayResult], Awaitable[None]] | None = None,
    on_step_start: Callable[[int, int, str, str], Awaitable[None]] | None = None,
    confine_to: Path | None = None,
) -> list[ReplayResult]:
    """Replay *recipe* once per input binding in *runs*, sharing one set of stacks.

    Like calling :func:`run` for each item, but the needed MCP servers are spawned
    **once** and reused across every item — a batch of N items pays the
    server-startup cost once, not N times. Items run sequentially; a failed item
    does not stop the rest. Each item is validated against its own inputs, and an
    item that fails validation — or that would write to the same place as another
    item (:func:`batch_output_collisions`) — is recorded as a failed
    :class:`ReplayResult` without running a step. ``on_step``/``on_item`` stream
    progress tagged with the item's index in *runs*; the return value is one
    :class:`ReplayResult` per item, in order.

    The MCP servers are shared across items for speed, but one that crashes is
    re-spawned for the next item (see :func:`mcp_caller`), so a single failure does
    not cascade to the rest of the batch. ``on_step_start`` fires as
    ``(item, step_index, server, tool)`` just before a tool is called, so a caller
    can show which step is in flight.
    """
    pre = [validate(recipe, inputs, servers, base=confine_to) for inputs in runs]
    for item, reason in batch_output_collisions(recipe, runs, base=confine_to).items():
        if pre[item] is None:
            pre[item] = reason
    results: list[ReplayResult] = []

    async def _emit(item: int, res: ReplayResult) -> None:
        results.append(res)
        if on_item is not None:
            await on_item(item, res)

    # Nothing can run (a built-in step, an uninstalled stack, or no items at all):
    # report each item's failure without paying to spawn the stacks.
    if all(err is not None for err in pre):
        for item, err in enumerate(pre):
            await _emit(item, ReplayResult(ok=False, error=err))
        return results

    try:
        async with mcp_caller(servers, cwd=cwd, tool_timeout_sec=tool_timeout_sec) as caller:
            for item, (inputs, err) in enumerate(zip(runs, pre, strict=True)):
                if err is not None:
                    await _emit(item, ReplayResult(ok=False, error=err))
                    continue

                async def _step(sr: StepResult, _item: int = item) -> None:
                    if on_step is not None:
                        await on_step(_item, sr)

                async def _start(index: int, server: str, tool: str, _item: int = item) -> None:
                    if on_step_start is not None:
                        await on_step_start(_item, index, server, tool)

                res = await replay_with_caller(
                    recipe,
                    inputs,
                    caller=caller,
                    on_step=_step,
                    on_step_start=_start,
                    confine_to=confine_to,
                )
                await _emit(item, res)
    except ReplayError as exc:
        # Engine-level failure mid-batch (e.g. a stack teardown error): fail the
        # items that had not run yet rather than raising.
        for item in range(len(results), len(runs)):
            await _emit(item, ReplayResult(ok=False, error=str(exc)))
    return results
