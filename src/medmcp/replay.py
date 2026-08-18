"""Deterministic replay of a distilled workflow on new data (no LLM).

Loads a :class:`~medmcp.workflow.Recipe`, binds new values to its ``{{in_N}}``
inputs, and runs each step by calling the MCP tool directly over stdio — no
vibe-acp, no model reasoning. An output a step produces (its ``produces`` keys)
is captured from the tool's structured result and substituted into later steps'
``{{stepM.<key>}}`` placeholders, so a multi-step pipeline chains exactly as it
did when it was recorded.

This module has no UI/vibe-acp dependency: callers pass in the resolved
MCP server launch configs (``command``/``args``/``env`` as produced by
``app._load_mcp_servers``), so it can be driven from the UI or a CLI alike.

Replaying runs real tools with real side effects (file writes, etc.). It is
deliberately separate from the chat permission flow: callers are expected to
confirm with the user *before* calling :func:`run` (the UI previews the resolved
steps first). Built-in vibe-acp tools (``builtin:*`` such as ``bash``) are not
MCP tools and cannot be replayed; :func:`validate` reports them rather than
silently skipping a step the pipeline depends on.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any, cast

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from medmcp.workflow import Recipe

JsonDict = dict[str, Any]

# A callable that runs one tool and returns (ok, structured_output, error_text).
ToolCaller = Callable[[str, str, JsonDict], Awaitable[tuple[bool, JsonDict, "str | None"]]]

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

    match = re.search(r"structured:\s*(\{.*\})", text, re.DOTALL)
    if match is not None:
        try:
            parsed = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return {}
        if isinstance(parsed, dict):
            return cast("JsonDict", parsed)
    return {}


def _result_failed(result: mcp_types.CallToolResult) -> bool:
    """Detect a failed tool call (protocol error flag or known failure markers)."""
    if result.isError:
        return True
    # Defense-in-depth for stacks that report a failure as a non-error result
    # (text only). FastMCP wraps a raised exception as "Error executing tool …";
    # "(exit N)" catches an embedded non-zero subprocess code. Mirrors
    # ``distill._is_failed_result`` so a recipe and its replay agree on failure.
    text = _result_text(result)
    return (
        "ok: False" in text
        or "returncode: 1" in text
        or "Error executing tool" in text
        or re.search(r"\(exit ([1-9]\d*)\)", text) is not None
    )


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
            for server in list(server_stacks):
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
    bound: dict[str, Any] = dict(inputs)
    for inp in recipe.inputs:
        if inp.name in bound or not inp.default:
            continue
        resolved = resolve_value(inp.default, bound)
        if unresolved_refs(resolved):
            continue
        bound[inp.name] = resolved
    return bound


def validate(recipe: Recipe, inputs: dict[str, Any], servers: list[JsonDict]) -> str | None:
    """Return a human-readable error if *recipe* can't be replayed, else ``None``.

    Checks that every declared input has a value, every referenced stack is
    installed, and that the recipe has no built-in (non-MCP) tool steps.
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
    return None


async def replay_with_caller(
    recipe: Recipe,
    inputs: dict[str, Any],
    *,
    caller: ToolCaller,
    on_step: Callable[[StepResult], Awaitable[None]] | None = None,
) -> ReplayResult:
    """Execute *recipe* step-by-step using *caller*; abort on the first failure.

    *inputs* maps placeholder names (``in_1`` …) to concrete values. Outputs a
    step produces are captured and bound as ``{{stepM.<key>}}`` for later steps.
    """
    # Defaults are filled here, the one place every replay path funnels through
    # (single run and each batch item alike), so no caller can forget them.
    bindings: dict[str, Any] = apply_input_defaults(recipe, inputs)
    result = ReplayResult()

    for index, step in enumerate(recipe.steps, start=1):
        args = resolve_arguments(step.arguments, bindings)
        pending = unresolved_refs(args)
        if pending:
            step_result = StepResult(
                index=index,
                server=step.server,
                tool=step.tool,
                arguments=args,
                ok=False,
                error=f"unresolved placeholders: {', '.join(sorted(pending))}",
            )
            result.steps.append(step_result)
            result.ok = False
            result.error = step_result.error
            if on_step is not None:
                await on_step(step_result)
            break

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
) -> ReplayResult:
    """Validate and replay *recipe* on *inputs*, spawning the needed MCP stacks.

    Returns a :class:`ReplayResult`; an engine-level problem (failed validation,
    a stack that won't start) is reported via ``ReplayResult.error`` with an empty
    or partial ``steps`` list rather than raising.
    """
    error = validate(recipe, inputs, servers)
    if error is not None:
        return ReplayResult(ok=False, error=error)

    try:
        async with mcp_caller(servers, cwd=cwd, tool_timeout_sec=tool_timeout_sec) as caller:
            return await replay_with_caller(recipe, inputs, caller=caller, on_step=on_step)
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
) -> list[ReplayResult]:
    """Replay *recipe* once per input binding in *runs*, sharing one set of stacks.

    Like calling :func:`run` for each item, but the needed MCP servers are spawned
    **once** and reused across every item — a batch of N items pays the
    server-startup cost once, not N times. Items run sequentially; a failed item
    does not stop the rest. Each item is validated against its own inputs, and an
    item that fails validation is recorded as a failed :class:`ReplayResult`
    without running a step. ``on_step``/``on_item`` stream progress tagged with the
    item's index in *runs*; the return value is one :class:`ReplayResult` per item,
    in order.

    The MCP servers are shared across items for speed, but one that crashes is
    re-spawned for the next item (see :func:`mcp_caller`), so a single failure does
    not cascade to the rest of the batch.
    """
    pre = [validate(recipe, inputs, servers) for inputs in runs]
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

                res = await replay_with_caller(recipe, inputs, caller=caller, on_step=_step)
                await _emit(item, res)
    except ReplayError as exc:
        # Engine-level failure mid-batch (e.g. a stack teardown error): fail the
        # items that had not run yet rather than raising.
        for item in range(len(results), len(runs)):
            await _emit(item, ReplayResult(ok=False, error=str(exc)))
    return results
