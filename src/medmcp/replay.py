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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from medmcp.workflow import Recipe

JsonDict = dict[str, Any]

# A callable that runs one tool and returns (ok, structured_output, error_text).
ToolCaller = Callable[[str, str, JsonDict], Awaitable[tuple[bool, JsonDict, "str | None"]]]

_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")
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
            ref = whole.group(1).strip()
            return bindings.get(ref, value)

        def _sub(match: re.Match[str]) -> str:
            ref = match.group(1).strip()
            return str(bindings[ref]) if ref in bindings else match.group(0)

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
    text = _result_text(result)
    return "ok: False" in text or "returncode: 1" in text


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
) -> AsyncIterator[ToolCaller]:
    """Yield a :data:`ToolCaller` backed by lazily-spawned MCP stdio servers.

    Each needed stack's server is started on first use and reused for the rest of
    the run; all are shut down when the context exits.
    """
    server_map = {str(s["name"]): s for s in servers}
    timeout = timedelta(seconds=tool_timeout_sec)

    async with AsyncExitStack() as stack:
        sessions: dict[str, ClientSession] = {}

        async def _session(server: str) -> ClientSession:
            if server not in sessions:
                cfg = server_map.get(server)
                if cfg is None:
                    raise ReplayError(f"stack {server!r} is not installed")
                read, write = await stack.enter_async_context(
                    stdio_client(_server_params(cfg, cwd=cwd))
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                sessions[server] = session
            return sessions[server]

        async def _call(
            server: str, tool: str, args: JsonDict
        ) -> tuple[bool, JsonDict, str | None]:
            session = await _session(server)
            result = await session.call_tool(tool, args, read_timeout_seconds=timeout)
            structured = extract_structured(result)
            if _result_failed(result):
                return False, structured, _result_text(result) or "tool reported failure"
            return True, structured, None

        yield _call


# ── Validation & execution ────────────────────────────────────────────────────


def validate(recipe: Recipe, inputs: dict[str, Any], servers: list[JsonDict]) -> str | None:
    """Return a human-readable error if *recipe* can't be replayed, else ``None``.

    Checks that every declared input has a value, every referenced stack is
    installed, and that the recipe has no built-in (non-MCP) tool steps.
    """
    missing_inputs = [i.name for i in recipe.inputs if i.name not in inputs]
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
        return f"required stack(s) not installed: {', '.join(needed)}"

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
    bindings: dict[str, Any] = dict(inputs)
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
