"""``medmcp-task-tool`` — the MCP server that offers ``get_task`` and ``cancel_task``.

vibe spawns it like a stack (``medmcp-tasks`` in the synced config, only while the
stack pool — and with it the broker — is on). It holds no state: both tools ask
the core over the broker socket (ops ``task_get`` / ``task_cancel``, see
:mod:`medmcp.tasks`) and return the core's answer as the tool result.

``get_task`` of a task started in the same chat is answered without an approval
dialog (the one written exception, see ``_handle_permission`` in ``server.py``);
``cancel_task`` asks like any call, and the core runs it only on the approval the
person gave for exactly that call.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, cast

import mcp.types as mcp_types

from medmcp import brokerclient
from medmcp.tasks import CANCEL_TASK, GET_TASK, TASK_SERVER

JsonDict = dict[str, Any]

_ARGS_SCHEMA: JsonDict = {
    "type": "object",
    "properties": {"task_id": {"type": "string", "description": "The task id (tsk_…)."}},
    "required": ["task_id"],
    "additionalProperties": False,
}

TOOLS: list[mcp_types.Tool] = [
    mcp_types.Tool(
        name=GET_TASK,
        description=(
            "Status of a background task: a long stack call that answered with a task id"
            " and continues on its own. Returns status (working, completed, failed,"
            " cancelled), a status message and the elapsed time; once completed, the"
            " call's result. A finished task's result comes to you in a new message by"
            " itself: use this only when the user asks about a task, never to wait."
        ),
        inputSchema=_ARGS_SCHEMA,
    ),
    mcp_types.Tool(
        name=CANCEL_TASK,
        description=(
            "Stop a background task (task id tsk_…) or background job (its job id) that"
            " is still running. Asks the user first."
        ),
        inputSchema=_ARGS_SCHEMA,
    ),
]

_OPS: dict[str, str] = {GET_TASK: "task_get", CANCEL_TASK: "task_cancel"}


def _error(text: str) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)], isError=True
    )


async def call(name: str, arguments: JsonDict) -> mcp_types.CallToolResult:
    """Answer one call of a task tool by asking the core."""
    op = _OPS.get(name)
    if op is None:
        return _error(f"unknown tool {name!r}")
    answer = await brokerclient.ask(op, {"args": arguments}, timeout=15.0)
    if answer is None:
        return _error("MedMCP could not be reached to read the task.")
    raw = answer.get("result")
    if not isinstance(raw, dict):
        return _error("MedMCP gave no answer for this task.")
    return mcp_types.CallToolResult.model_validate(cast("JsonDict", raw))


async def _serve() -> None:
    """Offer the task tools over stdio until vibe disconnects."""
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    server: Server[object, object] = Server(TASK_SERVER)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:  # pyright: ignore[reportUnusedFunction]
        return TOOLS

    @server.call_tool(validate_input=False)
    async def _call_tool(  # pyright: ignore[reportUnusedFunction]
        name: str, arguments: JsonDict
    ) -> mcp_types.CallToolResult:
        return await call(name, arguments)

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Entry point of ``medmcp-task-tool``."""
    import anyio

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    anyio.run(_serve)


if __name__ == "__main__":
    main()
