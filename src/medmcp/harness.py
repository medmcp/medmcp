"""What the Unified Harness (vibe ≥2.26) puts on the ACP wire, and how we read it.

The harness names tools by *route*: a built-in is ``file_system.bash`` or
``skill.read``, an MCP tool is ``mcp_<server>.<tool>`` with the server name
squashed to an identifier. The rest of MedMCP keys on the names it always has —
``bash``, ``medmcp-neuro_skull_strip`` — so the translation happens once, at the
edge, and nothing downstream learns the routes. Stdlib-only so ``pathguard`` and
the CLI can use it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import cast

# Built-in routes → the vibe tool name they stand for, as the harness's own
# permission resolver maps them (``RUST_BUILTIN_TOOL_SOURCES``). The process
# family is the shell's background mode and is accounted to ``bash``.
BUILTIN_ROUTES: dict[str, str] = {
    "file_system.read_file": "read_file",
    "file_system.write_file": "write_file",
    "file_system.search_replace": "edit",
    "file_system.bash": "bash",
    "skill.read": "skill",
    "process.start": "bash",
    "process.output": "bash",
    "process.write": "bash",
    "process.list": "bash",
    "process.stop": "bash",
}

# Tools that belong to the harness itself: the model's catalogue search, the
# TypeScript sandbox it calls stack tools from, and its scratchpad. None of them
# is a step of a workflow, and none runs anything on its own.
HARNESS_TOOLS: frozenset[str] = frozenset(
    {"search_tool_functions", "run_typescript", "unified_harness_scratchpad"}
)

# The text our path guard hands back (``pathguard.py``). Under the harness a hook
# denial reaches the chat as an unnamed failed effect whose output is this text,
# so the marker is how the server tells "turned back" from "failed".
PATHGUARD_MARKER = "RETRY THIS CALL WITH A CORRECTED PATH."

_IDENTIFIER_CHAR = re.compile(r"[A-Za-z0-9_]")
_MCP_PREFIX = "mcp_"


def normalize_segment(value: str) -> str:
    """A name as the harness spells it inside a route: identifier characters only."""
    normalized = "".join(c if _IDENTIFIER_CHAR.fullmatch(c) else "_" for c in value)
    return normalized or "_"


def mcp_group(server_name: str) -> str:
    """The route group the harness gives an MCP server (``medmcp-neuro`` → ``mcp_medmcp_neuro``)."""
    return f"{_MCP_PREFIX}{normalize_segment(server_name)}"


def canonical_tool_name(route: str, server_names: Iterable[str] = ()) -> str:
    """MedMCP's name for a tool the harness reported by *route*.

    A built-in route becomes its vibe tool name; an MCP route becomes
    ``<server>_<tool>`` with the server's real name when it is among
    *server_names* (several spellings can squash to one group, so the known
    names decide), else the squashed spelling; anything else is returned as is —
    a harness tool, or a name that already is canonical.
    """
    if route in BUILTIN_ROUTES:
        return BUILTIN_ROUTES[route]
    group, dot, tool = route.partition(".")
    if not dot or not group.startswith(_MCP_PREFIX):
        return route
    for name in server_names:
        if mcp_group(name) == group:
            return f"{name}_{tool}"
    return f"{group[len(_MCP_PREFIX) :]}_{tool}"


def unwrap_result(raw_output: object) -> tuple[str, object]:
    """The text and the structured value of a tool result, with the harness envelope removed.

    The harness reports an MCP result as ``{"type": "success", "content": [{"type":
    "text", "text": …}, …]}``; vibe before 2.26 reported the result itself. Returns
    the concatenated text and, when that text is a JSON object, the parsed object —
    else the text (legacy renderings carried their ``structured:`` blob in it).
    Anything that is not an envelope is returned as its own text and value.
    """
    if isinstance(raw_output, dict):
        envelope = cast("dict[str, object]", raw_output)
        blocks = envelope.get("content")
        if not isinstance(blocks, list):
            return json.dumps(envelope), envelope
        parts: list[str] = []
        for block in cast("list[object]", blocks):
            if isinstance(block, dict):
                text_value = cast("dict[str, object]", block).get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        text = "\n".join(parts)
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                parsed: object = json.loads(stripped)
            except ValueError:
                return text, text
            return text, cast("dict[str, object]", parsed) if isinstance(parsed, dict) else text
        return text, text
    if isinstance(raw_output, str):
        return raw_output, raw_output
    return ("" if raw_output is None else json.dumps(raw_output)), raw_output


def is_harness_tool(name: str) -> bool:
    """Whether *name* is one of the harness's own tools (never a workflow step)."""
    return name in HARNESS_TOOLS
