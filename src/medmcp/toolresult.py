"""What a tool result says — shared by provenance, distillation and replay.

Three modules read tool results and must agree: provenance records what a call
produced, distillation decides which calls enter a recipe and which outputs feed
later steps, and replay decides whether a replayed step failed. Each used to
carry its own copy of these two rules; a recipe and its replay now agree on
"what was produced" and "what failed" by construction.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, cast

JsonDict = dict[str, Any]

# vibe renders a tool's structured output into the transcript as a Python-literal
# ``structured: {...}`` blob after the text; ``ast.literal_eval`` reads it back.
_STRUCTURED_RE = re.compile(r"structured:\s*(\{.*\})", re.DOTALL)

# FastMCP wraps any exception a tool raises into a result whose text begins with
# "Error executing tool <name>:" and sets the MCP ``isError`` flag. vibe-acp,
# however, renders that result into ``messages.jsonl`` as ``ok: True`` (its
# ``ok:`` tracks JSON-RPC success, not the tool's ``isError``), so by the time
# distillation reads the transcript the wrapper text is the only failure signal
# left. ``_NONZERO_EXIT_RE`` additionally catches "(exit N)" for tools that
# embed a subprocess exit code in the message.
_TOOL_ERROR_MARKER = "Error executing tool"
# The path guard's own refusal text (``pathguard.py``): a call that never ran.
_PATHGUARD_MARKER = "RETRY THIS CALL WITH A CORRECTED PATH."
_NONZERO_EXIT_RE = re.compile(r"\(exit ([1-9]\d*)\)")


def parse_structured_blob(text: str) -> JsonDict:
    """The ``structured: {...}`` object in a rendered tool result; ``{}`` when none."""
    match = _STRUCTURED_RE.search(text)
    if match is None:
        # vibe ≥2.26 hands an MCP tool's structured result over as plain JSON.
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                parsed_json: object = json.loads(stripped)
            except ValueError:
                return {}
            return cast("JsonDict", parsed_json) if isinstance(parsed_json, dict) else {}
        return {}
    try:
        parsed = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return {}
    return cast("JsonDict", parsed) if isinstance(parsed, dict) else {}


def looks_failed(text: str) -> bool:
    """Whether a tool result's text carries one of the known failure markers."""
    return (
        "ok: False" in text
        or "returncode: 1" in text
        or _TOOL_ERROR_MARKER in text
        or _PATHGUARD_MARKER in text
        or _NONZERO_EXIT_RE.search(text) is not None
    )
