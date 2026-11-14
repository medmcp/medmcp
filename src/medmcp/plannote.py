"""A ``post_tool`` hook that hands the model the core's note on a call outside the plan.

When the person allows the third (and every later) call outside an approved plan,
the core wants the agent to stop and propose a revised plan (see
:func:`medmcp.plan.off_plan_reminder`). The ACP permission answer carries no
text, so the note has to ride on the call's result. A stack call's result passes
through the core's broker, which adds it there; the result of one of vibe's own
tools (``bash``, ``write_file``, …) never does, so this hook asks the core for the
note when such a call completes and appends it (``additional_context``).

The hook contract is "exit 0 and print a JSON object, or nothing, on stdout".
Every failure prints nothing: a broken hook leaves the result as it was.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, cast

from medmcp import harness, plan

JsonDict = dict[str, Any]

# vibe's own tools: the ones whose results never pass through the broker.
_BUILTINS: frozenset[str] = frozenset(
    {*harness.BUILTIN_ROUTES.values(), "grep", "edit", "todo", "web_fetch"}
)


def note_for(payload: JsonDict) -> str | None:
    """The core's note for the call in a ``post_tool`` *payload*, or ``None``."""
    name = harness.canonical_tool_name(str(payload.get("tool_name") or ""))
    if name not in _BUILTINS:
        return None
    raw_args = payload.get("tool_input")
    args = cast("JsonDict", raw_args) if isinstance(raw_args, dict) else {}
    response = asyncio.run(plan.ask_core("plan_note", {"tool": name, "args": args}, timeout=3.0))
    if response is None:
        return None
    note = response.get("note")
    return note if isinstance(note, str) and note else None


def main() -> None:
    """Entry point of ``medmcp-plan-note``: read the hook payload, print the note if any."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        note = note_for(cast("JsonDict", payload)) if isinstance(payload, dict) else None
    except Exception:
        note = None
    if note:
        print(json.dumps({"hook_specific_output": {"additional_context": note}}))
