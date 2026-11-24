"""What a stack tool says it does to the data: its declared *effect*.

A stack declares at most one effect per tool in its ``tools/list``:

``read_only``
    Reads its inputs and leaves every file as it was.
``writes_new``
    Writes output files; may re-create its own earlier outputs, never changes the
    user's files.
``destructive``
    Deletes, moves or changes files the user brought (inputs, an existing dataset).

Overwriting one particular existing file is a property of a call, not of a tool:
the approval card's path check shows it per call, whatever the tag says.

The declaration is read from the tool's ``_meta["org.medmcp/effect"]``. When that
key is absent, the standard MCP annotations stand in for it: ``readOnlyHint`` true
→ ``read_only``; else ``destructiveHint`` true → ``destructive``; else both hints
explicitly false → ``writes_new``. Anything else is *untagged*: the MCP spec's
defaults (an unset ``destructiveHint`` counting as true) are deliberately not
applied, so a tool that says nothing is shown as saying nothing. A ``_meta`` key
with any other value is ignored, and the tool is untagged (no fallback to the
annotations, which the stack chose to override).

A declaration is information for the person and the model. It never approves a
call; the only policy it drives makes approval *stricter* (a destructive tool is
asked about every time, see :mod:`medmcp.server`).

Kept free of everything but ``mcp.types`` so the per-call proxy can import it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

import mcp.types as mcp_types

EFFECT_META_KEY: Final = "org.medmcp/effect"
READ_ONLY: Final = "read_only"
WRITES_NEW: Final = "writes_new"
DESTRUCTIVE: Final = "destructive"
EFFECTS: Final = frozenset({READ_ONLY, WRITES_NEW, DESTRUCTIVE})

# The line the proxy appends to a tagged tool's description: vibe drops MCP
# annotations and ``_meta`` before the model, so the description is the only way
# the model learns what a tool does to the data.
MODEL_LINES: Final[Mapping[str, str]] = {
    READ_ONLY: "Effect: read-only.",
    WRITES_NEW: "Effect: writes new files.",
    DESTRUCTIVE: "Effect: destructive (deletes, moves or changes the user's files).",
}


def declared_effect(meta: object, annotations: object) -> str | None:
    """The effect a tool declares, from its ``_meta`` and its annotations (both raw dicts).

    Args:
        meta: The tool's ``_meta`` object, or ``None``.
        annotations: The tool's ``annotations`` object, or ``None``.

    Returns:
        ``"read_only"``, ``"writes_new"``, ``"destructive"``, or ``None`` (untagged).
    """
    if isinstance(meta, Mapping) and EFFECT_META_KEY in meta:
        value = cast("Mapping[str, Any]", meta)[EFFECT_META_KEY]
        return value if isinstance(value, str) and value in EFFECTS else None
    if not isinstance(annotations, Mapping):
        return None
    hints = cast("Mapping[str, Any]", annotations)
    read_only = hints.get("readOnlyHint")
    destructive = hints.get("destructiveHint")
    if read_only is True:
        return READ_ONLY
    if destructive is True:
        return DESTRUCTIVE
    if read_only is False and destructive is False:
        return WRITES_NEW
    return None


def tool_effect(tool: mcp_types.Tool) -> str | None:
    """The effect *tool* declares (see :func:`declared_effect`)."""
    annotations = tool.annotations.model_dump() if tool.annotations is not None else None
    return declared_effect(tool.meta, annotations)


def describe(description: str | None, effect: str | None) -> str | None:
    """*description* with the effect's line appended; unchanged for an untagged tool.

    The original text is kept byte for byte; the line goes after a blank line
    (one newline when the text already ends with one).
    """
    line = MODEL_LINES.get(effect) if effect is not None else None
    if line is None:
        return description
    if not description:
        return line
    separator = "\n" if description.endswith("\n") else "\n\n"
    return f"{description}{separator}{line}"
