"""Schema for distilled, reusable MedMCP workflows (Tier 2).

A :class:`Recipe` is the machine-readable, replayable form of a workflow
distilled from a session's provenance record (see :mod:`medmcp.distill`). It is
emitted as ``recipe.yaml`` next to a human/LLM-facing ``SKILL.md`` so the same
distillation feeds both a deterministic re-run and the existing skill system.

Concrete file paths and devices captured during a session are lifted into named
placeholders so the recipe can be replayed against new inputs:

- ``{{in_N}}``        — an input the workflow consumes but did not itself produce
- ``{{stepM.<key>}}`` — an output produced by an earlier step and reused later
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JsonDict = dict[str, Any]


@dataclass
class WorkflowInput:
    """A named input the workflow consumes (lifted from a concrete value)."""

    name: str
    """Placeholder name, e.g. ``in_1`` (referenced as ``{{in_1}}``)."""
    example: str
    """The concrete value seen in the originating session, kept as documentation."""
    description: str = ""
    """Optional human description of what this input is."""

    def to_dict(self) -> JsonDict:
        """Return a plain-dict form suitable for YAML/JSON serialization."""
        out: JsonDict = {"name": self.name, "example": self.example}
        if self.description:
            out["description"] = self.description
        return out


@dataclass
class RecipeStep:
    """A single tool invocation in a distilled workflow."""

    server: str
    """MCP server the tool belongs to, or ``"builtin"`` for vibe-acp tools."""
    tool: str
    """Tool name without the server prefix (e.g. ``skull_strip``)."""
    arguments: JsonDict
    """Resolved arguments, with concrete paths replaced by placeholders."""
    produces: dict[str, str] = field(default_factory=dict[str, str])
    """Map of output key → placeholder name for values this step produces."""

    def to_dict(self) -> JsonDict:
        """Return a plain-dict form suitable for YAML/JSON serialization."""
        out: JsonDict = {
            "server": self.server,
            "tool": self.tool,
            "arguments": self.arguments,
        }
        if self.produces:
            out["produces"] = self.produces
        return out


@dataclass
class Recipe:
    """A distilled, parameterized, replayable workflow."""

    name: str
    description: str
    inputs: list[WorkflowInput] = field(default_factory=list[WorkflowInput])
    steps: list[RecipeStep] = field(default_factory=list[RecipeStep])

    def to_dict(self) -> JsonDict:
        """Return a plain-dict form suitable for YAML/JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
        }
