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
class StackRequirement:
    """A stack the workflow needs, pinned for reproducibility.

    Captured at distillation from the session's provenance manifest and filtered
    to the stacks the recipe actually uses. ``version`` applies to uv-tool stacks;
    ``image`` (+ best-effort ``digest``) applies to container stacks.
    """

    stack: str
    """The MCP server/stack name (e.g. ``medmcp-neuro``)."""
    version: str = ""
    """Package version for a uv-tool stack (e.g. ``0.1.0``); empty if unknown."""
    image: str = ""
    """Container image ref for a container stack (e.g. ``ghcr.io/medmcp/neuro:main``)."""
    digest: str = ""
    """Resolved image digest (``sha256:…``) for exact pinning; empty if unresolved."""

    def to_dict(self) -> JsonDict:
        """Return a plain-dict form suitable for YAML/JSON serialization."""
        out: JsonDict = {"stack": self.stack}
        for key in ("version", "image", "digest"):
            value = getattr(self, key)
            if value:
                out[key] = value
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
    requires: list[StackRequirement] = field(default_factory=list[StackRequirement])
    manual_steps: list[str] = field(default_factory=list[str])
    """Built-in (non-MCP) steps dropped from the replayable recipe, kept as docs.

    These are real actions from the session (e.g. ``builtin:edit``) that the replay
    engine can't run deterministically; recording them here lets the workflow note
    the manual work instead of silently losing it.
    """

    def to_dict(self) -> JsonDict:
        """Return a plain-dict form suitable for YAML/JSON serialization."""
        out: JsonDict = {
            "name": self.name,
            "description": self.description,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.requires:
            out["requires"] = [r.to_dict() for r in self.requires]
        if self.manual_steps:
            out["manual_steps"] = list(self.manual_steps)
        return out
