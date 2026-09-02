"""Schema and on-disk layout of distilled, replayable MedMCP workflows (Tier 2).

A :class:`Recipe` is the machine-readable, replayable form of a workflow
distilled from a session's provenance record (see :mod:`medmcp.distill`). It is
the whole workflow: one ``recipe.yaml`` in one directory, ``<root>/<name>/``,
which the replay engine runs, the UI lists, and an export wraps. Nothing else
is written there, and nothing there is ever loaded as an agent skill.

Concrete file paths and devices captured during a session are lifted into named
placeholders so the recipe can be replayed against new inputs:

- ``{{in_N}}``        — an input the workflow consumes but did not itself produce
- ``{{stepM.<key>}}`` — an output produced by an earlier step and reused later

Earlier releases kept workflows in ``draft/`` and ``active/`` subdirectories
(promotion marked a draft reviewed, back when a promoted workflow was loaded as
a skill) beside a rendered ``SKILL.md`` and a cached ``prose.json``.
:func:`migrate_layout` folds that layout into this one the first time a root is
touched, so every accessor here calls it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

JsonDict = dict[str, Any]

RECIPE_FILE = "recipe.yaml"
"""The one file a workflow directory holds."""

_LEGACY_KINDS: tuple[str, ...] = ("active", "draft")
"""Pre-0.3 subdirectories of the workflows root; ``active`` is folded in first."""

_LEGACY_FILES: tuple[str, ...] = ("SKILL.md", "prose.json")
"""Generated files earlier releases wrote next to the recipe; removed on migration."""


@dataclass
class WorkflowInput:
    """A named input the workflow consumes (lifted from a concrete value)."""

    name: str
    """Placeholder name, e.g. ``in_1`` (referenced as ``{{in_1}}``)."""
    example: str
    """The concrete value seen in the originating session, kept as documentation."""
    description: str = ""
    """Optional human description of what this input is."""
    default: str = ""
    """Optional placeholder expression filling this input when left unbound.

    Distillation sets this where the session's value was derivable from another
    input — an ``output_dir`` that was simply the input file's folder, say. The
    input stays declared, because the session genuinely had it and the caller may
    want to point it elsewhere; the default only spares them retyping something
    they have already said. Resolved by the replay engine, so it may reference
    other inputs (e.g. ``{{dir(in_1)}}``).
    """

    def to_dict(self) -> JsonDict:
        """Return a plain-dict form suitable for YAML/JSON serialization."""
        out: JsonDict = {"name": self.name, "example": self.example}
        if self.description:
            out["description"] = self.description
        if self.default:
            out["default"] = self.default
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
    """Built-in (non-MCP) steps dropped from the replayable recipe, kept as a note.

    These are real actions from the session (e.g. ``builtin:edit``) that the replay
    engine can't run deterministically; recording them lets a run say what it
    will not do instead of silently losing it.
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

    @classmethod
    def from_dict(cls, data: JsonDict) -> Recipe:
        """Rebuild a recipe from its dict form (a ``recipe.yaml`` or a share envelope).

        Only the known keys are read, so a mapping carrying extra fields — an
        export envelope's format marker, or the blocks older exports included —
        loads cleanly.
        """
        inputs = [
            WorkflowInput(
                name=str(i.get("name", "")),
                example=str(i.get("example", "")),
                description=str(i.get("description", "")),
                default=str(i.get("default", "")),
            )
            for i in cast("list[JsonDict]", data.get("inputs") or [])
        ]
        steps = [
            RecipeStep(
                server=str(s.get("server", "")),
                tool=str(s.get("tool", "")),
                arguments=cast("JsonDict", s.get("arguments") or {}),
                produces=cast("dict[str, str]", s.get("produces") or {}),
            )
            for s in cast("list[JsonDict]", data.get("steps") or [])
        ]
        requires = [
            StackRequirement(
                stack=str(r.get("stack", "")),
                version=str(r.get("version", "")),
                image=str(r.get("image", "")),
                digest=str(r.get("digest", "")),
            )
            for r in cast("list[JsonDict]", data.get("requires") or [])
        ]
        manual_steps = [str(m) for m in cast("list[Any]", data.get("manual_steps") or [])]
        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            inputs=inputs,
            steps=steps,
            requires=requires,
            manual_steps=manual_steps,
        )


# ── On-disk layout ───────────────────────────────────────────────────────────


def read_recipe(directory: Path) -> Recipe:
    """Load the recipe stored in workflow *directory*."""
    data = yaml.safe_load((directory / RECIPE_FILE).read_text(encoding="utf-8"))
    return Recipe.from_dict(cast("JsonDict", data) if isinstance(data, dict) else {})


def write_recipe(directory: Path, recipe: Recipe) -> None:
    """Write *recipe* as ``recipe.yaml`` into *directory* (created if missing)."""
    directory.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(recipe.to_dict(), sort_keys=False, allow_unicode=True)
    (directory / RECIPE_FILE).write_text(text, encoding="utf-8")


def unique_name(root: Path, base: str, *, tag: str = "") -> str:
    """Return *base* if no workflow under *root* has it, else a suffixed variant.

    With a *tag* the first alternative is ``<base>-<tag>``, then
    ``<base>-<tag>-2`` and so on; without one it is ``<base>-2``, ``<base>-3``…
    """
    if not (root / base).exists():
        return base
    stem = f"{base}-{tag}" if tag else base
    if tag and not (root / stem).exists():
        return stem
    n = 2
    while (root / f"{stem}-{n}").exists():
        n += 1
    return f"{stem}-{n}"


def migrate_layout(root: Path) -> None:
    """Fold the pre-0.3 ``active/`` and ``draft/`` subdirectories of *root* into it.

    Promoted workflows move first, so a draft that shared a name with one (it
    was shadowed before, but not deleted) lands as ``<name>-draft`` rather than
    replacing it. The generated ``SKILL.md`` and ``prose.json`` were only read to
    render the skill document, which is gone, so they are removed on the way.
    A root without the old subdirectories is left untouched.
    """
    for kind in _LEGACY_KINDS:
        legacy = root / kind
        if not legacy.is_dir():
            continue
        for src in sorted(p for p in legacy.iterdir() if p.is_dir()):
            dst = root / unique_name(root, src.name, tag=kind if kind == "draft" else "")
            shutil.move(str(src), str(dst))
            for stale in _LEGACY_FILES:
                (dst / stale).unlink(missing_ok=True)
        shutil.rmtree(legacy, ignore_errors=True)


def workflow_dir(root: Path, name: str) -> Path | None:
    """Return ``<root>/<name>`` if it holds a recipe, else ``None``."""
    migrate_layout(root)
    candidate = root / name
    return candidate if (candidate / RECIPE_FILE).is_file() else None


def list_workflows(root: Path) -> list[JsonDict]:
    """List the workflows under *root* as ``{name, description}`` rows, by name.

    A directory whose recipe cannot be read is listed with an empty description
    rather than hiding a workflow the user can still delete.
    """
    if not root.is_dir():
        return []
    migrate_layout(root)
    rows: list[JsonDict] = []
    for d in sorted(root.iterdir()):
        if not (d / RECIPE_FILE).is_file():
            continue
        try:
            description = read_recipe(d).description
        except (yaml.YAMLError, OSError):
            description = ""
        rows.append({"name": d.name, "description": description})
    return rows
