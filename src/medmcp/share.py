"""Export / import a distilled workflow as a single self-contained YAML file.

A workflow normally lives as a directory under
``.vibe/workflows/{draft,active}/<name>/`` (``recipe.yaml`` + ``SKILL.md`` +
``prose.json``). To share one with a colleague we serialize it into a single
inline ``<name>.workflow.yaml`` envelope that carries everything needed to
reproduce and document it:

- the replayable ``steps`` and ``inputs`` (the recipe),
- the ``requires`` block (stacks pinned by version or image+digest),
- a human-readable ``documentation`` block (what the workflow does), and
- the cached ``prose`` so the narrative can still be refined after import.

Import always lands the workflow as a **draft** for review before promotion, and
never overwrites an existing one (names collide → a unique suffix is added).

This module has no UI/vibe-acp dependency (like the rest of the distill cluster)
so it can be driven from the server or the ``medmcp`` CLI alike.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from medmcp import distill, provenance
from medmcp.workflow import Recipe, RecipeStep, StackRequirement, WorkflowInput

JsonDict = dict[str, Any]

# Envelope format version. Bump only on a breaking schema change; import accepts
# any version it understands (currently exactly this one).
FORMAT_VERSION: int = 1
FORMAT_KEY: str = "medmcp_workflow"

# Suggested filename extension for an exported workflow.
EXPORT_SUFFIX: str = ".workflow.yaml"


class WorkflowShareError(Exception):
    """Raised when an import payload is malformed or an unknown format version."""


def _workflows_root(workflows_root: Path | None) -> Path:
    """Resolve the workflows root, defaulting to ``.vibe/workflows``."""
    return workflows_root if workflows_root is not None else provenance.VIBE_HOME / "workflows"


def _find_workflow_dir(name: str, workflows_root: Path | None) -> Path:
    """Return the dir for workflow *name* (active preferred, then draft).

    Raises ``FileNotFoundError`` if neither location has it.
    """
    root = _workflows_root(workflows_root)
    for kind in ("active", "draft"):
        candidate = root / kind / name
        if (candidate / "recipe.yaml").exists():
            return candidate
    raise FileNotFoundError(f"no workflow named {name!r} in {root}")


def _skill_body(skill_md: str) -> str:
    """Strip the leading YAML frontmatter from a ``SKILL.md``, returning the body."""
    if skill_md.startswith("---"):
        end = skill_md.find("\n---", 3)
        if end != -1:
            newline = skill_md.find("\n", end + 1)
            if newline != -1:
                return skill_md[newline + 1 :].lstrip("\n")
    return skill_md


def _read_prose(workflow_dir: Path) -> JsonDict | None:
    """Read a workflow's cached ``prose.json`` (``None`` if absent/mechanical)."""
    path = workflow_dir / "prose.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return cast("JsonDict", data) if isinstance(data, dict) else None


# ── Export ────────────────────────────────────────────────────────────────────


def export_workflow(name: str, *, workflows_root: Path | None = None) -> str:
    """Serialize workflow *name* into a single inline-YAML envelope (returns text).

    Pulls the recipe, the human-readable documentation (the rendered ``SKILL.md``
    body), and the cached prose. Any required image without a digest is resolved
    best-effort at export time so the shared file pins as tightly as possible.

    Raises ``FileNotFoundError`` if no draft/active workflow has that name.
    """
    workflow_dir = _find_workflow_dir(name, workflows_root)
    recipe = distill.load_recipe(workflow_dir)

    # Tighten the pin: fill any missing digests now (the image may not have been
    # present when the workflow was distilled). Best-effort — never fails export.
    for req in recipe.requires:
        if req.image and not req.digest:
            req.digest = distill.resolve_digest(req.image)

    skill_path = workflow_dir / "SKILL.md"
    documentation = (
        _skill_body(skill_path.read_text(encoding="utf-8")) if skill_path.exists() else ""
    )

    envelope: JsonDict = {
        FORMAT_KEY: FORMAT_VERSION,
        "name": recipe.name,
        "description": recipe.description,
    }
    if recipe.requires:
        envelope["requires"] = [r.to_dict() for r in recipe.requires]
    envelope["inputs"] = [i.to_dict() for i in recipe.inputs]
    envelope["steps"] = [s.to_dict() for s in recipe.steps]
    if recipe.manual_steps:
        envelope["manual_steps"] = list(recipe.manual_steps)
    if documentation:
        envelope["documentation"] = documentation
    prose = _read_prose(workflow_dir)
    if prose is not None:
        envelope["prose"] = prose

    return yaml.safe_dump(envelope, sort_keys=False, allow_unicode=True)


# ── Import ────────────────────────────────────────────────────────────────────


def _recipe_from_envelope(env: JsonDict) -> Recipe:
    """Reconstruct a :class:`Recipe` from a validated envelope dict."""
    inputs = [
        WorkflowInput(
            name=str(i.get("name", "")),
            example=str(i.get("example", "")),
            description=str(i.get("description", "")),
            # Without this a shared workflow loses its derived defaults on
            # import, handing the recipient back the very input the default
            # exists to spare them — and silently, since export writes it.
            default=str(i.get("default", "")),
        )
        for i in cast("list[JsonDict]", env.get("inputs", []))
    ]
    steps = [
        RecipeStep(
            server=str(s.get("server", "")),
            tool=str(s.get("tool", "")),
            arguments=cast("JsonDict", s.get("arguments", {})),
            produces=cast("dict[str, str]", s.get("produces", {})),
        )
        for s in cast("list[JsonDict]", env.get("steps", []))
    ]
    requires = [
        StackRequirement(
            stack=str(r.get("stack", "")),
            version=str(r.get("version", "")),
            image=str(r.get("image", "")),
            digest=str(r.get("digest", "")),
        )
        for r in cast("list[JsonDict]", env.get("requires", []))
    ]
    manual_steps = [str(m) for m in cast("list[Any]", env.get("manual_steps", []))]
    return Recipe(
        name=distill.slugify(str(env.get("name", ""))),
        description=str(env.get("description", "")),
        inputs=inputs,
        steps=steps,
        requires=requires,
        manual_steps=manual_steps,
    )


def _unique_draft_name(base: str, workflows_root: Path | None) -> str:
    """Return a draft name not already used by a draft or active workflow."""
    root = _workflows_root(workflows_root)

    def taken(candidate: str) -> bool:
        return any((root / kind / candidate).exists() for kind in ("draft", "active"))

    if not taken(base):
        return base
    suffixed = f"{base}-imported"
    if not taken(suffixed):
        return suffixed
    n = 2
    while taken(f"{suffixed}-{n}"):
        n += 1
    return f"{suffixed}-{n}"


def import_workflow(yaml_text: str, *, workflows_root: Path | None = None) -> Path:
    """Import a shared workflow envelope as a new **draft**; return its directory.

    Validates the envelope (format version + a non-empty step list), reconstructs
    the recipe, and writes ``recipe.yaml`` + ``SKILL.md`` + ``prose.json`` into a
    fresh ``draft/<name>/``. A name already in use gets a unique suffix so an
    import never clobbers an existing workflow.

    Raises ``WorkflowShareError`` on a malformed payload or unknown version.
    """
    try:
        loaded = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise WorkflowShareError(f"not a valid workflow file: {exc}") from exc
    if not isinstance(loaded, dict):
        raise WorkflowShareError("not a valid workflow file (expected a YAML mapping)")
    env = cast("JsonDict", loaded)

    version = env.get(FORMAT_KEY)
    if version != FORMAT_VERSION:
        raise WorkflowShareError(
            f"unsupported workflow format {version!r} (this build understands {FORMAT_VERSION})"
        )
    if not str(env.get("name", "")).strip():
        raise WorkflowShareError("workflow file is missing a name")
    if not isinstance(env.get("steps"), list) or not env["steps"]:
        raise WorkflowShareError("workflow file has no steps")

    recipe = _recipe_from_envelope(env)
    recipe.name = _unique_draft_name(recipe.name, workflows_root)

    draft_dir = _workflows_root(workflows_root) / "draft" / recipe.name
    draft_dir.mkdir(parents=True, exist_ok=True)
    prose = env.get("prose") if isinstance(env.get("prose"), dict) else None
    prose_dict = cast("JsonDict | None", prose)

    recipe_yaml = yaml.safe_dump(recipe.to_dict(), sort_keys=False, allow_unicode=True)
    (draft_dir / "recipe.yaml").write_text(recipe_yaml, encoding="utf-8")
    (draft_dir / "SKILL.md").write_text(
        distill.render_skill_md(recipe, prose_dict), encoding="utf-8"
    )
    (draft_dir / "prose.json").write_text(
        json.dumps(prose_dict) if prose_dict is not None else "null", encoding="utf-8"
    )
    return draft_dir
