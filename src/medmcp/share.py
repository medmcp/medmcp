"""Export / import a distilled workflow as a single self-contained YAML file.

A workflow lives as ``.vibe/workflows/<name>/recipe.yaml``. To share one with a
colleague we serialize it into a single ``<name>.workflow.yaml`` envelope: the
recipe's ``inputs`` and ``steps``, the ``requires`` block (stacks pinned by
version or image+digest), and its name and description — what the replay
engine needs, and nothing else. Files written by earlier releases also carried
a ``documentation`` block and cached ``prose``; both are ignored on import.

Import lands the workflow beside the existing ones and never overwrites one
(names collide → a unique suffix is added).

This module has no UI/vibe-acp dependency (like the rest of the distill cluster)
so it can be driven from the server or the ``medmcp`` CLI alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from medmcp import distill, workflow

JsonDict = dict[str, Any]

# Envelope format version. Bump only on a breaking schema change; import accepts
# any version it understands (currently exactly this one).
FORMAT_VERSION: int = 1
FORMAT_KEY: str = "medmcp_workflow"

# Suggested filename extension for an exported workflow.
EXPORT_SUFFIX: str = ".workflow.yaml"


class WorkflowShareError(Exception):
    """Raised when an import payload is malformed or an unknown format version."""


# ── Export ────────────────────────────────────────────────────────────────────


def export_workflow(name: str, *, workflows_root: Path | None = None) -> str:
    """Serialize workflow *name* into a single inline-YAML envelope (returns text).

    Any required image without a digest is resolved best-effort at export time
    so the shared file pins as tightly as possible.

    Raises ``FileNotFoundError`` if no workflow has that name.
    """
    root = distill.resolve_root(workflows_root)
    workflow_dir = workflow.workflow_dir(root, name)
    if workflow_dir is None:
        raise FileNotFoundError(f"no workflow named {name!r} in {root}")
    recipe = distill.load_recipe(workflow_dir)

    # Tighten the pin: fill any missing digests now (the image may not have been
    # present when the workflow was distilled). Best-effort — never fails export.
    for req in recipe.requires:
        if req.image and not req.digest:
            req.digest = distill.resolve_digest(req.image)

    envelope: JsonDict = {FORMAT_KEY: FORMAT_VERSION, **recipe.to_dict()}
    return yaml.safe_dump(envelope, sort_keys=False, allow_unicode=True)


# ── Import ────────────────────────────────────────────────────────────────────


def import_workflow(yaml_text: str, *, workflows_root: Path | None = None) -> Path:
    """Import a shared workflow envelope as a new workflow; return its directory.

    Validates the envelope (format version + a non-empty step list),
    reconstructs the recipe, and writes ``recipe.yaml`` into a fresh
    ``<name>/``. A name already in use gets a unique suffix so an import never
    clobbers an existing workflow.

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

    root = distill.resolve_root(workflows_root)
    recipe = workflow.Recipe.from_dict(env)
    recipe.name = workflow.unique_name(root, distill.slugify(recipe.name), tag="imported")
    target = root / recipe.name
    workflow.write_recipe(target, recipe)
    return target
