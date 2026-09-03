"""Tier-2 distillation: turn a session's raw log into a replayable workflow.

Reads vibe-acp's ``messages.jsonl`` (the authoritative record of exact tool
names, arguments, and structured results), filters out exploratory/failed
calls, and lifts concrete file paths into named placeholders to produce a
:class:`~medmcp.workflow.Recipe`, written as ``recipe.yaml`` — the one file a
workflow consists of. No model takes part: the recipe is a recording, and the
replay engine (:mod:`medmcp.replay`) is the only thing that runs it. The
workflow is named after the chat it came from (the caller passes the chat's
title; the request that opened the chat is the fallback, and its description).

Output lands in ``.vibe/workflows/<name>/``. Nothing there is ever loaded as a
skill: a workflow runs through the replay engine, never by the agent deciding
to invoke it.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Collection
from pathlib import Path, PurePosixPath
from typing import Any, cast

from medmcp import provenance, workflow
from medmcp.workflow import Recipe, RecipeStep, StackRequirement, WorkflowInput
from medmcp.workspace_note import display_content_text, strip_workspace_note

JsonDict = dict[str, Any]

# Read-only / orchestration tools that carry no reusable workflow meaning; they
# are dropped from the distilled recipe so it captures the actual pipeline.
EXPLORATORY_TOOLS: frozenset[str] = frozenset(
    {"read_file", "grep", "todo", "skill", "task", "ls", "web_fetch", "web_search"}
)

# Pool-machinery / internal tools that the model isn't meant to call and that
# carry no workflow meaning (the persistent backend pool calls ``warmup`` itself).
# Kept in sync with ``proxy._HIDDEN_TOOLS`` (which hides them from the model) via
# ``tests/test_distill.py`` — distill must drop any tool the proxy hides, or a
# session recorded before the proxy existed would bake ``warmup`` into a recipe.
INTERNAL_TOOLS: frozenset[str] = frozenset({"warmup"})

# Shell tools whose command is inspected so read-only invocations can be dropped.
_SHELL_TOOLS: frozenset[str] = frozenset({"bash", "shell", "sh"})

# Shell utilities that only inspect state. A ``bash`` call running one of these
# (with no output redirection) is exploration, not a reusable workflow step.
_READONLY_SHELL_CMDS: frozenset[str] = frozenset(
    {
        "ls", "cat", "find", "head", "tail", "pwd", "echo", "stat", "file",
        "tree", "wc", "du", "df", "which", "type", "realpath", "dirname", "basename",
    }
)  # fmt: skip


def _is_exploratory_call(tool_name: str, arguments: JsonDict) -> bool:
    """Return True if a call only inspects state and carries no workflow meaning.

    Covers the named :data:`EXPLORATORY_TOOLS` plus shell calls whose command is
    a bare read-only inspection (e.g. ``bash`` running ``ls -la data/``). A
    command with output redirection (``>``/``>>``) is kept since it writes.
    """
    if tool_name in EXPLORATORY_TOOLS:
        return True
    if tool_name in _SHELL_TOOLS:
        command = arguments.get("command")
        if isinstance(command, str):
            tokens = command.strip().split()
            if tokens and tokens[0] in _READONLY_SHELL_CMDS and ">" not in command:
                return True
    return False


_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}(\.gz)?$")


# ── Reading the raw log ──────────────────────────────────────────────────────


def parse_messages_file(path: Path) -> list[JsonDict]:
    """Parse a vibe-acp ``messages.jsonl`` file into a list of message dicts."""
    messages: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            messages.append(cast("JsonDict", json.loads(line)))
        except json.JSONDecodeError:
            continue
    return messages


def _first_user_message(messages: list[JsonDict]) -> str:
    """Return the first non-injected user message, for context/naming.

    The workspace server sends a "[workspace context: …]" note with the prompt
    so the agent can resolve "this image" (server.py:_workspace_note). That note
    is live-turn metadata, not part of the user's request — it would otherwise
    pollute the distilled workflow's name and description. vibe ≥2.23 persists
    the note-free text on the message (``user_display_content``), which is
    preferred; older transcripts fall back to stripping the note from the stored
    text (via :func:`strip_workspace_note`).
    """
    for msg in messages:
        if msg.get("role") == "user" and not msg.get("injected"):
            display = msg.get("user_display_content")
            if isinstance(display, dict):
                text = display_content_text(cast("JsonDict", display))
                if text:
                    return text
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return strip_workspace_note(content)
    return ""


def _looks_like_path(value: str) -> bool:
    """Heuristic: does *value* look like a file path or filename?

    Excludes values containing whitespace so that shell command strings (e.g.
    ``ls -R data/x``) aren't mistaken for paths; imaging paths in this domain do
    not contain spaces.
    """
    if not value or any(c.isspace() for c in value):
        return False
    return "/" in value or bool(_EXT_RE.search(value))


def _structured_output_paths(result_text: str) -> dict[str, str]:
    """Extract ``key → path`` pairs from a tool result's structured output."""
    import ast

    match = re.search(r"structured:\s*(\{.*\})", result_text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in cast("JsonDict", parsed).items():
        if isinstance(value, str) and _looks_like_path(value):
            out[str(key)] = value
    return out


# FastMCP wraps any exception a tool raises into a result whose text begins with
# "Error executing tool <name>:" and sets the MCP ``isError`` flag. vibe-acp,
# however, renders that result into ``messages.jsonl`` as ``ok: True`` (its
# ``ok:`` tracks JSON-RPC success, not the tool's ``isError``), so by the time
# distillation reads the transcript the wrapper text is the only failure signal
# left. ``_NONZERO_EXIT_RE`` additionally catches "(exit N)" for tools that
# embed a subprocess exit code in the message.
_TOOL_ERROR_MARKER: str = "Error executing tool"
_NONZERO_EXIT_RE = re.compile(r"\(exit ([1-9]\d*)\)")


def _is_failed_result(result_text: str) -> bool:
    """Detect a failed tool result so dead-ends are excluded from the recipe."""
    return (
        "ok: False" in result_text
        or "returncode: 1" in result_text
        or _TOOL_ERROR_MARKER in result_text
        or _NONZERO_EXIT_RE.search(result_text) is not None
    )


# Markers vibe-acp writes into a tool result when the user rejected/cancelled the
# call, so it never actually ran (see vibe ``get_user_cancellation_message`` and
# the ACP reject path). Such calls must not become workflow steps.
_REJECTION_MARKERS: tuple[str, ...] = (
    "User rejected the tool call",
    "user_cancellation",  # the <user_cancellation> tag wrapping skip/cancel text
    "skipped by user",
    "interrupted by user",
    "User cancelled the operation",
)


def _is_rejected_result(result_text: str) -> bool:
    """Detect a tool call the user rejected or cancelled (so it didn't execute)."""
    return any(marker in result_text for marker in _REJECTION_MARKERS)


def _iter_tool_calls(messages: list[JsonDict]) -> list[tuple[str, JsonDict, str]]:
    """Yield ``(tool_name, arguments, result_text)`` for each completed tool call.

    Pairs each assistant ``tool_calls`` entry with its matching ``role="tool"``
    result by ``tool_call_id``. Calls without a result (interrupted) are skipped.
    """
    results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            if isinstance(call_id, str):
                results[call_id] = str(msg.get("content") or "")

    calls: list[tuple[str, JsonDict, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in cast("list[JsonDict]", msg.get("tool_calls") or []):
            fn = cast("JsonDict", tc.get("function") or {})
            name = str(fn.get("name") or "")
            call_id = str(tc.get("id") or "")
            if not name or call_id not in results:
                continue
            try:
                args = cast("JsonDict", json.loads(str(fn.get("arguments") or "{}")))
            except json.JSONDecodeError:
                args = {}
            calls.append((name, args, results[call_id]))
    return calls


# ── Deterministic recipe extraction ──────────────────────────────────────────


def _manual_step_label(tool: str, arguments: JsonDict) -> str:
    """Describe a dropped built-in step for the workflow's manual-steps note.

    For shell tools the command is included (it's the actual action); other
    built-ins are labelled by tool name alone.
    """
    if tool in _SHELL_TOOLS:
        command = arguments.get("command")
        if isinstance(command, str) and command.strip():
            return f"builtin:{tool} `{command.strip()}`"
    return f"builtin:{tool}"


_DIR_REF_RE = re.compile(r"dir\((.+)\)")


def _derive_container_dir_defaults(inputs: dict[str, WorkflowInput]) -> None:
    """Give an input that was another input's folder a derived default.

    A tool's ``output_dir`` is usually the directory its input file already sits
    in, so replay asks for the same thing twice. It stays a declared input —
    the session had it, and a caller may legitimately want the results
    somewhere else — but gains a default so it need not be retyped, and so the
    outputs follow the file being replayed on rather than the folder the
    workflow was recorded from.

    Only an unambiguous parent-of relationship earns a default. With two inputs
    in the same folder there is no principled anchor, and silently picking one
    would make the destination follow whichever happened to be lifted first.
    """
    for path, inp in inputs.items():
        anchors = [
            other_inp.name
            for other, other_inp in inputs.items()
            if other != path and str(PurePosixPath(other).parent) == path.rstrip("/")
        ]
        if len(anchors) == 1:
            inp.default = f"{{{{dir({anchors[0]})}}}}"


def build_recipe(
    messages: list[JsonDict],
    *,
    server_names: list[str],
    name: str,
    description: str,
) -> Recipe:
    """Extract a parameterized :class:`Recipe` from raw session messages.

    File paths are lifted into placeholders: a path produced by an earlier step
    becomes ``{{stepM.<key>}}`` when reused, and any other path becomes a
    workflow input ``{{in_N}}`` (deduplicated by value).
    """
    recipe = Recipe(name=name, description=description)
    produced: dict[str, str] = {}  # concrete path → placeholder reference
    inputs: dict[str, WorkflowInput] = {}  # concrete path → input definition
    manual_steps: list[str] = []  # dropped built-in steps, kept as documentation

    step_no = 0
    for tool_name, args, result_text in _iter_tool_calls(messages):
        if (
            _is_exploratory_call(tool_name, args)
            or _is_failed_result(result_text)
            or _is_rejected_result(result_text)
        ):
            continue
        server, tool = provenance.split_tool_name(tool_name, server_names)

        # Internal pool-machinery tools (warmup) carry no workflow meaning — drop
        # them silently so they never become a replayable step.
        if tool in INTERNAL_TOOLS:
            continue
        # Built-in (non-MCP) tools can't be replayed deterministically. Drop them
        # from the recipe so the workflow stays replayable, but record them as
        # manual steps so the real work isn't silently lost (see render_skill_md).
        if server == "builtin":
            manual_steps.append(_manual_step_label(tool, args))
            continue
        step_no += 1

        # Parameterize input paths against earlier outputs / declared inputs.
        new_args: JsonDict = {}
        for key, value in args.items():
            if isinstance(value, str) and _looks_like_path(value):
                if value in produced:
                    new_args[key] = produced[value]
                elif value in inputs:
                    new_args[key] = f"{{{{{inputs[value].name}}}}}"
                else:
                    placeholder = f"in_{len(inputs) + 1}"
                    # Describe the input by its first use so the user knows what to
                    # supply on replay (e.g. "the input_path for medmcp-neuro:skull_strip").
                    inputs[value] = WorkflowInput(
                        name=placeholder,
                        example=value,
                        description=f"the {key} for {server}:{tool}",
                    )
                    new_args[key] = f"{{{{{placeholder}}}}}"
            else:
                new_args[key] = value

        # Register this step's outputs so later steps can reference them.
        produces: dict[str, str] = {}
        for out_key, out_path in _structured_output_paths(result_text).items():
            ref = f"step{step_no}.{out_key}"
            produced[out_path] = f"{{{{{ref}}}}}"
            produces[out_key] = ref

        recipe.steps.append(
            RecipeStep(server=server, tool=tool, arguments=new_args, produces=produces)
        )

    _derive_container_dir_defaults(inputs)
    recipe.inputs = list(inputs.values())
    recipe.manual_steps = manual_steps
    return recipe


def slugify(text: str) -> str:
    """Turn arbitrary text into a filesystem/skill-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "workflow"


def resolve_digest(image: str) -> str:
    """Resolve *image*'s digest best-effort; never raise (distillation must not fail)."""
    try:
        from medmcp import settings

        return settings.resolve_image_digest(image) or ""
    except Exception:
        return ""


def build_requirements(recipe: Recipe, manifest: JsonDict | None) -> list[StackRequirement]:
    """Build reproducibility requirements for *recipe* from the session *manifest*.

    Filtered to the stacks the recipe actually uses (built-in steps need none).
    Container stacks pin by image (+ best-effort digest); uv-tool stacks by version.
    A used stack absent from the manifest is still listed by name so an importer
    knows it is required.
    """
    used = sorted({s.server for s in recipe.steps if s.server != "builtin"})
    by_name = {
        str(s.get("name")): s for s in cast("list[JsonDict]", (manifest or {}).get("stacks") or [])
    }
    requires: list[StackRequirement] = []
    for server in used:
        entry = by_name.get(server)
        if entry is None:
            requires.append(StackRequirement(stack=server))
            continue
        image = str(entry.get("image") or "")
        requires.append(
            StackRequirement(
                stack=server,
                version=str(entry.get("version") or ""),
                image=image,
                digest=resolve_digest(image) if image else "",
            )
        )
    return requires


# ── Workflow directories ─────────────────────────────────────────────────────


def resolve_root(override: Path | None = None) -> Path:
    """Return the workflows root (``.vibe/workflows`` unless *override*), migrated.

    Every function here that takes a ``workflows_root`` keyword goes through
    this, so the old ``draft/``/``active/`` layout is folded in before any
    lookup or write (see :func:`medmcp.workflow.migrate_layout`).
    """
    root = override if override is not None else provenance.VIBE_HOME / "workflows"
    workflow.migrate_layout(root)
    return root


def load_recipe(directory: Path) -> Recipe:
    """Reconstruct a :class:`Recipe` from a workflow directory's ``recipe.yaml``."""
    return workflow.read_recipe(directory)


def _require_dir(name: str, root: Path) -> Path:
    """Return workflow *name*'s directory under *root* or raise ``FileNotFoundError``."""
    d = workflow.workflow_dir(root, name)
    if d is None:
        raise FileNotFoundError(f"no workflow named {name!r} (looked in {root})")
    return d


# ── Top-level distillation ───────────────────────────────────────────────────


def distill_session(
    session_id: str,
    *,
    name_hint: str = "",
    workflows_root: Path | None = None,
    chain_stop_ids: Collection[str] = (),
) -> Path:
    """Distill *session_id* into a workflow directory and return its path.

    Reads the session's ``messages.jsonl`` — concatenated across the session's
    compaction chain (vibe rolls a compacted conversation over to a new dir;
    see :func:`provenance.find_vibe_session_dirs`), so tool calls made after a
    compaction distill too — ``chain_stop_ids`` keeps the walk out of forks
    (pass the UI session registry's ids) — builds a parameterized recipe and
    writes it as ``<workflows_root>/<name>/recipe.yaml``.

    The workflow is named from *name_hint* — the chat's title, which the caller
    looks up — falling back to the request that opened the chat; that request
    is also the description. A name already in use gets a numeric suffix rather
    than replacing another workflow.

    Raises ``FileNotFoundError`` if the session's raw log cannot be located.
    """
    session_dirs = provenance.find_vibe_session_dirs(session_id, stop_ids=chain_stop_ids)
    if not session_dirs:
        raise FileNotFoundError(f"no vibe session log found for session {session_id!r}")
    message_paths = [d / "messages.jsonl" for d in session_dirs if (d / "messages.jsonl").exists()]
    if not message_paths:
        raise FileNotFoundError(f"no messages.jsonl in {session_dirs[0]}")

    messages = [m for path in message_paths for m in parse_messages_file(path)]
    manifest = provenance.read_manifest(session_id)
    server_names = (
        [str(s.get("name")) for s in cast("list[JsonDict]", manifest.get("stacks") or [])]
        if manifest is not None
        else []
    )

    context = _first_user_message(messages)
    if name_hint.strip():
        base = slugify(name_hint)
    elif context:
        base = slugify(context)
    else:
        base = f"workflow-{session_id[:8]}"
    recipe = build_recipe(
        messages,
        server_names=server_names,
        name=base,
        description=context[:120] if context else "Distilled MedMCP workflow",
    )
    recipe.requires = build_requirements(recipe, manifest)

    root = resolve_root(workflows_root)
    recipe.name = workflow.unique_name(root, base)
    target = root / recipe.name
    workflow.write_recipe(target, recipe)
    return target


def delete_workflow(name: str, *, workflows_root: Path | None = None) -> Path:
    """Delete workflow *name*; return the directory that was removed.

    Raises ``FileNotFoundError`` if no workflow with that name exists.
    """
    target = _require_dir(name, resolve_root(workflows_root))
    shutil.rmtree(target)
    return target


def rename_workflow(name: str, new_name: str, *, workflows_root: Path | None = None) -> Path:
    """Rename a workflow (its name in the recipe and its directory); return the new dir.

    Raises ``FileNotFoundError`` if *name* does not exist and ``FileExistsError``
    if the slug of *new_name* already belongs to another workflow — a rename
    never replaces one.
    """
    root = resolve_root(workflows_root)
    src = _require_dir(name, root)
    new_slug = slugify(new_name)
    dst = root / new_slug
    if dst != src and dst.exists():
        raise FileExistsError(f"a workflow named {new_slug!r} already exists")
    recipe = load_recipe(src)
    recipe.name = new_slug
    workflow.write_recipe(src, recipe)
    if dst != src:
        shutil.move(str(src), str(dst))
    return dst
