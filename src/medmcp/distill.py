"""Tier-2 distillation: turn a session's raw log into a reusable workflow.

Reads vibe-acp's ``messages.jsonl`` (the authoritative record of exact tool
names, arguments, and structured results), filters out exploratory/failed
calls, and lifts concrete file paths into named placeholders to produce a
:class:`~medmcp.workflow.Recipe`. The recipe is emitted two ways from one
distillation (Option C):

- ``recipe.yaml`` — machine-readable + replayable.
- ``SKILL.md``    — frontmatter + ``## Steps`` / ``## Gotchas``, droppable into a
  ``skill_paths`` directory so it plugs into the existing skill system.

The prose (workflow name, description, narrative steps, gotchas) is written by a
hybrid pass: the step sequence is extracted deterministically, then the local
Ollama model is asked to write the human-facing narrative. If the model is
unavailable the narrative falls back to a mechanical rendering so distillation
never hard-fails.

Output lands in ``.vibe/workflows/draft/<name>/`` for human review; promotion to
an active ``skill_paths`` location is a separate, deliberate step.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, cast

import httpx
import yaml

from medmcp import provenance
from medmcp.workflow import Recipe, RecipeStep, StackRequirement, WorkflowInput
from medmcp.workspace_note import strip_workspace_note

JsonDict = dict[str, Any]

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4-medmcp")
PROSE_TIMEOUT: float = 60.0

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

    The workspace server appends a "[workspace context: …]" note to the prompt
    text so the agent can resolve "this image" (server.py:_workspace_note). That
    note is live-turn metadata, not part of the user's request, so it is stripped
    here (via :func:`strip_workspace_note`) — it would otherwise pollute the
    distilled workflow's name and description.
    """
    for msg in messages:
        if msg.get("role") == "user" and not msg.get("injected"):
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


# ── Hybrid prose pass ────────────────────────────────────────────────────────


def _prose_prompt(recipe: Recipe, context: str) -> str:
    """Build the prompt that asks the model to narrate the distilled steps."""
    steps_desc = "\n".join(
        f"{i}. {s.server}:{s.tool} arguments={json.dumps(s.arguments)}"
        for i, s in enumerate(recipe.steps, start=1)
    )
    manual_note = ""
    if recipe.manual_steps:
        manual_note = (
            "\n\nManual steps (built-in tools that were used but CANNOT be replayed "
            "automatically) — call these out in gotchas_markdown so the reader performs "
            "them by hand:\n" + "\n".join(f"- {m}" for m in recipe.manual_steps)
        )
    return (
        "You are documenting a medical-imaging workflow so a colleague can reuse it "
        "on their own data. Below is the original user request and the exact sequence "
        "of tool calls that were executed. Write a concise, reusable workflow.\n\n"
        "GENERALIZE — this is the most important rule. The workflow must be reusable "
        "beyond the specific scan it was run on:\n"
        "- Describe inputs at the imaging-modality / anatomy level, NOT the specific "
        "contrast, sequence, or filename. For example, a workflow run on a T1 scan is a "
        "brain MRI workflow; refer to 'a brain MRI scan', not 'a T1 scan' or "
        "'t1n_3d.nii.gz'.\n"
        "- Only mention a specific contrast/sequence (T1, FLAIR, T2, b0, …) when a step "
        "genuinely depends on it; otherwise keep it generic.\n"
        "- The name and description must be modality-level and not tied to one dataset, "
        "subject, or path.\n\n"
        "NAME THE TOOLS — every step in steps_markdown MUST explicitly name the exact "
        "tool it uses, written in backticks as `server:tool` (e.g. `medmcp-neuro:skull_strip`, "
        "or `builtin:bash` for a built-in tool). This tells the reader precisely which "
        "tool/skill to call. Never omit, invent, or rephrase a tool name — use only the "
        "tools listed under 'Executed steps' below, exactly as written. Keep the surrounding "
        "description generic, but the tool name itself must be verbatim.\n\n"
        "Respond with ONLY a JSON object — no markdown fences, no extra text:\n"
        '{"name": "<short-kebab-case-name>", "description": "<one sentence>", '
        '"steps_markdown": "<numbered markdown list; each step names its `server:tool`>", '
        '"gotchas_markdown": "<markdown bullet list of caveats, or empty string>"}\n\n'
        f"Original request:\n{context}\n\n"
        f"Executed steps (tool names are authoritative — reuse them verbatim):\n{steps_desc}\n"
        f"{manual_note}"
    )


def _parse_prose_response(raw_text: str) -> JsonDict | None:
    """Extract the prose JSON object from the model reply (strips code fences)."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Greedy match so nested braces in the markdown body aren't truncated.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return cast("JsonDict", parsed) if isinstance(parsed, dict) else None


def generate_prose(recipe: Recipe, context: str) -> JsonDict | None:
    """Ask the local model to narrate *recipe*; return prose dict or ``None``.

    Failures are swallowed and reported as ``None`` so distillation can fall
    back to a mechanical rendering rather than blocking.
    """
    prompt = _prose_prompt(recipe, context)
    try:
        with httpx.Client(timeout=PROSE_TIMEOUT) as client:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.3, "num_predict": 2048},
                },
            )
            resp.raise_for_status()
            data = cast("JsonDict", resp.json())
            message = cast("JsonDict", data.get("message") or {})
            raw_text = str(message.get("content") or "").strip()
    except Exception:
        return None
    return _parse_prose_response(raw_text)


# ── SKILL.md rendering ───────────────────────────────────────────────────────


def _mechanical_steps_markdown(recipe: Recipe) -> str:
    """Render a plain numbered list of steps when no LLM narrative is available."""
    lines: list[str] = []
    for i, step in enumerate(recipe.steps, start=1):
        args = json.dumps(step.arguments)
        lines.append(f"{i}. **`{step.server}:{step.tool}`** — `{args}`")
    return "\n".join(lines) or "_No reusable steps were extracted._"


def _required_tools(recipe: Recipe) -> list[tuple[str, str]]:
    """Return the distinct ``(server, tool)`` pairs the recipe uses, first-seen order."""
    seen: list[tuple[str, str]] = []
    for step in recipe.steps:
        pair = (step.server, step.tool)
        if pair not in seen:
            seen.append(pair)
    return seen


def _manual_steps_markdown(recipe: Recipe) -> str:
    """Render the note about manual (built-in) steps replay can't run, or ``""``."""
    if not recipe.manual_steps:
        return ""
    lines = [
        "This workflow originally included manual steps using built-in tools that "
        "the replay engine cannot run automatically. Perform them by hand where the "
        "pipeline needs them:",
    ]
    lines += [f"- `{m}`" for m in recipe.manual_steps]
    return "\n".join(lines)


def _requirements_markdown(recipe: Recipe) -> str:
    """Render the stacks the workflow needs, pinned for reproducibility.

    Derived from the recipe's ``requires`` (captured at distillation): container
    stacks show their image (+ digest when resolved), uv-tool stacks their version.
    """
    lines: list[str] = []
    for req in recipe.requires:
        if req.image:
            pin = f"image `{req.image}`" + (f" (`{req.digest}`)" if req.digest else "")
        elif req.version:
            pin = f"version `{req.version}`"
        else:
            pin = ""
        lines.append(f"- `{req.stack}`" + (f" — {pin}" if pin else ""))
    return "\n".join(lines)


def _tools_markdown(recipe: Recipe) -> str:
    """Render the explicit list of tools/skills the workflow requires.

    Derived directly from the recipe (not the LLM) so it is always accurate: each
    line names a ``server:tool`` and the stack it comes from.
    """
    lines: list[str] = []
    for server, tool in _required_tools(recipe):
        if server == "builtin":
            lines.append(f"- `{tool}` — built-in tool")
        else:
            lines.append(f"- `{server}:{tool}` — from the `{server}` stack")
    return "\n".join(lines)


def render_skill_md(recipe: Recipe, prose: JsonDict | None) -> str:
    """Render the ``SKILL.md`` document for *recipe* (using *prose* if present)."""
    description = recipe.description
    steps_md = _mechanical_steps_markdown(recipe)
    gotchas_md = ""
    if prose is not None:
        description = str(prose.get("description") or description)
        steps_md = str(prose.get("steps_markdown") or steps_md)
        gotchas_md = str(prose.get("gotchas_markdown") or "")

    # Always surface dropped manual steps (deterministic), ahead of any LLM gotchas.
    manual_md = _manual_steps_markdown(recipe)
    gotchas_md = "\n\n".join(part for part in (manual_md, gotchas_md.strip()) if part)

    title = recipe.name.replace("-", " ").strip().capitalize()
    parts: list[str] = [
        "---",
        f"name: {recipe.name}",
        f"description: {description}",
        "---",
        "",
        f"# {title} workflow",
    ]
    tools_md = _tools_markdown(recipe)
    if tools_md:
        parts += ["", "## Tools", "", tools_md]
    reqs_md = _requirements_markdown(recipe)
    if reqs_md:
        parts += ["", "## Requirements", "", reqs_md]
    parts += ["", "## Steps", "", steps_md]
    if gotchas_md.strip():
        parts += ["", "## Gotchas", "", gotchas_md.strip()]
    if recipe.inputs:
        parts += ["", "## Inputs", ""]
        for i in recipe.inputs:
            desc = f" — {i.description}" if i.description else ""
            parts.append(f"- `{{{{{i.name}}}}}`{desc} (e.g. `{i.example}`)")
    return "\n".join(parts).rstrip() + "\n"


# ── Draft persistence & editing ──────────────────────────────────────────────


def _workflows_root(workflows_root: Path | None) -> Path:
    """Resolve the workflows root, defaulting to ``.vibe/workflows``."""
    return workflows_root if workflows_root is not None else provenance.VIBE_HOME / "workflows"


def _draft_dir(name: str, workflows_root: Path | None) -> Path:
    """Return the draft directory for workflow *name*."""
    return _workflows_root(workflows_root) / "draft" / name


def _write_draft(draft_dir: Path, recipe: Recipe, prose: JsonDict | None) -> None:
    """Write ``recipe.yaml``, ``SKILL.md`` and ``prose.json`` for a draft.

    ``prose.json`` caches the LLM narrative so rename/refine can re-render the
    ``SKILL.md`` without re-running distillation.
    """
    draft_dir.mkdir(parents=True, exist_ok=True)
    recipe_yaml = yaml.safe_dump(recipe.to_dict(), sort_keys=False, allow_unicode=True)
    (draft_dir / "recipe.yaml").write_text(recipe_yaml, encoding="utf-8")
    (draft_dir / "SKILL.md").write_text(render_skill_md(recipe, prose), encoding="utf-8")
    (draft_dir / "prose.json").write_text(
        json.dumps(prose) if prose is not None else "null", encoding="utf-8"
    )


def _read_prose(draft_dir: Path) -> JsonDict | None:
    """Read the cached prose for a draft, or ``None`` if absent/mechanical."""
    path = draft_dir / "prose.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return cast("JsonDict", data) if isinstance(data, dict) else None


def load_recipe(draft_dir: Path) -> Recipe:
    """Reconstruct a :class:`Recipe` from a draft's ``recipe.yaml``."""
    data = cast("JsonDict", yaml.safe_load((draft_dir / "recipe.yaml").read_text(encoding="utf-8")))
    inputs = [
        WorkflowInput(
            name=str(i.get("name", "")),
            example=str(i.get("example", "")),
            description=str(i.get("description", "")),
        )
        for i in cast("list[JsonDict]", data.get("inputs", []))
    ]
    steps = [
        RecipeStep(
            server=str(s.get("server", "")),
            tool=str(s.get("tool", "")),
            arguments=cast("JsonDict", s.get("arguments", {})),
            produces=cast("dict[str, str]", s.get("produces", {})),
        )
        for s in cast("list[JsonDict]", data.get("steps", []))
    ]
    requires = [
        StackRequirement(
            stack=str(r.get("stack", "")),
            version=str(r.get("version", "")),
            image=str(r.get("image", "")),
            digest=str(r.get("digest", "")),
        )
        for r in cast("list[JsonDict]", data.get("requires", []))
    ]
    manual_steps = [str(m) for m in cast("list[Any]", data.get("manual_steps", []))]
    return Recipe(
        name=str(data.get("name", "")),
        description=str(data.get("description", "")),
        inputs=inputs,
        steps=steps,
        requires=requires,
        manual_steps=manual_steps,
    )


# ── Top-level distillation ───────────────────────────────────────────────────


def distill_session(
    session_id: str,
    *,
    use_llm: bool = True,
    workflows_root: Path | None = None,
) -> Path:
    """Distill *session_id* into a draft workflow directory and return its path.

    Reads the session's ``messages.jsonl`` — concatenated across the session's
    compaction chain (vibe rolls a compacted conversation over to a new dir;
    see :func:`provenance.find_vibe_session_dirs`), so tool calls made after a
    compaction distill too — builds a parameterized recipe, adds a (hybrid)
    prose narrative, and writes ``recipe.yaml`` + ``SKILL.md`` into
    ``<workflows_root>/draft/<name>/`` for human review.

    Raises ``FileNotFoundError`` if the session's raw log cannot be located.
    """
    session_dirs = provenance.find_vibe_session_dirs(session_id)
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
    fallback_name = slugify(context) if context else f"workflow-{session_id[:8]}"
    recipe = build_recipe(
        messages,
        server_names=server_names,
        name=fallback_name,
        description=context[:120] if context else "Distilled MedMCP workflow",
    )

    recipe.requires = build_requirements(recipe, manifest)

    prose = generate_prose(recipe, context) if use_llm else None
    if prose is not None and isinstance(prose.get("name"), str):
        recipe.name = slugify(str(prose["name"]))
        recipe.description = str(prose.get("description") or recipe.description)

    draft_dir = _draft_dir(recipe.name, workflows_root)
    _write_draft(draft_dir, recipe, prose)
    return draft_dir


def promote_draft(name: str, *, workflows_root: Path | None = None) -> Path:
    """Move draft workflow *name* into ``active/`` and return the new path.

    Promotion makes the workflow discoverable as a skill (the active directory is
    added to ``skill_paths``). Raises ``FileNotFoundError`` if no such draft exists.
    """
    src = _draft_dir(name, workflows_root)
    if not (src / "SKILL.md").exists():
        raise FileNotFoundError(f"no draft workflow named {name!r} (looked in {src})")
    dst = _workflows_root(workflows_root) / "active" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return dst


def unpromote_workflow(name: str, *, workflows_root: Path | None = None) -> Path:
    """Move a promoted workflow from ``active/`` back to ``draft/`` for editing.

    The inverse of :func:`promote_draft`: it returns the workflow to the draft
    state so it can be renamed/refined/re-tested, then promoted again. Returns the
    draft path. Raises ``FileNotFoundError`` if no promoted workflow has that name.
    """
    src = _workflows_root(workflows_root) / "active" / name
    if not (src / "SKILL.md").exists():
        raise FileNotFoundError(f"no promoted workflow named {name!r} (looked in {src})")
    dst = _draft_dir(name, workflows_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    return dst


def discard_draft(name: str, *, workflows_root: Path | None = None) -> None:
    """Delete a draft workflow. Raises ``FileNotFoundError`` if it doesn't exist."""
    src = _draft_dir(name, workflows_root)
    if not (src / "SKILL.md").exists():
        raise FileNotFoundError(f"no draft workflow named {name!r} (looked in {src})")
    shutil.rmtree(src)


def delete_workflow(name: str, *, workflows_root: Path | None = None) -> Path:
    """Delete a personal workflow by *name* from ``active/`` or ``draft/``.

    Returns the directory that was removed. Raises ``FileNotFoundError`` if no
    workflow with that name exists in either location.
    """
    root = _workflows_root(workflows_root)
    for kind in ("active", "draft"):
        target = root / kind / name
        if (target / "SKILL.md").exists():
            shutil.rmtree(target)
            return target
    raise FileNotFoundError(f"no workflow named {name!r} in {root}")


def rename_draft(name: str, new_name: str, *, workflows_root: Path | None = None) -> Path:
    """Rename a draft workflow (its name, files, and directory). Returns the new dir."""
    src = _draft_dir(name, workflows_root)
    if not (src / "SKILL.md").exists():
        raise FileNotFoundError(f"no draft workflow named {name!r} (looked in {src})")
    new_slug = slugify(new_name)
    recipe = load_recipe(src)
    recipe.name = new_slug
    prose = _read_prose(src)
    if prose is not None:
        prose["name"] = new_slug
    # Rewrite in place (so name fields update), then relocate if the slug changed.
    _write_draft(src, recipe, prose)
    dst = src.parent / new_slug
    if dst != src:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    return dst


def refine_draft(name: str, instruction: str, *, workflows_root: Path | None = None) -> Path:
    """Regenerate a draft's narrative from a plain-language *instruction*.

    Re-runs the prose model with the user's instruction, keeping the workflow's
    identity (name) and step recipe unchanged. Raises ``FileNotFoundError`` if the
    draft is missing, or ``RuntimeError`` if the model returns nothing.
    """
    src = _draft_dir(name, workflows_root)
    if not (src / "SKILL.md").exists():
        raise FileNotFoundError(f"no draft workflow named {name!r} (looked in {src})")
    recipe = load_recipe(src)
    context = (
        f"Current workflow description: {recipe.description}\n"
        f"Revise the workflow per this instruction: {instruction}"
    )
    prose = generate_prose(recipe, context)
    if prose is None:
        raise RuntimeError("the model did not return a refined workflow")
    # Refine content only; keep the existing name/identity.
    prose["name"] = recipe.name
    recipe.description = str(prose.get("description") or recipe.description)
    _write_draft(src, recipe, prose)
    return src
