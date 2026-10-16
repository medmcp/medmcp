"""Shared material for the replay tests: a two-step neuro recipe and the stack it needs."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from medmcp import replay
from medmcp.workflow import Recipe, RecipeStep, WorkflowInput

JsonDict = dict[str, Any]

# The shape of ``replay.mcp_caller`` — what a test double standing in for it must return.
McpCallerFactory = Callable[..., contextlib.AbstractAsyncContextManager[replay.ToolCaller]]


def two_step_recipe() -> Recipe:
    """Skull strip, then register the stripped brain: step 2 consumes step 1's output."""
    return Recipe(
        name="wf",
        description="d",
        inputs=[WorkflowInput(name="in_1", example="/orig.nii")],
        steps=[
            RecipeStep(
                server="medmcp-neuro",
                tool="skull_strip",
                arguments={"input_path": "{{in_1}}", "device": "cuda"},
                produces={"brain_path": "step1.brain_path"},
            ),
            RecipeStep(
                server="medmcp-neuro",
                tool="register_to_template",
                arguments={"input_path": "{{step1.brain_path}}"},
                produces={"registered_path": "step2.registered_path"},
            ),
        ],
    )


def neuro_servers() -> list[JsonDict]:
    """The one installed stack the two-step recipe needs."""
    return [{"name": "medmcp-neuro", "command": "neuro", "args": []}]
