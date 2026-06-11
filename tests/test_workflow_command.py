"""Tests for the 'Save workflow' composer command definition."""

from __future__ import annotations

from medmcp.app import (
    MANAGE_WORKFLOWS_COMMAND,
    SAVE_WORKFLOW_COMMAND,
    _workflow_commands,  # pyright: ignore[reportPrivateUsage]
)


def test_workflow_command_shape() -> None:
    """The composer exposes Save and Manage buttons with the expected ids."""
    commands = _workflow_commands()
    assert [c["id"] for c in commands] == [SAVE_WORKFLOW_COMMAND, MANAGE_WORKFLOWS_COMMAND]
    for cmd in commands:
        assert cmd["button"] is True
        # Not persistent: one-shot actions, not sticky modes.
        assert cmd["persistent"] is False
        assert cmd["description"]
