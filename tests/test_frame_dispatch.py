"""Tests for the ACP frame dispatch path.

Covers ``_handle_tool_call``, ``_handle_tool_call_update``, and
``_process_session_frame``. These functions touch ``cl.Step``, ``cl.Message``,
and ``_client.respond``; all three are patched out with
``unittest.mock.AsyncMock`` so the tests stay hermetic.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medmcp.app import (
    _handle_tool_call,
    _handle_tool_call_update,
    _process_session_frame,
)

# ── Helpers ────────────────────────────────────────────────


def _make_step_mock() -> MagicMock:
    """Build a ``cl.Step`` stand-in with awaitable ``send`` / ``update``."""
    step = MagicMock(name="Step")
    step.send = AsyncMock()
    step.update = AsyncMock()
    # name/input/output are plain attributes the dispatch code assigns to.
    step.name = None
    step.input = None
    step.output = None
    return step


def _make_message_mock() -> MagicMock:
    """Build a ``cl.Message`` stand-in with awaitable ``stream_token``."""
    msg = MagicMock(name="Message")
    msg.stream_token = AsyncMock()
    msg.send = AsyncMock()
    msg.update = AsyncMock()
    return msg


# ── _handle_tool_call ─────────────────────────────────────


async def test_tool_call_creates_step_and_caches_info() -> None:
    """A first ``tool_call`` event should construct a step and cache title/rawInput."""
    step = _make_step_mock()
    tool_steps: dict[str, Any] = {}
    tool_call_info: dict[str, Any] = {}

    with patch("medmcp.app.cl.Step", return_value=step) as step_ctor:
        await _handle_tool_call(
            {
                "toolCallId": "tc1",
                "title": "bash: ls",
                "rawInput": {"cmd": "ls"},
            },
            tool_steps,
            tool_call_info,
            parent_id="p1",
        )

    # A new step was constructed with the right kwargs.
    step_ctor.assert_called_once_with(name="bash: ls", type="tool", parent_id="p1")
    # The cache holds both fields for later permission backfill.
    assert tool_call_info["tc1"]["title"] == "bash: ls"
    assert tool_call_info["tc1"]["rawInput"] == {"cmd": "ls"}
    # The step was sent and its input is the indented JSON form.
    step.send.assert_awaited_once()
    assert step.input is not None
    assert json.loads(step.input) == {"cmd": "ls"}
    # And it was registered for dedup on the next event.
    assert tool_steps["tc1"] is step


async def test_tool_call_dedup_on_repeat_updates_existing_step() -> None:
    """A second ``tool_call`` with the same id should update the existing step.

    vibe-acp emits ``tool_call`` twice (first to announce the tool name, then
    again with the resolved ``rawInput``). The second emission must NOT create
    a duplicate UI step.
    """
    step = _make_step_mock()
    tool_steps: dict[str, Any] = {}
    tool_call_info: dict[str, Any] = {}

    with patch("medmcp.app.cl.Step", return_value=step) as step_ctor:
        # First emission: title only.
        await _handle_tool_call(
            {"toolCallId": "tc1", "title": "bash"},
            tool_steps,
            tool_call_info,
            parent_id=None,
        )
        # Second emission: same id, with rawInput and refined title.
        await _handle_tool_call(
            {"toolCallId": "tc1", "title": "bash: ls", "rawInput": {"cmd": "ls"}},
            tool_steps,
            tool_call_info,
            parent_id=None,
        )

    # Constructor only called once — the dedup branch took over.
    step_ctor.assert_called_once()
    # The existing step was updated, not replaced.
    assert step.name == "bash: ls"
    assert step.input is not None and json.loads(step.input) == {"cmd": "ls"}
    step.update.assert_awaited()


async def test_tool_call_falls_back_to_generic_title() -> None:
    """A first ``tool_call`` with no title should use the generic ``"tool"`` label."""
    step = _make_step_mock()
    tool_steps: dict[str, Any] = {}
    tool_call_info: dict[str, Any] = {}

    with patch("medmcp.app.cl.Step", return_value=step) as step_ctor:
        await _handle_tool_call(
            {"toolCallId": "tc1"},
            tool_steps,
            tool_call_info,
            parent_id=None,
        )

    step_ctor.assert_called_once_with(name="tool", type="tool", parent_id=None)


# ── _handle_tool_call_update ──────────────────────────────


async def test_tool_call_update_prefers_raw_output_over_text_blocks() -> None:
    """When both ``rawOutput`` and content text are present, ``rawOutput`` wins."""
    step = _make_step_mock()
    tool_steps: dict[str, Any] = {"tc1": step}

    await _handle_tool_call_update(
        {
            "toolCallId": "tc1",
            "status": "completed",
            "rawOutput": {"exit": 0},
            "content": [
                {"type": "content", "content": {"text": "stdout-text"}},
            ],
        },
        tool_steps,
    )

    assert step.output is not None
    assert json.loads(step.output) == {"exit": 0}
    step.update.assert_awaited_once()


async def test_tool_call_update_falls_back_to_text_blocks() -> None:
    """Without ``rawOutput``, content text blocks should be joined into output."""
    step = _make_step_mock()
    tool_steps: dict[str, Any] = {"tc1": step}

    await _handle_tool_call_update(
        {
            "toolCallId": "tc1",
            "status": "completed",
            "content": [
                {"type": "content", "content": {"text": "line1"}},
                {"type": "content", "content": {"text": "line2"}},
            ],
        },
        tool_steps,
    )

    assert step.output == "line1\nline2"
    step.update.assert_awaited_once()


async def test_tool_call_update_only_finalizes_on_terminal_status() -> None:
    """``step.update`` should only fire when status is ``completed`` or ``failed``."""
    step = _make_step_mock()
    tool_steps: dict[str, Any] = {"tc1": step}

    await _handle_tool_call_update(
        {"toolCallId": "tc1", "status": "in_progress", "rawOutput": "x"},
        tool_steps,
    )
    step.update.assert_not_awaited()

    await _handle_tool_call_update(
        {"toolCallId": "tc1", "status": "completed", "rawOutput": "x"},
        tool_steps,
    )
    assert step.update.await_count == 1

    await _handle_tool_call_update(
        {"toolCallId": "tc1", "status": "failed", "rawOutput": "x"},
        tool_steps,
    )
    assert step.update.await_count == 2


async def test_tool_call_update_ignores_unknown_tool_call_id() -> None:
    """An update for a tool call we never registered should be a no-op."""
    tool_steps: dict[str, Any] = {}
    # Must not raise.
    await _handle_tool_call_update(
        {"toolCallId": "ghost", "status": "completed", "rawOutput": "x"},
        tool_steps,
    )


# ── _process_session_frame ────────────────────────────────


async def test_process_frame_streams_agent_message_chunk_text() -> None:
    """A text agent_message_chunk should stream into the assistant message."""
    msg = _make_message_mock()
    getter = AsyncMock(return_value=msg)

    await _process_session_frame(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "hello"},
                }
            },
        },
        assistant_msg_getter=getter,
        tool_steps={},
        tool_call_info={},
        parent_id=None,
    )

    getter.assert_awaited_once()
    msg.stream_token.assert_awaited_once_with("hello")


async def test_process_frame_ignores_non_text_agent_message_chunk() -> None:
    """A non-text content type should not trigger streaming.

    The dispatcher must not crash on unknown content types — vibe-acp may emit
    image or other content types we don't render today.
    """
    getter = AsyncMock()

    await _process_session_frame(
        {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "image", "url": "..."},
                }
            },
        },
        assistant_msg_getter=getter,
        tool_steps={},
        tool_call_info={},
        parent_id=None,
    )

    getter.assert_not_awaited()


async def test_process_frame_request_permission_backfills_from_cache() -> None:
    """``request_permission`` should backfill title/rawInput from the tool_call cache.

    vibe-acp's ``session/request_permission`` payload only carries
    ``toolCallId`` — without backfill, the approval dialog would show
    ``<unknown>`` and the user would have no idea what they're approving.
    """
    tool_call_info = {
        "tc1": {"title": "bash: ls", "rawInput": {"cmd": "ls"}},
    }

    captured: dict[str, Any] = {}

    async def fake_ask(tc: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any]:
        captured["tc"] = tc
        captured["options"] = options
        return {"outcome": "selected", "optionId": "approve"}

    fake_respond = AsyncMock()

    with (
        patch("medmcp.app._ask_user_for_permission", side_effect=fake_ask),
        patch("medmcp.app._client.respond", new=fake_respond),
    ):
        await _process_session_frame(
            {
                "method": "session/request_permission",
                "id": 42,
                "params": {
                    "toolCall": {"toolCallId": "tc1"},
                    "options": [{"optionId": "approve", "name": "Approve"}],
                },
            },
            assistant_msg_getter=AsyncMock(),
            tool_steps={},
            tool_call_info=tool_call_info,
            parent_id=None,
        )

    # title and rawInput were backfilled from the cache before being shown.
    assert captured["tc"]["title"] == "bash: ls"
    assert captured["tc"]["rawInput"] == {"cmd": "ls"}
    # The response was sent back to vibe-acp under the original request id.
    fake_respond.assert_awaited_once_with(
        42, {"outcome": {"outcome": "selected", "optionId": "approve"}}
    )


async def test_process_frame_request_permission_ignores_non_int_id() -> None:
    """A request_permission frame with a non-int id should be a silent no-op."""
    fake_ask = AsyncMock()
    fake_respond = AsyncMock()

    with (
        patch("medmcp.app._ask_user_for_permission", new=fake_ask),
        patch("medmcp.app._client.respond", new=fake_respond),
    ):
        await _process_session_frame(
            {
                "method": "session/request_permission",
                "id": "not-an-int",
                "params": {"toolCall": {"toolCallId": "tc1"}, "options": []},
            },
            assistant_msg_getter=AsyncMock(),
            tool_steps={},
            tool_call_info={},
            parent_id=None,
        )

    # Neither the prompt nor the response was sent.
    fake_ask.assert_not_called()
    fake_respond.assert_not_called()


@pytest.mark.parametrize("method", ["session/something_else", "unrelated", None])
async def test_process_frame_ignores_unknown_methods(method: str | None) -> None:
    """Frames whose method we don't handle should be silent no-ops."""
    getter = AsyncMock()
    await _process_session_frame(
        {"method": method, "params": {}},
        assistant_msg_getter=getter,
        tool_steps={},
        tool_call_info={},
        parent_id=None,
    )
    getter.assert_not_awaited()
