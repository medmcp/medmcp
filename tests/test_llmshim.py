"""Tests for the Ollama Glimmer tool-call shim (``medmcp.llmshim``).

The shapes exercised here are the ones measured against the live model: Ollama
rejects any ``arguments`` that is not a JSON-object string, and a silently
truncated stream is identifiable by its missing ``finish_reason``.
"""

from __future__ import annotations

import json
from typing import Any

# pyright: reportPrivateUsage=false
import httpx
import pytest

from medmcp import llmshim
from medmcp.llmshim import (
    DEFAULT_RENAMES,
    StreamOutcome,
    failure_event,
    parse_sse_line,
    repair_request,
    restore_response,
    sanitize_arguments,
)

JsonDict = dict[str, Any]


# ── sanitize_arguments ─────────────────────────────────────


def test_sanitize_arguments_passes_valid_object_strings_through() -> None:
    """A well-formed object string is the one shape Ollama accepts; leave it alone."""
    raw = '{"path": "/data/a.nii.gz"}'
    assert sanitize_arguments(raw) == raw


def test_sanitize_arguments_rejects_every_shape_ollama_400s_on() -> None:
    """Each of these produced ``400 invalid tool call arguments`` against the model."""
    for bad in (None, "", "   ", '{"path": "/a', "[]", "null", 42, ["a"]):
        assert sanitize_arguments(bad) == "{}", f"not neutralised: {bad!r}"


def test_sanitize_arguments_reserialises_a_dict_rather_than_dropping_it() -> None:
    """A dict is still meaningful, so re-serialise instead of discarding its content."""
    out = sanitize_arguments({"path": "/a"})
    assert json.loads(out) == {"path": "/a"}


# ── repair_request ─────────────────────────────────────────


def _payload_with_skill() -> JsonDict:
    """Build a request carrying the colliding tool name in all three places."""
    return {
        "model": "muse-medmcp",
        "tools": [
            {"type": "function", "function": {"name": "skill", "parameters": {}}},
            {"type": "function", "function": {"name": "bash", "parameters": {}}},
        ],
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "skill", "arguments": ""},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "skill", "content": "ok"},
        ],
    }


def test_repair_request_renames_the_colliding_tool_everywhere() -> None:
    """The rename must reach tools, prior tool calls, and tool results alike."""
    out = repair_request(_payload_with_skill(), DEFAULT_RENAMES)

    assert [t["function"]["name"] for t in out["tools"]] == ["load_skill", "bash"]
    assert out["messages"][1]["tool_calls"][0]["function"]["name"] == "load_skill"
    assert out["messages"][2]["name"] == "load_skill"


def test_repair_request_sanitises_arguments_in_history() -> None:
    """One malformed historical call is enough to 400 every later request."""
    out = repair_request(_payload_with_skill(), DEFAULT_RENAMES)
    assert out["messages"][1]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_repair_request_does_not_mutate_the_caller_payload() -> None:
    """Retries re-send the original payload, so repair must not edit it in place."""
    payload = _payload_with_skill()
    repair_request(payload, DEFAULT_RENAMES)
    assert payload["tools"][0]["function"]["name"] == "skill"


def test_repair_request_leaves_unmapped_tools_alone() -> None:
    """An empty rename map disables renaming without disabling sanitising."""
    out = repair_request(_payload_with_skill(), {})
    assert [t["function"]["name"] for t in out["tools"]] == ["skill", "bash"]


# ── restore_response ───────────────────────────────────────


def test_restore_response_renames_back_in_a_non_streaming_message() -> None:
    """Vibe executes its own builtin, so it must see the original name."""
    body: JsonDict = {
        "choices": [{"message": {"tool_calls": [{"function": {"name": "load_skill"}}]}}]
    }
    out = restore_response(body, {"load_skill": "skill"})
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "skill"


def test_restore_response_renames_back_in_a_streaming_delta() -> None:
    """Streaming carries tool calls in ``delta`` rather than ``message``."""
    event: JsonDict = {
        "choices": [{"delta": {"tool_calls": [{"function": {"name": "load_skill"}}]}}]
    }
    out = restore_response(event, {"load_skill": "skill"})
    assert out["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "skill"


# ── parse_sse_line ─────────────────────────────────────────


def test_parse_sse_line_decodes_events_and_ignores_framing() -> None:
    """Only ``data:`` payloads are events; framing and junk decode to nothing."""
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line("") is None
    assert parse_sse_line(": keepalive") is None
    assert parse_sse_line("data: not json") is None


# ── StreamOutcome ──────────────────────────────────────────


def _delta(**kw: object) -> JsonDict:
    """Build a one-choice streaming chunk carrying the given delta fields."""
    return {"choices": [{"index": 0, "delta": dict(kw), "finish_reason": None}]}


def test_stream_outcome_flags_a_truncated_stream_as_unhealthy() -> None:
    """The real silent failure: reasoning only, no finish_reason, no usage."""
    outcome = StreamOutcome()
    for _ in range(3):
        outcome.feed(_delta(content="", reasoning="thinking"))
    assert not outcome.healthy


def test_stream_outcome_accepts_a_stream_that_finished() -> None:
    """A healthy turn always carries a finish_reason; that is the whole signal."""
    outcome = StreamOutcome()
    outcome.feed(_delta(content="", reasoning="thinking"))
    outcome.feed(_delta(tool_calls=[{"function": {"name": "bash"}}]))
    outcome.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    assert outcome.healthy


def test_stream_outcome_streams_reasoning_but_withholds_the_answer() -> None:
    """Buffered relay: reasoning is display-only so it flows live; the answer waits."""
    outcome = StreamOutcome("buffered")
    assert outcome.feed(_delta(content="", reasoning="thinking")) is not None
    assert outcome.feed(_delta(content="hello")) is None
    assert outcome.feed(_delta(tool_calls=[{"function": {"name": "bash"}}])) is None

    assert len(outcome.release()) == 2


def test_stream_outcome_orders_usage_frames_after_the_answer() -> None:
    """Usage arrives on a choice-less frame and must not jump ahead of content."""
    outcome = StreamOutcome()
    outcome.feed(_delta(content="hello"))
    outcome.feed({"usage": {"prompt_tokens": 1}, "choices": []})
    assert outcome.release()[-1].get("usage") == {"prompt_tokens": 1}


# ── failure_event ──────────────────────────────────────────


def test_failure_event_is_terminal_and_attributed_to_the_shim() -> None:
    """A visible error beats the silent empty turn it replaces."""
    event = failure_event("muse-medmcp", "stream truncated")
    choice = event["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "medmcp" in choice["delta"]["content"]
    assert "stream truncated" in choice["delta"]["content"]


# ── relay modes ────────────────────────────────────────────


def test_stream_outcome_live_streams_answer_text_but_withholds_tool_calls() -> None:
    """Live relay: content goes out at once (and is remembered), tool calls wait."""
    outcome = StreamOutcome("live")
    text = {"choices": [{"index": 0, "delta": {"content": "Hel"}}]}
    call = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1"}]}}]}
    assert outcome.feed(text) is text
    assert outcome.relayed_visible is True
    assert outcome.feed(call) is None
    assert outcome.buffered == [call]


def test_stream_outcome_buffered_withholds_answer_text() -> None:
    """Buffered relay is the original behaviour: nothing visible until healthy."""
    outcome = StreamOutcome("buffered")
    text = {"choices": [{"index": 0, "delta": {"content": "Hel"}}]}
    assert outcome.feed(text) is None
    assert outcome.relayed_visible is False
    assert outcome.buffered == [text]


def test_stream_outcome_rejects_an_unknown_relay_mode() -> None:
    """A typo in MEDMCP_SHIM_RELAY must not silently pick a mode."""
    with pytest.raises(ValueError):
        StreamOutcome("eager")


def _sse(events: list[JsonDict], *, done: bool) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


def _text(chunk: str, finish: str | None = None) -> JsonDict:
    return {"choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": finish}]}


def _call(finish: str | None = None) -> JsonDict:
    delta = {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "load_skill"}}]}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _client(responses: list[bytes]) -> httpx.AsyncClient:
    """An upstream that answers each attempt with the next canned SSE body."""
    calls = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=next(calls), headers={"content-type": "text/event-stream"}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _collect(relay: str, responses: list[bytes], attempts: int = 3) -> str:
    async with _client(responses) as client:
        chunks = [
            chunk
            async for chunk in llmshim._stream_with_retry(
                "http://upstream/v1/chat/completions",
                {"model": "m", "stream": True},
                {"load_skill": "skill"},
                attempts,
                "m",
                relay=relay,
                client=client,
            )
        ]
    return b"".join(chunks).decode()


@pytest.mark.asyncio
async def test_tool_call_truncation_is_retried_invisibly_in_live_relay() -> None:
    """The common failure strikes at the tool call, which live relay still withholds."""
    truncated = _sse([_call()], done=False)  # no finish_reason: Ollama's parser failure
    final: JsonDict = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    healthy = _sse([_call(), final], done=True)
    out = await _collect("live", [truncated, healthy])
    assert out.count('"name": "skill"') == 1  # one call, restored name, no duplicate
    assert '"finish_reason": "tool_calls"' in out
    assert out.endswith("data: [DONE]\n\n")
    assert "medmcp-shim-error" not in out


@pytest.mark.asyncio
async def test_truncation_after_visible_text_ends_the_stream_as_incomplete() -> None:
    """Text already shown cannot be retried: no [DONE], no finish_reason, no shim message."""
    truncated = _sse([_text("Hello, "), _text("wor")], done=False)
    healthy = _sse([_text("Hello, world", "stop")], done=True)
    out = await _collect("live", [truncated, healthy])
    assert "Hello, " in out and "wor" in out
    assert 'finish_reason": "stop"' not in out
    assert "[DONE]" not in out
    assert "medmcp-shim-error" not in out
    assert out.count("Hello") == 1  # the healthy second attempt was never requested


@pytest.mark.asyncio
async def test_buffered_relay_retries_text_truncation_invisibly() -> None:
    """The original mode: the person only ever sees the healthy attempt."""
    truncated = _sse([_text("Hello, "), _text("wor")], done=False)
    healthy = _sse([_text("Hello, world", "stop")], done=True)
    out = await _collect("buffered", [truncated, healthy])
    assert out.count("Hello") == 1
    assert "Hello, world" in out
    assert out.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_live_relay_reports_failure_after_every_attempt_truncates_invisibly() -> None:
    """Tool-call turns that never complete still end in the labelled shim message."""
    truncated = _sse([_call()], done=False)
    out = await _collect("live", [truncated, truncated], attempts=2)
    assert "medmcp-shim-error" in out
    assert out.endswith("data: [DONE]\n\n")
