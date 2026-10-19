"""Tests for the Ollama Glimmer tool-call shim (``medmcp.llmshim``).

The shapes exercised here are the ones measured against the live model: Ollama
rejects any ``arguments`` that is not a JSON-object string, and a silently
truncated stream is identifiable by its missing ``finish_reason``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

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


async def test_buffered_relay_retries_text_truncation_invisibly() -> None:
    """The original mode: the person only ever sees the healthy attempt."""
    truncated = _sse([_text("Hello, "), _text("wor")], done=False)
    healthy = _sse([_text("Hello, world", "stop")], done=True)
    out = await _collect("buffered", [truncated, healthy])
    assert out.count("Hello") == 1
    assert "Hello, world" in out
    assert out.endswith("data: [DONE]\n\n")


async def test_live_relay_reports_failure_after_every_attempt_truncates_invisibly() -> None:
    """Tool-call turns that never complete still end in the labelled shim message."""
    truncated = _sse([_call()], done=False)
    out = await _collect("live", [truncated, truncated], attempts=2)
    assert "medmcp-shim-error" in out
    assert out.endswith("data: [DONE]\n\n")


def _rejecting_client(status: int) -> tuple[httpx.AsyncClient, list[int]]:
    """An upstream that answers every request with *status*; returns the request log."""
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(status)
        return httpx.Response(status, json={"error": {"message": "bad request"}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


async def test_client_error_is_passed_through_once() -> None:
    """A 400 is the upstream rejecting the request: no retries, the body passed through."""
    client, seen = _rejecting_client(400)
    async with client:
        resp = await llmshim._complete_with_retry(
            "http://upstream/v1/chat/completions", {"model": "m"}, {}, 3, client=client
        )
    assert resp.status_code == 400
    assert len(seen) == 1


async def test_client_error_ends_a_stream_after_one_attempt() -> None:
    """The streamed path cannot change its status code; it reports the 4xx in-band, once."""
    client, seen = _rejecting_client(400)
    async with client:
        chunks = [
            chunk
            async for chunk in llmshim._stream_with_retry(
                "http://upstream/v1/chat/completions",
                {"model": "m", "stream": True},
                {},
                3,
                "m",
                relay="live",
                client=client,
            )
        ]
    out = b"".join(chunks).decode()
    assert len(seen) == 1
    assert "medmcp-shim-error" in out
    assert "HTTP 400" in out


# ── Glimmer mode: the shim renders and parses tool calls itself ──────────────


_BASH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 300},
                "paths": {"type": "array", "items": {"type": "string"}},
                "verbose": {"type": "boolean"},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["command"],
        },
    },
}
_FUNCTIONS: dict[str, dict[str, Any]] = {"bash": _BASH["function"]}
_BLOCK = (
    "<atem:function_calls>\n"
    '<atem:invoke name="bash">\n'
    '<atem:parameter name="command">ls -la</atem:parameter>\n'
    '<atem:parameter name="timeout_seconds">30</atem:parameter>\n'
    '<atem:parameter name="paths">["a", "b"]</atem:parameter>\n'
    '<atem:parameter name="verbose">true</atem:parameter>\n'
    '<atem:parameter name="note">null</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)


class TestGlimmerRender:
    """The tool section matches what Ollama's renderer puts into the prompt."""

    def test_section_has_metadata_schemas_and_example(self) -> None:
        """Namespace metadata, every schema as JSON, and the worked example, in that order."""
        section = llmshim.render_tool_section([_BASH["function"]])
        assert section.startswith("In this environment you have access to a set of tools")
        meta = section.index('{"name": "bash", "description": ""}')
        schemas = section.index("// Function schemas")
        schema = section.index(
            '{"name": "bash", "description": "Run a shell command.", "parameters": '
        )
        example = section.index("to=example_tool_name.example_function_name")
        assert meta < schemas < schema < example
        assert section.endswith("</atem:function_calls>")

    def test_namespace_is_the_part_before_the_dot(self) -> None:
        """``group.tool`` names share one metadata line, as in Ollama."""
        fns: list[dict[str, Any]] = [
            {"name": "mcp_neuro.a", "parameters": {}},
            {"name": "mcp_neuro.b", "parameters": {}},
        ]
        section = llmshim.render_tool_section(fns)
        assert section.count('{"name": "mcp_neuro", "description": ""}') == 1

    def test_request_moves_tools_into_the_system_message(self) -> None:
        """``tools``/``tool_choice`` leave the request; the section lands after the prompt."""
        req = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are X."},
                {"role": "user", "content": "hi"},
            ],
            "tools": [_BASH],
            "tool_choice": "auto",
        }
        out, functions = llmshim.glimmer_request(req)
        assert "tools" not in out and "tool_choice" not in out
        assert out["messages"][0]["content"].startswith("You are X.\n\nIn this environment")
        assert list(functions) == ["bash"]
        assert "tools" in req  # the caller's payload is untouched

    def test_request_without_tools_is_unchanged(self) -> None:
        """No tools, nothing to render: the same object comes back."""
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        out, functions = llmshim.glimmer_request(req)
        assert out is req and functions == {}

    def test_request_without_system_message_gets_one(self) -> None:
        """The section needs a system message to live in."""
        req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [_BASH]}
        out, _ = llmshim.glimmer_request(req)
        assert out["messages"][0]["role"] == "system"
        assert out["messages"][1]["role"] == "user"


class TestAtemParse:
    """ATEM blocks become OpenAI tool calls with schema-typed arguments."""

    def test_values_are_typed_from_the_schema(self) -> None:
        """Integer, array, boolean and null parameters convert; strings stay verbatim."""
        calls = llmshim.parse_atem_block(_BLOCK, _FUNCTIONS)
        assert len(calls) == 1
        call = calls[0]
        assert call["type"] == "function" and call["id"].startswith("call_")
        assert call["function"]["name"] == "bash"
        assert json.loads(call["function"]["arguments"]) == {
            "command": "ls -la",
            "timeout_seconds": 30,
            "paths": ["a", "b"],
            "verbose": True,
            "note": None,
        }

    def test_whitespace_between_parameters_is_tolerated(self) -> None:
        """Blank lines and indentation, which trip Ollama's parser, are fine here."""
        block = (
            '<atem:function_calls>\n\n  <atem:invoke name="bash">\n\n'
            '    <atem:parameter name="command">pwd</atem:parameter>\n\n'
            "  </atem:invoke>\n</atem:function_calls>"
        )
        calls = llmshim.parse_atem_block(block, _FUNCTIONS)
        assert json.loads(calls[0]["function"]["arguments"]) == {"command": "pwd"}

    def test_string_values_keep_their_spaces_and_lines(self) -> None:
        """Strings are not stripped, as the prompt promises the model."""
        block = (
            '<atem:function_calls><atem:invoke name="bash">'
            '<atem:parameter name="command">  echo "a"\n  echo "b"  </atem:parameter>'
            "</atem:invoke></atem:function_calls>"
        )
        calls = llmshim.parse_atem_block(block, _FUNCTIONS)
        assert (
            json.loads(calls[0]["function"]["arguments"])["command"] == '  echo "a"\n  echo "b"  '
        )

    def test_unknown_tool_and_untyped_value_pass_through(self) -> None:
        """A name the schema lacks is still a call; its values stay strings unless JSON-like."""
        block = (
            '<atem:function_calls><atem:invoke name="mystery">'
            '<atem:parameter name="n">7</atem:parameter>'
            '<atem:parameter name="o">{"k": 1}</atem:parameter>'
            "</atem:invoke></atem:function_calls>"
        )
        calls = llmshim.parse_atem_block(block, _FUNCTIONS)
        assert json.loads(calls[0]["function"]["arguments"]) == {"n": "7", "o": {"k": 1}}

    def test_two_invokes_are_two_calls(self) -> None:
        """One block may carry several calls; indices count up."""
        block = (
            '<atem:function_calls><atem:invoke name="bash"><atem:parameter name="command">a'
            "</atem:parameter></atem:invoke>"
            '<atem:invoke name="bash"><atem:parameter name="command">b'
            "</atem:parameter></atem:invoke></atem:function_calls>"
        )
        calls = llmshim.parse_atem_block(block, _FUNCTIONS)
        assert [c["index"] for c in calls] == [0, 1]


class TestAtemSplitter:
    """Text around a block streams; the block is held until it closes."""

    def test_marker_split_across_deltas(self) -> None:
        """A partial opening marker is held back, never shown as text."""
        sp = llmshim.AtemSplitter(_FUNCTIONS)
        out = [
            sp.feed(x)
            for x in (
                "Sure, ",
                "let me <at",
                "em:function_calls>",
                _BLOCK[len("<atem:function_calls>") :],
            )
        ]
        assert out[:2] == ["Sure, ", "let me "]
        assert "".join(out[2:]) == ""
        assert not sp.open and len(sp.calls) == 1
        assert sp.flush() == ""

    def test_lone_angle_bracket_is_released_when_it_is_not_the_marker(self) -> None:
        """Held text that turns out not to be the marker comes out with the next delta."""
        sp = llmshim.AtemSplitter(_FUNCTIONS)
        assert sp.feed("a <") == "a "
        assert sp.feed(" b") == "< b"

    def test_unterminated_block_is_never_released(self) -> None:
        """A cut stream leaves the block open: nothing leaks, no call is produced."""
        sp = llmshim.AtemSplitter(_FUNCTIONS)
        assert sp.feed('<atem:function_calls>\n<atem:invoke name="bash">') == ""
        assert sp.open and sp.calls == [] and sp.flush() == ""


class TestGlimmerStream:
    """The full streaming path: content in, OpenAI tool-call chunks out."""

    @staticmethod
    def _events(*contents: str, finish: str = "stop") -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for text in contents:
            out.append(
                {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
            )
        out.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        out.append({"choices": [], "usage": {"total_tokens": 1}})
        return out

    def test_block_becomes_tool_calls_and_finish_reason_is_rewritten(self) -> None:
        """Ollama says ``stop`` (it saw text); the shim says ``tool_calls`` and carries them."""
        outcome = llmshim.StreamOutcome("live", functions=_FUNCTIONS)
        forwarded = [e for e in (outcome.feed(ev) for ev in self._events("Running: ", _BLOCK)) if e]
        released = outcome.release()
        assert outcome.healthy
        assert forwarded[0]["choices"][0]["delta"]["content"] == "Running: "
        final = next(e for e in released if e["choices"] and e["choices"][0].get("finish_reason"))
        assert final["choices"][0]["finish_reason"] == "tool_calls"
        calls = final["choices"][0]["delta"]["tool_calls"]
        assert calls[0]["function"]["name"] == "bash"
        assert json.loads(calls[0]["function"]["arguments"])["command"] == "ls -la"

    def test_plain_answer_is_untouched(self) -> None:
        """No block: text streams through and ``stop`` stays ``stop``."""
        outcome = llmshim.StreamOutcome("live", functions=_FUNCTIONS)
        forwarded = [e for e in (outcome.feed(ev) for ev in self._events("Hello", " there")) if e]
        assert [e["choices"][0]["delta"]["content"] for e in forwarded] == ["Hello", " there"]
        final = next(
            e for e in outcome.release() if e["choices"] and e["choices"][0].get("finish_reason")
        )
        assert final["choices"][0]["finish_reason"] == "stop"
        assert "tool_calls" not in final["choices"][0]["delta"]

    def test_stream_cut_inside_a_block_is_unhealthy(self) -> None:
        """An open block at the end means truncation, whatever Ollama's finish_reason."""
        outcome = llmshim.StreamOutcome("live", functions=_FUNCTIONS)
        for ev in self._events('<atem:function_calls>\n<atem:invoke name="bash">'):
            outcome.feed(ev)
        assert not outcome.healthy

    def test_non_streaming_completion_is_parsed_too(self) -> None:
        """The same block in a ``message.content`` becomes ``tool_calls``."""
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Let me look.\n" + _BLOCK},
                    "finish_reason": "stop",
                }
            ]
        }
        out = llmshim.parse_completion(body, _FUNCTIONS)
        choice = out["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] == "Let me look.\n"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "bash"


class TestToolMode:
    """``MEDMCP_SHIM_TOOLS`` picks who parses tool calls."""

    def test_glimmer_mode_forwards_no_tools_and_restores_calls(self) -> None:
        """Through the app: tools leave the request, the ATEM block comes back as tool_calls."""
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            sse = (
                "".join(
                    f"data: {json.dumps(ev)}\n\n" for ev in TestGlimmerStream._events("", _BLOCK)
                )
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"}
            )

        async def run() -> list[dict[str, object]]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                payload, functions = llmshim.glimmer_request(
                    {
                        "model": "m",
                        "stream": True,
                        "messages": [{"role": "user", "content": "ls"}],
                        "tools": [_BASH],
                    }
                )
                chunks: list[dict[str, object]] = []
                async for raw in llmshim._stream_with_retry(
                    "http://up/v1/chat/completions",
                    payload,
                    {},
                    1,
                    "m",
                    client=client,
                    functions=functions,
                ):
                    ev = llmshim.parse_sse_line(raw.decode().strip())
                    if ev:
                        chunks.append(ev)
                return chunks

        chunks = asyncio.run(run())
        assert "tools" not in seen
        assert "// Function schemas" in str(seen["messages"])
        final = next(c for c in chunks if c["choices"] and c["choices"][0].get("finish_reason"))  # type: ignore[index]
        assert final["choices"][0]["finish_reason"] == "tool_calls"  # type: ignore[index]

    def test_unknown_tool_mode_is_rejected(self) -> None:
        """A typo in the env var must not silently pick a mode."""
        with pytest.raises(ValueError):
            llmshim.create_app("http://up", tool_mode="magic")
