"""Tests for LLM-generated chat titles (transcript window, cadence, cleanup, request)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

# pyright: reportPrivateUsage=false
from medmcp import provenance, titles
from medmcp.titles import TitleCadence, TitlePolicy, build_transcript, clean_title

JsonDict = dict[str, Any]

SESSION_ID = "abcd1234-1111-2222-3333-444455556666"


def _assistant_call(name: str) -> JsonDict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": "{}"}}],
    }


class TestBuildTranscript:
    """The window the model sees: roles rendered, noise skipped, size bounded."""

    def test_user_and_assistant_text_render_with_roles(self) -> None:
        """System prompts are dropped; user and assistant text render with their roles."""
        msgs = [
            {"role": "system", "content": "You are MedMCP."},
            {"role": "user", "content": "Skull strip the T1"},
            {"role": "assistant", "content": "Done: brain mask written."},
        ]
        assert build_transcript(msgs) == (
            "user: Skull strip the T1\n\nassistant: Done: brain mask written."
        )

    def test_tool_calls_name_the_step_and_drop_the_server_prefix(self) -> None:
        """A tool call contributes its step name (server prefix removed) as topic signal."""
        msgs = [_assistant_call("medmcp-neuro_skull_strip")]
        assert build_transcript(msgs) == "assistant: (ran skull_strip)"

    def test_injected_and_compaction_messages_are_skipped(self) -> None:
        """A /retry continuation or the compaction summary is not something the person said."""
        msgs = [
            {"role": "user", "content": "harness text", "injected": True},
            {"role": "user", "content": "summary", "context_boundary": "compaction"},
            {"role": "user", "content": "Register to MNI"},
        ]
        assert build_transcript(msgs) == "user: Register to MNI"

    def test_workspace_note_is_stripped_and_display_content_preferred(self) -> None:
        """The viewer note is live-turn metadata: dropped, with the note-free text preferred."""
        note = "Segment this\n\n[workspace context: the user is viewing data/x/t1.nii.gz]"
        assert build_transcript([{"role": "user", "content": note}]) == "user: Segment this"
        with_display = {
            "role": "user",
            "content": note,
            "user_display_content": {
                "version": "1",
                "host": "medmcp",
                "content": [{"type": "text", "text": "Segment this scan"}],
            },
        }
        assert build_transcript([with_display]) == "user: Segment this scan"

    def test_tool_results_are_clamped(self) -> None:
        """A tool result is clamped so one large output can't crowd the window."""
        policy = TitlePolicy(max_tool_result_chars=10)
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": "x" * 50}]
        assert build_transcript(msgs, policy=policy) == "tool: " + "x" * 10

    def test_long_transcript_keeps_head_and_tail(self) -> None:
        """Over the size cap, the opening and the latest exchange survive with the middle elided."""
        policy = TitlePolicy(max_transcript_chars=40, head_transcript_chars=15)
        msgs = [{"role": "user", "content": "A" * 30}, {"role": "assistant", "content": "B" * 30}]
        out = build_transcript(msgs, policy=policy)
        assert out.startswith("user: AAAAAAAAA")
        assert out.endswith("B" * 25)
        assert "[…]" in out

    def test_empty_when_nothing_to_say(self) -> None:
        """No titleable content yields an empty transcript (and so no request)."""
        assert build_transcript([]) == ""
        assert build_transcript([{"role": "system", "content": "x"}]) == ""


class TestCleanTitle:
    """Model answers are shaped into one clean line, or rejected."""

    def test_first_line_quotes_and_whitespace(self) -> None:
        """Only the first line counts; wrapping quotes and runs of whitespace go."""
        assert clean_title('  "Skull-strip   subject 12"\nmore text') == "Skull-strip subject 12"

    def test_control_characters_are_dropped(self) -> None:
        """Control characters never reach a header or a terminal title."""
        assert clean_title("Regis\x07ter T1\x00\x7f") == "Register T1"

    @pytest.mark.parametrize(
        "raw", [None, "", "   ", "New chat", "new chat.", "Untitled", "`New session`"]
    )
    def test_generic_or_empty_answers_are_rejected(self, raw: str | None) -> None:
        """Empty and placeholder answers mean 'nothing usable', not a title."""
        assert clean_title(raw) is None

    def test_over_long_titles_are_capped(self) -> None:
        """A runaway answer is cut at the length cap with an ellipsis."""
        assert clean_title("word " * 40, policy=TitlePolicy(max_title_chars=12)) == "word word wo…"


class TestCadence:
    """When a refresh is due: first turn, then bounded periodic refreshes."""

    def test_first_title_after_the_opening_turn_then_every_n_turns(self) -> None:
        """The first title is due after the opening turn, then every N turns."""
        cadence = TitleCadence(TitlePolicy(refresh_every_turns=3, max_generations=10))
        due = [cadence.begin_if_due() is not None for _ in range(8)]
        assert due == [True, False, False, True, False, False, True, False]

    def test_generations_are_capped(self) -> None:
        """Generations stop at the cap however many turns follow."""
        cadence = TitleCadence(TitlePolicy(refresh_every_turns=1, max_generations=2))
        assert [cadence.begin_if_due() is not None for _ in range(4)] == [True, True, False, False]

    def test_restore_reschedules_but_keeps_the_attempt_counted(self) -> None:
        """A refresh that yields nothing retries next turn, but the attempt still counts."""
        cadence = TitleCadence(TitlePolicy(refresh_every_turns=3, max_generations=2))
        ticket = cadence.begin_if_due()
        assert ticket is not None
        cadence.restore(ticket)
        # The initial title is pending again, so the very next turn retries…
        assert cadence.begin_if_due() is not None
        # …but two attempts have now been spent, so no third.
        for _ in range(6):
            assert cadence.begin_if_due() is None


class TestGenerateTitle:
    """The request itself, against a fake model endpoint."""

    @staticmethod
    def _client(reply: str, seen: list[JsonDict]) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"role": "assistant", "content": reply}})

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @pytest.mark.asyncio
    async def test_sends_transcript_and_returns_cleaned_title(self) -> None:
        """The request carries the transcript and the previous title; the answer is cleaned."""
        seen: list[JsonDict] = []
        msgs = [{"role": "user", "content": "Skull strip subject 12's T1"}]
        title = await titles.generate_title(
            msgs, previous_title="Old name", client=self._client('"Skull-strip subject 12"\n', seen)
        )
        assert title == "Skull-strip subject 12"
        assert len(seen) == 1
        request = seen[0]
        assert request["stream"] is False and request["think"] is False
        assert request["options"]["temperature"] == 0.0
        user = request["messages"][-1]["content"]
        assert user.startswith("Current title: Old name")
        assert "user: Skull strip subject 12's T1" in user

    @pytest.mark.asyncio
    async def test_nothing_to_title_makes_no_request(self) -> None:
        """An empty transcript never reaches the model."""
        seen: list[JsonDict] = []
        assert await titles.generate_title([], client=self._client("x", seen)) is None
        assert seen == []

    @pytest.mark.asyncio
    async def test_failures_are_swallowed(self) -> None:
        """A backend failure yields None — a title is a nicety, never an error."""

        def boom(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="nope")

        client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
        msgs = [{"role": "user", "content": "hello"}]
        assert await titles.generate_title(msgs, client=client) is None


def test_enabled_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated titles are on by default and MEDMCP_AUTO_TITLES=0 turns them off."""
    monkeypatch.delenv(titles.DISABLE_ENV_VAR, raising=False)
    assert titles.enabled() is True
    monkeypatch.setenv(titles.DISABLE_ENV_VAR, "0")
    assert titles.enabled() is False
    monkeypatch.setenv(titles.DISABLE_ENV_VAR, "1")
    assert titles.enabled() is True


def test_load_session_messages_reads_the_chain(tmp_path: Path) -> None:
    """Messages come from vibe's session dir (absent → empty, never an error)."""
    logs = tmp_path / "logs" / "session"
    sess = logs / f"session_20260901_{SESSION_ID[:8]}"
    sess.mkdir(parents=True)
    (sess / "meta.json").write_text(json.dumps({"session_id": SESSION_ID}))
    (sess / "messages.jsonl").write_text(json.dumps({"role": "user", "content": "hi"}) + "\n")
    with patch.object(provenance, "VIBE_HOME", tmp_path):
        assert titles.load_session_messages(SESSION_ID) == [{"role": "user", "content": "hi"}]
        assert titles.load_session_messages("missing-id") == []
