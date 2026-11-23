"""Chat export: a chat rendered as a standalone printable page."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medmcp import chat_export, provenance, server, sessions

from .conftest import AUTH_HEADERS

SESSION_ID = "abcd1234-1111-2222-3333-444455556666"


def _write_chat(home: Path, messages: list[dict[str, object]]) -> None:
    """Lay down a vibe session dir for SESSION_ID with *messages*."""
    d = home / "logs" / "session" / "session_20261015_120000_abcd1234"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"session_id": SESSION_ID}))
    (d / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages))


MESSAGES: list[dict[str, object]] = [
    {"role": "system", "content": "SYSTEM PROMPT TEXT"},
    {
        "role": "user",
        "content": "skull strip it\n\n[workspace context: the file …]",
        "user_display_content": {
            "version": "1",
            "host": "medmcp",
            "content": [{"type": "text", "text": "skull strip it"}],
        },
    },
    {"role": "user", "content": "compaction summary", "injected": True},
    {
        "role": "assistant",
        "reasoning_content": "SECRET REASONING",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": "medmcp-neuro-core_skull_strip",
                    "arguments": '{"input_path": "/ws/t1.nii.gz"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "skull_strip", "content": "mask written"},
    {"role": "assistant", "content": "Done. **Brain volume:** 1.2 L <script>alert(1)</script>"},
]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A vibe home with one chat and its provenance record."""
    monkeypatch.setattr(provenance, "VIBE_HOME", tmp_path)

    def _title(_sid: str) -> str:
        return "Skull-strip subject 12"

    monkeypatch.setattr(sessions, "chat_title", _title)
    _write_chat(tmp_path, MESSAGES)
    prov = tmp_path / "provenance" / SESSION_ID
    prov.mkdir(parents=True)
    (prov / "run.jsonl").write_text(
        json.dumps(
            {"tool_call_id": "call_1", "status": "completed", "permission_decision": "allow_once"}
        )
    )
    return tmp_path


def test_page_has_conversation_and_tool_calls(home: Path) -> None:
    """User text, the agent's answer, and the tool call with its decision and result."""
    del home
    page = chat_export.render_chat_html(SESSION_ID, title="Skull-strip subject 12")
    assert page is not None
    assert "<h1>Skull-strip subject 12</h1>" in page
    assert "skull strip it" in page
    assert "workspace context" not in page
    assert "medmcp-neuro-core_skull_strip" in page
    assert "/ws/t1.nii.gz" in page
    assert "Approved" in page
    assert "mask written" in page
    assert "<strong>Brain volume:</strong>" in page


def test_page_leaves_out_reasoning_system_and_injected(home: Path) -> None:
    """Hidden reasoning, the system prompt, and harness-injected turns stay out."""
    del home
    page = chat_export.render_chat_html(SESSION_ID) or ""
    assert "SECRET REASONING" not in page
    assert "SYSTEM PROMPT TEXT" not in page
    assert "compaction summary" not in page


def test_transcript_html_is_escaped(home: Path) -> None:
    """Markup in a message is shown as text, never run."""
    del home
    page = chat_export.render_chat_html(SESSION_ID) or ""
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_long_tool_output_is_clipped(home: Path) -> None:
    """A huge listing is cut, with a note of how much was left out."""
    long = [*MESSAGES[:4], {"role": "tool", "tool_call_id": "call_1", "content": "x" * 10000}]
    d = home / "logs" / "session" / "session_20261015_120000_abcd1234"
    (d / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in long))
    page = chat_export.render_chat_html(SESSION_ID) or ""
    assert "more characters]" in page
    assert "x" * 10000 not in page


def test_unknown_chat_renders_nothing(home: Path) -> None:
    """No transcript, no page."""
    del home
    assert chat_export.render_chat_html("ffffffff-0000-0000-0000-000000000000") is None


class TestEndpoint:
    """GET /api/sessions/{id}/export."""

    @pytest.fixture
    def client(self, home: Path) -> TestClient:
        """A same-origin client against the fake vibe home."""
        del home
        return TestClient(server.app, base_url="http://127.0.0.1:8100", headers=AUTH_HEADERS)

    def test_serves_the_page(self, client: TestClient) -> None:
        """The page comes back as HTML with its Save as PDF button."""
        resp = client.get(f"/api/sessions/{SESSION_ID}/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "Skull-strip subject 12" in resp.text
        assert "window.print()" in resp.text

    def test_unknown_chat_is_404(self, client: TestClient) -> None:
        """A well-formed id with no transcript is not found."""
        assert client.get("/api/sessions/ffffffff-0000/export").status_code == 404

    @pytest.mark.parametrize("bad", ["..", ".hidden", "a b"])
    def test_malformed_id_is_404(self, client: TestClient, bad: str) -> None:
        """An id that is not vibe's shape never reaches a path join."""
        assert client.get(f"/api/sessions/{bad}/export").status_code == 404
