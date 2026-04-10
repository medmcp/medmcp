"""Tests for the permission-prompt formatting and explanation generation in app.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from medmcp.app import (
    OLLAMA_BASE_URL,  # pyright: ignore[reportPrivateUsage]
    OLLAMA_MODEL,  # pyright: ignore[reportPrivateUsage]
    _format_permission_prompt,  # pyright: ignore[reportPrivateUsage]
    _generate_human_readable,  # pyright: ignore[reportPrivateUsage]
)

# ── _format_permission_prompt ─────────────────────────────


class TestFormatPermissionPrompt:
    """Verify that _format_permission_prompt renders the expected markdown."""

    def test_title_only(self) -> None:
        """Minimal tool call with just a title."""
        result = _format_permission_prompt({"title": "bash: echo hi"})
        assert "`bash: echo hi`" in result
        assert "Approve tool call?" in result

    def test_with_raw_input(self) -> None:
        """Raw input should appear inside a json code fence."""
        result = _format_permission_prompt(
            {
                "title": "bash: ls",
                "rawInput": {"command": "ls -la"},
            }
        )
        assert "```json" in result
        assert '"command"' in result

    def test_with_human_readable(self) -> None:
        """Human-readable description should render as a blockquote."""
        result = _format_permission_prompt(
            {
                "title": "bash: find . -name '*.log' -delete",
                "humanReadable": "Delete all .log files in the current directory tree.",
            }
        )
        assert "> Delete all .log files" in result

    def test_human_readable_before_raw_input(self) -> None:
        """The human-readable line must appear before the raw input block."""
        result = _format_permission_prompt(
            {
                "title": "bash: rm -rf /tmp/old",
                "rawInput": {"command": "rm -rf /tmp/old"},
                "humanReadable": "Remove the /tmp/old directory.",
            }
        )
        hr_pos = result.index("> Remove the /tmp/old directory.")
        raw_pos = result.index("```json")
        assert hr_pos < raw_pos

    def test_missing_human_readable(self) -> None:
        """When humanReadable is absent the output should not contain a blockquote."""
        result = _format_permission_prompt(
            {
                "title": "bash: ls",
                "rawInput": "ls",
            }
        )
        assert "> " not in result

    def test_empty_human_readable_ignored(self) -> None:
        """An empty humanReadable string should be treated as absent."""
        result = _format_permission_prompt(
            {
                "title": "bash: ls",
                "humanReadable": "",
            }
        )
        assert result.count(">") == 0

    def test_fallback_title(self) -> None:
        """When title is missing, fall back to 'tool call'."""
        result = _format_permission_prompt({})
        assert "`tool call`" in result


# ── _generate_human_readable ──────────────────────────────


def _mock_ollama_response(text: str) -> httpx.Response:
    """Build a fake httpx.Response mimicking Ollama's chat/completions endpoint."""
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
        },
        request=httpx.Request("POST", f"{OLLAMA_BASE_URL}/v1/chat/completions"),
    )


class TestGenerateHumanReadable:
    """Verify _generate_human_readable calls Ollama and parses the response."""

    @pytest.mark.asyncio
    async def test_returns_explanation(self) -> None:
        """A successful Ollama response should yield the explanation text."""
        mock_response = _mock_ollama_response("Lists all files in the current directory.")
        with patch("medmcp.app.httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = instance

            result = await _generate_human_readable({"title": "bash: ls -la"})

        assert result == "Lists all files in the current directory."
        instance.post.assert_called_once()
        call_kwargs: dict[str, Any] = instance.post.call_args[1]
        assert call_kwargs["json"]["model"] == OLLAMA_MODEL

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_choices(self) -> None:
        """An empty choices array should return None."""
        response = httpx.Response(
            200,
            json={"choices": []},
            request=httpx.Request("POST", f"{OLLAMA_BASE_URL}/v1/chat/completions"),
        )
        with patch("medmcp.app.httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            instance.post.return_value = response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = instance

            result = await _generate_human_readable({"title": "bash: ls"})

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self) -> None:
        """Network failures should return None, not raise."""
        with patch("medmcp.app.httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            instance.post.side_effect = httpx.ConnectError("connection refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = instance

            result = await _generate_human_readable({"title": "bash: ls"})

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_blank_content(self) -> None:
        """A blank content string should be treated as no explanation."""
        response = _mock_ollama_response("   ")
        with patch("medmcp.app.httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            instance.post.return_value = response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = instance

            result = await _generate_human_readable({"title": "bash: ls"})

        assert result is None

    @pytest.mark.asyncio
    async def test_dict_raw_input_serialized(self) -> None:
        """Dict rawInput should be JSON-serialized in the prompt sent to Ollama."""
        mock_response = _mock_ollama_response("Writes hello to a file.")
        with patch("medmcp.app.httpx.AsyncClient") as mock_client_cls:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = instance

            await _generate_human_readable(
                {"title": "write_file", "rawInput": {"path": "/tmp/x", "content": "hello"}}
            )

        call_kwargs: dict[str, Any] = instance.post.call_args[1]
        prompt: str = call_kwargs["json"]["messages"][0]["content"]
        # The dict should have been serialized, not repr'd
        assert '"path"' in prompt
