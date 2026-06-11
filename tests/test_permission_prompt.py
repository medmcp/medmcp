"""Tests for the permission-prompt formatting and explanation generation in app.py."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from medmcp.app import _format_permission_prompt  # pyright: ignore[reportPrivateUsage]
from medmcp.explain import (
    RISK_CATEGORIES,
    generate_explanation,
    parse_explanation_response,
)
from medmcp.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

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

    def test_risks_rendered_as_badges(self) -> None:
        """Known risk keys should render as labelled badge lines."""
        result = _format_permission_prompt(
            {
                "title": "bash: rm -rf /data",
                "humanReadable": "Permanently deletes the /data folder.",
                "risks": ["file_delete", "code_exec"],
            }
        )
        label_delete, _ = RISK_CATEGORIES["file_delete"]
        label_exec, _ = RISK_CATEGORIES["code_exec"]
        assert label_delete in result
        assert label_exec in result

    def test_risks_appear_after_human_readable(self) -> None:
        """Risk badges must follow the blockquote explanation, before the JSON fence."""
        result = _format_permission_prompt(
            {
                "title": "bash: curl http://example.com",
                "rawInput": {"url": "http://example.com"},
                "humanReadable": "Downloads a file from the internet.",
                "risks": ["network"],
            }
        )
        hr_pos = result.index("> Downloads")
        label_network, _ = RISK_CATEGORIES["network"]
        risk_pos = result.index(label_network)
        raw_pos = result.index("```json")
        assert hr_pos < risk_pos < raw_pos

    def test_unknown_risk_keys_ignored(self) -> None:
        """Risk keys not in RISK_CATEGORIES should be silently dropped."""
        result = _format_permission_prompt(
            {
                "title": "bash: ls",
                "risks": ["not_a_real_risk", "file_read"],
            }
        )
        assert "not_a_real_risk" not in result
        label_read, _ = RISK_CATEGORIES["file_read"]
        assert label_read in result

    def test_no_risks_no_badge_line(self) -> None:
        """An empty risks list should not add any badge output."""
        result_with = _format_permission_prompt({"title": "bash: ls", "risks": ["file_read"]})
        result_without = _format_permission_prompt({"title": "bash: ls", "risks": []})
        assert len(result_with) > len(result_without)


# ── parse_explanation_response ───────────────────────────


class TestParseExplanationResponse:
    """Unit tests for the JSON parser — no Ollama calls needed."""

    def test_valid_json(self) -> None:
        """Clean JSON should parse to (explanation, risks)."""
        payload = json.dumps({"explanation": "Lists folder contents.", "risks": ["file_read"]})
        result = parse_explanation_response(payload)
        assert result == ("Lists folder contents.", ["file_read"])

    def test_json_in_code_fence(self) -> None:
        """JSON wrapped in a markdown code fence should still parse correctly."""
        payload = '```json\n{"explanation": "Reads a file.", "risks": ["file_read"]}\n```'
        result = parse_explanation_response(payload)
        assert result is not None
        explanation, risks = result
        assert explanation == "Reads a file."
        assert risks == ["file_read"]

    def test_preamble_text_stripped(self) -> None:
        """If the model adds leading text before the JSON object, it should be stripped."""
        payload = 'Here is the result:\n{"explanation": "Does something.", "risks": []}'
        result = parse_explanation_response(payload)
        assert result is not None
        assert result[0] == "Does something."

    def test_invalid_risk_keys_filtered(self) -> None:
        """Risk keys not in RISK_CATEGORIES should be removed."""
        payload = json.dumps(
            {"explanation": "Does something.", "risks": ["file_read", "invented_risk"]}
        )
        result = parse_explanation_response(payload)
        assert result is not None
        _, risks = result
        assert risks == ["file_read"]
        assert "invented_risk" not in risks

    def test_all_valid_risk_keys_accepted(self) -> None:
        """All keys in RISK_CATEGORIES should be accepted."""
        all_keys = list(RISK_CATEGORIES)
        payload = json.dumps({"explanation": "Does everything.", "risks": all_keys})
        result = parse_explanation_response(payload)
        assert result is not None
        _, risks = result
        assert set(risks) == set(all_keys)

    def test_missing_explanation_returns_none(self) -> None:
        """A payload without an explanation key should return None."""
        payload = json.dumps({"risks": ["file_read"]})
        assert parse_explanation_response(payload) is None

    def test_empty_explanation_returns_none(self) -> None:
        """An empty explanation string should be treated as absent."""
        payload = json.dumps({"explanation": "   ", "risks": []})
        assert parse_explanation_response(payload) is None

    def test_empty_risks_list_allowed(self) -> None:
        """An empty risks list is valid — some tool calls have no notable risks."""
        payload = json.dumps({"explanation": "Checks the date.", "risks": []})
        result = parse_explanation_response(payload)
        assert result == ("Checks the date.", [])

    def test_invalid_json_returns_none(self) -> None:
        """Garbage text that cannot be parsed as JSON should return None."""
        assert parse_explanation_response("not json at all") is None

    def test_non_dict_json_returns_none(self) -> None:
        """A JSON array at the top level should return None."""
        assert parse_explanation_response('["a", "b"]') is None


# ── generate_explanation ─────────────────────────────────


def _mock_ollama_response(body: str) -> httpx.Response:
    """Build a fake httpx.Response mimicking Ollama's native /api/chat endpoint."""
    return httpx.Response(
        200,
        json={"message": {"content": body}},
        request=httpx.Request("POST", f"{OLLAMA_BASE_URL}/api/chat"),
    )


class TestGenerateExplanation:
    """Verify generate_explanation calls Ollama and parses the structured response."""

    @pytest.mark.asyncio
    async def test_returns_explanation_and_risks(self) -> None:
        """A successful Ollama JSON response yields (explanation, risks)."""
        body = json.dumps({"explanation": "Lists all files in the folder.", "risks": ["file_read"]})
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = _mock_ollama_response(body)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            result = await generate_explanation({"title": "bash: ls -la"})

        assert result is not None
        explanation, risks = result
        assert explanation == "Lists all files in the folder."
        assert risks == ["file_read"]

    @pytest.mark.asyncio
    async def test_correct_model_and_temperature(self) -> None:
        """The request must use the configured model and temperature via options."""
        body = json.dumps({"explanation": "Does something.", "risks": []})
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = _mock_ollama_response(body)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            await generate_explanation({"title": "bash: ls"})

        call_kwargs: dict[str, Any] = instance.post.call_args[1]
        assert call_kwargs["json"]["model"] == OLLAMA_MODEL
        assert call_kwargs["json"]["options"]["temperature"] == 0.2
        assert call_kwargs["json"]["think"] is False
        assert call_kwargs["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_prompt_contains_physician_language(self) -> None:
        """The prompt must explicitly mention the physician audience."""
        body = json.dumps({"explanation": "x", "risks": []})
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = _mock_ollama_response(body)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            await generate_explanation({"title": "bash: ls"})

        call_kwargs: dict[str, Any] = instance.post.call_args[1]
        prompt: str = call_kwargs["json"]["messages"][0]["content"]
        assert "physician" in prompt.lower()

    @pytest.mark.asyncio
    async def test_prompt_lists_all_risk_keys(self) -> None:
        """Every key in RISK_CATEGORIES must appear in the prompt."""
        body = json.dumps({"explanation": "x", "risks": []})
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = _mock_ollama_response(body)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            await generate_explanation({"title": "bash: ls"})

        call_kwargs: dict[str, Any] = instance.post.call_args[1]
        prompt: str = call_kwargs["json"]["messages"][0]["content"]
        for key in RISK_CATEGORIES:
            assert key in prompt, f"Risk key '{key}' missing from prompt"

    @pytest.mark.asyncio
    async def test_dict_raw_input_serialized_in_prompt(self) -> None:
        """Dict rawInput should be JSON-serialized in the prompt, not repr'd."""
        body = json.dumps({"explanation": "Writes a file.", "risks": ["file_write"]})
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = _mock_ollama_response(body)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            await generate_explanation(
                {"title": "write_file", "rawInput": {"path": "/tmp/x", "content": "hello"}}
            )

        call_kwargs: dict[str, Any] = instance.post.call_args[1]
        prompt: str = call_kwargs["json"]["messages"][0]["content"]
        assert '"path"' in prompt

    @pytest.mark.asyncio
    async def test_long_raw_input_truncated(self) -> None:
        """RawInput longer than 400 chars should be truncated in the prompt."""
        body = json.dumps({"explanation": "Does something.", "risks": []})
        long_input = "x" * 600
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = _mock_ollama_response(body)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            await generate_explanation({"title": "bash: cat", "rawInput": long_input})

        call_kwargs: dict[str, Any] = instance.post.call_args[1]
        prompt: str = call_kwargs["json"]["messages"][0]["content"]
        assert "truncated" in prompt
        # The full 600-char string must not appear verbatim
        assert long_input not in prompt

    @pytest.mark.asyncio
    async def test_returns_none_on_missing_message(self) -> None:
        """A response with no 'message' key should return None."""
        response = httpx.Response(
            200,
            json={},
            request=httpx.Request("POST", f"{OLLAMA_BASE_URL}/api/chat"),
        )
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            result = await generate_explanation({"title": "bash: ls"})

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self) -> None:
        """Network failures should return None, not raise."""
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.side_effect = httpx.ConnectError("connection refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            result = await generate_explanation({"title": "bash: ls"})

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json_response(self) -> None:
        """A non-JSON response body should return None, not raise."""
        with patch("medmcp.explain.httpx.AsyncClient") as mock_cls:
            instance = AsyncMock()
            instance.post.return_value = _mock_ollama_response("I cannot help with that.")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = instance

            result = await generate_explanation({"title": "bash: ls"})

        assert result is None
