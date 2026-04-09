"""Tests for the pure wire-format and rendering helpers in :mod:`medmcp.app`.

These functions have no I/O and no Chainlit dependencies, so they can be
exercised with plain ``assert`` statements.
"""

from __future__ import annotations

import json
from typing import Any

from medmcp.app import (
    _encode,
    _extract_text_blocks,
    _format_permission_prompt,
    _rpc_response,
    _stringify_raw,
)

# ── _encode ────────────────────────────────────────────────


def test_encode_round_trips_through_json() -> None:
    """An encoded frame should decode back to the original dict."""
    msg = {"jsonrpc": "2.0", "id": 1, "method": "foo", "params": {"x": 42}}
    encoded = _encode(msg)
    assert encoded.endswith(b"\n")
    # Exactly one trailing newline — line-delimited protocol.
    assert not encoded[:-1].endswith(b"\n")
    assert json.loads(encoded.decode()) == msg


# ── _rpc_response ──────────────────────────────────────────


def test_rpc_response_shape() -> None:
    """A response frame should carry jsonrpc, id, and result keys."""
    frame = _rpc_response(7, {"ok": True})
    decoded = json.loads(frame.decode())
    assert decoded == {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
    assert frame.endswith(b"\n")


# ── _stringify_raw ─────────────────────────────────────────


def test_stringify_raw_passes_strings_through() -> None:
    """A string input should come back unchanged, not double-encoded."""
    assert _stringify_raw("hello") == "hello"


def test_stringify_raw_indents_dicts() -> None:
    """A dict input should come back as indented JSON."""
    out = _stringify_raw({"a": 1, "b": [2, 3]})
    # json.dumps with indent=2 always indents nested structures.
    assert "\n" in out
    assert json.loads(out) == {"a": 1, "b": [2, 3]}


def test_stringify_raw_falls_back_for_unserializable() -> None:
    """Non-JSON-serializable values should fall back to ``str()``, not raise."""

    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    weird = Weird()
    # Must not raise.
    out = _stringify_raw(weird)
    assert out == "<weird>"


# ── _extract_text_blocks ───────────────────────────────────


def test_extract_text_blocks_returns_empty_for_non_list() -> None:
    """Non-list inputs should yield an empty list, not raise."""
    assert _extract_text_blocks(None) == []
    assert _extract_text_blocks("not a list") == []
    assert _extract_text_blocks({"not": "a list"}) == []


def test_extract_text_blocks_pulls_text_from_valid_blocks() -> None:
    """Well-formed content blocks should yield their inner text."""
    content = [
        {"type": "content", "content": {"text": "first"}},
        {"type": "content", "content": {"text": "second"}},
    ]
    assert _extract_text_blocks(content) == ["first", "second"]


def test_extract_text_blocks_skips_wrong_type() -> None:
    """Blocks whose ``type`` is not ``content`` should be ignored."""
    content = [
        {"type": "image", "content": {"text": "ignored"}},
        {"type": "content", "content": {"text": "kept"}},
    ]
    assert _extract_text_blocks(content) == ["kept"]


def test_extract_text_blocks_skips_missing_or_malformed_inner() -> None:
    """Inner ``content`` that isn't a dict or has no ``text`` should be skipped."""
    content: list[Any] = [
        {"type": "content", "content": "not a dict"},
        {"type": "content", "content": {}},  # missing text key
        {"type": "content", "content": {"text": 123}},  # wrong text type
        {"type": "content", "content": {"text": "kept"}},
        "not a dict at all",  # also exercised by the outer guard
    ]
    assert _extract_text_blocks(content) == ["kept"]


# ── _format_permission_prompt ──────────────────────────────


def test_format_permission_prompt_with_dict_raw_input() -> None:
    """A dict ``rawInput`` should render inside a fenced ``json`` block."""
    body = _format_permission_prompt({"title": "bash: ls", "rawInput": {"cmd": "ls"}})
    assert "`bash: ls`" in body
    assert "```json" in body
    assert '"cmd": "ls"' in body


def test_format_permission_prompt_with_string_raw_input() -> None:
    """A string ``rawInput`` should be embedded verbatim, not re-encoded."""
    body = _format_permission_prompt({"title": "bash: ls", "rawInput": "ls -la"})
    assert "`bash: ls`" in body
    assert "ls -la" in body
    # No double-encoding into a JSON string literal.
    assert '"ls -la"' not in body


def test_format_permission_prompt_without_raw_input() -> None:
    """Missing ``rawInput`` should produce no fenced block, no crash."""
    body = _format_permission_prompt({"title": "bash: ls"})
    assert "`bash: ls`" in body
    assert "```" not in body


def test_format_permission_prompt_with_unserializable_raw_input() -> None:
    """An unserializable ``rawInput`` should fall back to ``str()``, not raise."""

    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    body = _format_permission_prompt({"title": "tool", "rawInput": Weird()})
    assert "`tool`" in body
    assert "<weird>" in body


def test_format_permission_prompt_default_title() -> None:
    """A missing title should fall back to a generic label."""
    body = _format_permission_prompt({})
    assert "`tool call`" in body
