"""Tests for ``_read_line``: the JSON-RPC frame parser used by ``VibeAcpClient``.

These tests use a real :class:`asyncio.StreamReader` driven via
``feed_data``/``feed_eof``, so we get end-to-end exercise of the actual code
path the subprocess reader hits in production.
"""

from __future__ import annotations

import asyncio

from medmcp.app import _read_line


def _make_reader(payload: bytes, *, eof: bool = True) -> asyncio.StreamReader:
    """Build a primed :class:`asyncio.StreamReader` for tests."""
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    if eof:
        reader.feed_eof()
    return reader


async def test_read_line_returns_dict_for_clean_json() -> None:
    """A single well-formed JSON line should round-trip into a dict."""
    reader = _make_reader(b'{"jsonrpc":"2.0","id":1,"result":{}}\n')
    msg = await _read_line(reader)
    assert msg == {"jsonrpc": "2.0", "id": 1, "result": {}}


async def test_read_line_returns_none_on_eof() -> None:
    """EOF without any data should yield ``None``, not raise."""
    reader = _make_reader(b"")
    assert await _read_line(reader) is None


async def test_read_line_skips_non_json_noise() -> None:
    """Non-JSON log noise on stdout should be skipped, not propagated."""
    reader = _make_reader(b'plain log line\n{"valid":true}\n')
    msg = await _read_line(reader)
    assert msg == {"valid": True}


async def test_read_line_skips_non_dict_json_values() -> None:
    """Bare numbers / strings / arrays are not JSON-RPC frames; skip them."""
    reader = _make_reader(b'42\n"hi"\n[1,2]\n{"ok":1}\n')
    msg = await _read_line(reader)
    assert msg == {"ok": 1}


async def test_read_line_returns_none_after_consuming_only_noise() -> None:
    """If the stream contains only noise and then closes, we should get ``None``."""
    reader = _make_reader(b"noise one\nnoise two\n")
    assert await _read_line(reader) is None
