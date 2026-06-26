"""Minimal FastMCP stdio server used by tests/test_backend_pool.py.

Exposes a handful of tools that let the backend-pool tests observe lifecycle
behaviour: ``echo`` (basic call), ``warmup`` + ``warmup_count`` (the pre-warm
hook plus a counter that proves in-process state persists across calls of a
warm backend), and ``crash`` (kills the process to exercise death/respawn).
"""

from __future__ import annotations

import asyncio
import os

from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("fake-stack")

_warmups: int = 0


@mcp.tool()
def echo(text: str) -> dict[str, str]:
    """Return *text* unchanged."""
    return {"text": text}


@mcp.tool()
def warmup() -> dict[str, bool]:
    """Pre-warm hook; increments an in-process counter."""
    global _warmups
    _warmups += 1
    return {"ok": True}


@mcp.tool()
def warmup_count() -> dict[str, int]:
    """Return how many times ``warmup`` ran in this process."""
    return {"count": _warmups}


@mcp.tool()
async def sleep(seconds: float) -> dict[str, float]:
    """Sleep *seconds* then return; used to prove calls overlap, not serialize."""
    await asyncio.sleep(seconds)
    return {"slept": seconds}


@mcp.tool()
def crash() -> dict[str, str]:
    """Terminate the process immediately (simulates a backend dying mid-call)."""
    os._exit(0)


if __name__ == "__main__":
    mcp.run()
