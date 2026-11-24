"""Minimal FastMCP stdio server used by tests/test_backend_pool.py.

Exposes a handful of tools that let the backend-pool tests observe lifecycle
behaviour: ``echo`` (basic call), ``warmup`` + ``warmup_count`` (the pre-warm
hook plus a counter that proves in-process state persists across calls of a
warm backend), and ``crash`` (kills the process to exercise death/respawn).
Four more declare a tool effect, one per way of declaring it (see
``medmcp.tooleffect``): ``inspect_image``, ``make_mask``, ``erase_file`` and
``odd_effect``; ``echo`` declares none.
"""

from __future__ import annotations

import asyncio
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

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


# Tools that declare an effect (medmcp.tooleffect), one per way of declaring it.
# None of them touches a file: they only report what they were asked to do.


@mcp.tool(meta={"org.medmcp/effect": "read_only"})
def inspect_image(input_path: str) -> dict[str, str]:
    """Report the header of an image (declared read-only through ``_meta``)."""
    return {"inspected": input_path}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
def make_mask(input_path: str, output_path: str) -> dict[str, str]:
    """Write a mask next to an image (writes new files, through both annotations)."""
    return {"input": input_path, "mask": output_path}


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True))
def erase_file(path: str) -> dict[str, str]:
    """Delete a file (declared destructive through ``destructiveHint``)."""
    return {"erased": path}


@mcp.tool(
    meta={"org.medmcp/effect": "harmless"},
    annotations=ToolAnnotations(readOnlyHint=True),
)
def odd_effect(path: str) -> dict[str, str]:
    """A ``_meta`` value outside the contract: untagged, the annotation not consulted."""
    return {"path": path}


if __name__ == "__main__":
    mcp.run()
