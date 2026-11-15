"""A one-tool stack whose tool reads an input file (``tests/e2e/denial_reason.py``)."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("denial-stack")


@mcp.tool()
def inspect(input_path: str) -> dict[str, object]:
    """Return the size of *input_path*."""
    return {"bytes": Path(input_path).stat().st_size}


if __name__ == "__main__":
    mcp.run()
