"""A one-tool stack whose input schema nests a model (``$defs``/``$ref``).

Used by ``tests/e2e/invalid_args.py``: pydantic writes the nested argument as a
``$ref``, which is the shape vibe's own check before the hooks lets through.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp: FastMCP = FastMCP("args-stack")


class Criteria(BaseModel):
    """What to pick: one label or several."""

    labels: str | list[str] | None = None


@mcp.tool()
def pick(criteria: Criteria) -> dict[str, object]:
    """Return the criteria as understood."""
    return {"picked": criteria.model_dump()}


if __name__ == "__main__":
    mcp.run()
