"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_stacks_d(  # pyright: ignore[reportUnusedFunction]  # autouse fixture, invoked by pytest
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Stop tests from discovering the repo's real ``stacks.d/*.toml`` manifests.

    ``load_mcp_servers`` reads ``settings.STACKS_D_PATH`` (the committed
    ``stacks.d/`` dir) as a real on-disk source; without this, any test that
    exercises discovery would pick up the shipped container manifests. Points it
    at a non-existent dir by default; the stacks.d tests override it themselves.
    """
    absent = tmp_path_factory.mktemp("no_stacks") / "absent"
    with patch("medmcp.settings.STACKS_D_PATH", absent):
        yield
