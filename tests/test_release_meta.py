"""Guards on the release plumbing.

A release is cut by pushing a `v*` tag, and `release.yml` refuses to publish
unless the tag, `pyproject.toml` and the changelog agree. Those checks run at
release time, when the fix is a retag; these run in CI, when it is an edit.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

import medmcp
from medmcp import server

ROOT = Path(__file__).resolve().parents[1]


def _declared_version() -> str:
    """Return the version literal in pyproject.toml."""
    match = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE)
    assert match is not None, "pyproject.toml has no version"
    return match.group(1)


def _extract() -> Callable[[str, str], str]:
    """Load `scripts/changelog_section.py`, which is tooling and not importable."""
    path = ROOT / "scripts" / "changelog_section.py"
    spec = importlib.util.spec_from_file_location("changelog_section", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(Callable[[str, str], str], module.extract)


class TestVersion:
    """One version string, reachable from the running instance."""

    def test_package_version_matches_pyproject(self) -> None:
        """`__version__` is read back from the distribution, so it cannot drift."""
        assert medmcp.__version__ == _declared_version()

    def test_healthz_reports_version_and_build(self) -> None:
        """An installed instance can say what it is — the first question a bug report asks."""
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        payload = cast(dict[str, str], client.get("/healthz").json())
        assert payload["status"] == "ok"
        assert payload["version"] == medmcp.__version__
        assert "build" in payload  # empty outside a built image


class TestChangelogSection:
    """The release notes come from the changelog; an empty section must be loud."""

    def test_extracts_body_without_the_heading(self) -> None:
        """The section stops at the next version heading."""
        text = "# Changelog\n\n## 0.2.0 — 2026-01-01\n\nnew things\n\n## 0.1.0\n\nold things\n"
        assert _extract()(text, "0.2.0") == "new things"

    def test_accepts_the_bracketed_spelling(self) -> None:
        """Keep a Changelog links versions as `## [0.2.0]`; both forms are in the file."""
        assert _extract()("## [0.2.0]\n\nbody\n", "0.2.0") == "body"

    def test_missing_version_raises(self) -> None:
        """A tag with no changelog section must stop the release, not ship empty notes."""
        with pytest.raises(LookupError):
            _extract()("## 0.1.0\n\nbody\n", "9.9.9")

    def test_shipped_changelog_has_a_section_for_the_current_version(self) -> None:
        """Whatever `pyproject.toml` declares must be releasable from this tree."""
        changelog = (ROOT / "CHANGELOG.md").read_text()
        assert _extract()(changelog, _declared_version()).strip()


def test_compose_still_carries_the_tag_default_release_rewrites() -> None:
    """`release.yml` pins the published compose by substituting this exact literal.

    Renaming it would leave `compose:vX.Y.Z` pulling `core:main` — a pinned
    install that keeps moving. The release job fails loudly on this, but CI is
    the cheaper place to find out.
    """
    compose = (ROOT / "docker-compose.ghcr.yml").read_text()
    assert "${MEDMCP_TAG:-main}" in compose
