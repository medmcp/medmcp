"""Smoke tests verifying the package and toolchain are wired up correctly (dummy test)."""

import medmcp


def test_package_imports() -> None:
    """The package should be importable and know its version.

    The number is not pinned here — it lives in pyproject.toml alone, and
    ``test_release_meta`` is what checks the two agree.
    """
    assert medmcp.__version__
