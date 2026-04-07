"""Smoke tests verifying the package and toolchain are wired up correctly (dummy test)."""

import medmcp


def test_package_imports() -> None:
    """The package should be importable."""
    assert medmcp.__version__ == "0.1.0"
