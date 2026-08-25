"""MedMCP: an LLM agent for medical imaging workflows."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the version declared in pyproject.toml, read back
    # from the installed distribution so the two can never drift.
    __version__ = version("medmcp")
except PackageNotFoundError:  # pragma: no cover - source tree with no install
    __version__ = "0+unknown"

__all__ = ["__version__"]
