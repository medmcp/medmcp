"""Tests for the runtime GPU selection (settings.load/save_gpu_selection)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from medmcp import settings


def test_load_defaults_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a persisted file, the selection follows MEDMCP_GPU."""
    monkeypatch.setenv("MEDMCP_GPU", "all")
    assert settings.load_gpu_selection() == "all"
    monkeypatch.setenv("MEDMCP_GPU", "4")
    assert settings.load_gpu_selection() == "4"


def test_save_persists_and_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save writes the file and applies the value to the process env."""
    sel = tmp_path / "gpu_selection.json"
    monkeypatch.setenv("MEDMCP_GPU", "all")
    with patch("medmcp.settings.GPU_SELECTION_PATH", sel):
        settings.save_gpu_selection("5")
        assert json.loads(sel.read_text()) == {"gpu": "5"}
    assert settings.load_gpu_selection() == "5"


def test_save_blank_falls_back_to_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank selection is normalised to "all" (every GPU)."""
    sel = tmp_path / "gpu_selection.json"
    monkeypatch.setenv("MEDMCP_GPU", "3")
    with patch("medmcp.settings.GPU_SELECTION_PATH", sel):
        settings.save_gpu_selection("   ")
        assert json.loads(sel.read_text()) == {"gpu": "all"}
    assert settings.load_gpu_selection() == "all"
