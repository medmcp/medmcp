"""The vibe patches applied at image build still fit the locked vibe version."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "patch_vibe.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("patch_vibe", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["patch_vibe"] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vibe_copy(tmp_path: Path) -> Path:
    """A writable copy of the installed vibe package."""
    spec = importlib.util.find_spec("vibe")
    assert spec is not None and spec.origin is not None
    target = tmp_path / "vibe"
    shutil.copytree(Path(spec.origin).parent, target)
    return target


def test_every_patch_applies_once(vibe_copy: Path) -> None:
    """Each snippet is found exactly once and replaced; a second run changes nothing."""
    patch_vibe = _load()
    first = patch_vibe.apply(vibe_copy)
    assert all(line.endswith(": applied") for line in first), first
    before = (vibe_copy / "core" / "scratchpad.py").read_text()
    second = patch_vibe.apply(vibe_copy)
    assert all(line.endswith("already applied") for line in second), second
    assert (vibe_copy / "core" / "scratchpad.py").read_text() == before


def test_patched_code_compiles(vibe_copy: Path) -> None:
    """The patched files are still valid Python."""
    _load().apply(vibe_copy)
    for rel in ("core/scratchpad.py", "core/tools/permissions.py"):
        compile((vibe_copy / rel).read_text(), rel, "exec")


def test_changed_vibe_fails_loudly(vibe_copy: Path) -> None:
    """If vibe's code moved, the build stops instead of shipping it unpatched."""
    file = vibe_copy / "core" / "tools" / "permissions.py"
    file.write_text(file.read_text().replace("def covers(", "def covers_renamed("))
    with pytest.raises(RuntimeError, match="does not apply"):
        _load().apply(vibe_copy)
