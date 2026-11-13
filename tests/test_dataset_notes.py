"""The dataset lines the workspace adds to a prompt (``medmcp.workspace_note``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medmcp.workspace_note import (
    DATASET_SUMMARY_RELPATH,
    build_dataset_notes,
    describe_dataset,
    find_dataset_dirs,
    strip_workspace_note,
)


def _scans(folder: Path) -> Path:
    anat = folder / "sub-01" / "ses-01" / "anat"
    anat.mkdir(parents=True)
    (anat / "sub-01_ses-01_T2w.nii.gz").write_bytes(b"")
    return folder


def _index(folder: Path, **summary: object) -> None:
    target = folder / DATASET_SUMMARY_RELPATH
    target.parent.mkdir(parents=True, exist_ok=True)
    base: dict[str, object] = {
        "n_subjects": 3,
        "n_sessions": 18,
        "n_series": 315,
        "sequences": {"T1": 54, "FLAIR": 18},
        "planes": {"axial": 300, "coronal": 15},
        "contrast": {"unknown": 315},
        "date_coverage": "17 sessions with a usable date / 18 sessions",
        "n_unknown_sequence": 4,
    }
    target.write_text(json.dumps({**base, **summary}), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty workspace root."""
    root = tmp_path / "workspace"
    root.mkdir()
    return root


class TestWhichFoldersAPromptMeans:
    """Which folders a prompt means."""

    def test_an_absolute_path_in_the_prompt(self, workspace: Path) -> None:
        """An absolute path in the prompt."""
        data = _scans(workspace / "atss_testdata")
        assert find_dataset_dirs(f"what is in {data}?", None, workspace) == [data]

    def test_a_path_below_an_indexed_folder_means_that_folder(self, workspace: Path) -> None:
        """A path below an indexed folder means that folder."""
        data = _scans(workspace / "atss_testdata")
        _index(data)
        deep = data / "sub-01" / "ses-01"
        assert find_dataset_dirs(f"look at {deep}", None, workspace) == [data]

    @pytest.mark.parametrize(
        ("folder", "prompt"),
        [
            ("atss_testdata", "here is the atss testdata folder, whats in it"),
            ("open_ms_data-master", "how many subjects are in the open ms cross sectional dataset"),
            ("Braintestdata_bids", "use braintestdata bids please"),
        ],
    )
    def test_a_folder_named_the_way_people_type_it(
        self, workspace: Path, folder: str, prompt: str
    ) -> None:
        """A folder named the way people type it."""
        data = _scans(workspace / folder)
        assert find_dataset_dirs(prompt, None, workspace) == [data]

    @pytest.mark.parametrize("prompt", ["segment the ms lesions", "show me the test data", "hi"])
    def test_a_generic_or_short_word_names_no_folder(self, workspace: Path, prompt: str) -> None:
        """A generic or short word names no folder."""
        _scans(workspace / "ms_data")
        _scans(workspace / "test")
        assert find_dataset_dirs(prompt, None, workspace) == []

    def test_the_viewed_file_brings_its_indexed_folder(self, workspace: Path) -> None:
        """The viewed file brings its indexed folder."""
        data = _scans(workspace / "study")
        _index(data)
        viewed = data / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T2w.nii.gz"
        assert find_dataset_dirs("what is this image", str(viewed), workspace) == [data]

    def test_nothing_outside_the_workspace(self, workspace: Path, tmp_path: Path) -> None:
        """Nothing outside the workspace."""
        outside = _scans(tmp_path / "elsewhere")
        assert find_dataset_dirs(f"index {outside} and {workspace}", None, workspace) == []


class TestWhatIsSaidAboutAFolder:
    """What is said about a folder."""

    def test_an_indexed_folder_is_described_from_its_summary(self, workspace: Path) -> None:
        """An indexed folder is described from its summary."""
        data = _scans(workspace / "study")
        _index(data)
        line = describe_dataset(data)
        assert line is not None
        assert line.startswith("[workspace context:")
        assert "3 subjects, 18 sessions, 315 scans" in line
        assert "T1 54, FLAIR 18" in line
        assert "17 sessions with a usable date / 18 sessions" in line
        assert "4 scans with unknown sequence" in line
        assert "summarize_dataset" in line
        assert "index_dataset" not in line

    def test_a_folder_of_scans_without_an_index_asks_for_one(self, workspace: Path) -> None:
        """A folder of scans without an index asks for one."""
        data = _scans(workspace / "study")
        line = describe_dataset(data)
        assert line is not None
        assert "index_dataset" in line
        assert str(data) in line

    def test_a_folder_without_scans_gets_no_line(self, workspace: Path) -> None:
        """A folder without scans gets no line."""
        tables = workspace / "tables"
        tables.mkdir()
        (tables / "patients.csv").write_text("id,age\n1,40\n")
        assert describe_dataset(tables) is None

    def test_a_broken_summary_falls_back_to_asking_for_an_index(self, workspace: Path) -> None:
        """A broken summary falls back to asking for an index."""
        data = _scans(workspace / "study")
        (data / ".medmcp").mkdir()
        (data / DATASET_SUMMARY_RELPATH).write_text("{not json")
        line = describe_dataset(data)
        assert line is not None
        assert "index_dataset" in line


def test_the_lines_are_removed_with_the_viewer_note(workspace: Path) -> None:
    """The lines are removed with the viewer note."""
    data = _scans(workspace / "study")
    _index(data)
    notes = build_dataset_notes(f"what is in {data}", None, workspace)
    assert notes
    stored = f"what is in {data}\n\n{notes}"
    assert strip_workspace_note(stored) == f"what is in {data}"


def test_no_folder_no_note(workspace: Path) -> None:
    """No folder no note."""
    assert build_dataset_notes("what can you do", None, workspace) == ""
