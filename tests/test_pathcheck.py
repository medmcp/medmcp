"""Tests for the tool-call path existence check (`medmcp.pathcheck`)."""

import json
from pathlib import Path

import pytest

from medmcp import pathcheck


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace with one existing file and one existing directory."""
    (tmp_path / "sub-01").mkdir()
    (tmp_path / "sub-01" / "t1.nii.gz").write_bytes(b"")
    (tmp_path / "derivatives").mkdir()
    return tmp_path


class TestClassifyRole:
    """Parameter-name classification."""

    @pytest.mark.parametrize(
        "param",
        ["input_path", "inputPath", "src_file", "moving_path", "fixed_path", "template_path"],
    )
    def test_input_names(self, param: str) -> None:
        """Names carrying an unambiguous input token classify as inputs."""
        assert pathcheck.classify_role(param) == "input"

    @pytest.mark.parametrize("param", ["output_dir", "outputPath", "out_file", "dest_dir"])
    def test_output_names(self, param: str) -> None:
        """Names carrying an unambiguous output token classify as outputs."""
        assert pathcheck.classify_role(param) == "output"

    @pytest.mark.parametrize("param", ["path", "bids_dir", "mask", "target_dir", "whatever"])
    def test_ambiguous_names_are_unknown(self, param: str) -> None:
        """Names that are an input in one tool and an output in another stay unknown.

        Guessing here would put a confident wrong flag on a legitimate argument.
        """
        assert pathcheck.classify_role(param) == "unknown"

    def test_contradictory_name_is_unknown(self) -> None:
        """A name claiming both roles resolves to neither rather than picking one."""
        assert pathcheck.classify_role("input_output_dir") == "unknown"


class TestLooksLikePath:
    """The value-shape heuristic."""

    @pytest.mark.parametrize("value", ["/abs/x.nii.gz", "sub-01/t1.nii", "report.pdf"])
    def test_accepts_paths(self, value: str) -> None:
        """A separator or a file suffix marks a value as a path."""
        assert pathcheck.looks_like_path(value)

    @pytest.mark.parametrize("value", ["", "ls -R data/x", "affine", "true"])
    def test_rejects_non_paths(self, value: str) -> None:
        """Notably a shell command, which contains a path but is not one."""
        assert not pathcheck.looks_like_path(value)


class TestInputPaths:
    """An input must already exist — this is the hallucination case."""

    def test_existing_input_is_ok(self, workspace: Path) -> None:
        """An input that is on disk reports ok, at info severity."""
        args = {"input_path": str(workspace / "sub-01" / "t1.nii.gz")}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert (finding["status"], finding["severity"]) == ("ok", "info")

    def test_missing_input_is_an_error(self, workspace: Path) -> None:
        """An invented input path is the one case that must show as an error."""
        args = {"input_path": str(workspace / "sub-99" / "invented.nii.gz")}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["status"] == "missing"
        assert finding["severity"] == "error"
        assert finding["role"] == "input"

    def test_relative_input_resolves_against_the_workspace(self, workspace: Path) -> None:
        """A relative value is judged relative to the workspace root."""
        (finding,) = pathcheck.check_tool_call_paths({"input_path": "sub-01/t1.nii.gz"}, workspace)
        assert finding["status"] == "ok"


class TestOutputPaths:
    """An output is *supposed* not to exist yet; only its parent must."""

    def test_absent_output_is_not_flagged(self, workspace: Path) -> None:
        """The regression that would make this feature cry wolf on every call."""
        args = {"output_dir": str(workspace / "derivatives" / "run-01")}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["severity"] == "info"
        assert finding["status"] == "ok"

    def test_output_with_missing_parent_is_an_error(self, workspace: Path) -> None:
        """An output under a folder that does not exist is a real problem."""
        args = {"output_dir": str(workspace / "nope" / "run-01")}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["status"] == "parent_missing"
        assert finding["severity"] == "error"

    def test_existing_output_warns_about_overwrite(self, workspace: Path) -> None:
        """An output that already exists would be overwritten — worth surfacing."""
        args = {"output_path": str(workspace / "sub-01" / "t1.nii.gz")}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["status"] == "will_overwrite"
        assert finding["severity"] == "warning"


class TestUnknownRole:
    """An unclassifiable parameter is reported, never judged."""

    def test_missing_unknown_is_never_an_error(self, workspace: Path) -> None:
        """A missing path under an ambiguous name is shown but not flagged red."""
        args = {"path": str(workspace / "maybe.csv")}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["status"] == "missing"
        assert finding["severity"] == "info"


class TestWorkspaceBoundary:
    """Paths outside the workspace are flagged whatever their role."""

    def test_outside_workspace_warns_even_when_it_exists(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        """A stack container cannot see it, so existing on the host is not enough."""
        outside = tmp_path.parent / "elsewhere.nii.gz"
        outside.write_bytes(b"")
        (finding,) = pathcheck.check_tool_call_paths({"input_path": str(outside)}, workspace)
        assert finding["status"] == "outside_workspace"
        assert finding["severity"] == "warning"

    def test_traversal_escape_is_caught(self, workspace: Path) -> None:
        """A ``..`` segment that leaves the workspace is normalised and caught."""
        args = {"input_path": str(workspace / ".." / "escaped.nii.gz")}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["status"] == "outside_workspace"

    def test_in_image_path_is_not_called_an_error(self, workspace: Path) -> None:
        """A path inside the stack image is unverifiable here, not wrong.

        A stack's own bundled reference data (e.g. the MNI template baked into the
        neuro image) is invisible to the core but perfectly visible to the tool. The
        check cannot tell that apart from a host path the stack cannot see, so it
        must not claim either — hence a warning, and a note that asserts nothing
        about what the stack can reach.
        """
        args = {"template_path": "/root/.medmcp_neuro_core/templates/MNI152.nii.gz"}
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["status"] == "outside_workspace"
        assert finding["severity"] == "warning"
        assert "cannot be verified" in finding["note"]


class TestArgumentShapes:
    """rawInput arrives in more than one shape."""

    def test_json_string_raw_input_is_parsed(self, workspace: Path) -> None:
        """vibe-acp ships some calls' rawInput as a JSON-encoded string."""
        args = json.dumps({"input_path": str(workspace / "sub-01" / "t1.nii.gz")})
        (finding,) = pathcheck.check_tool_call_paths(args, workspace)
        assert finding["status"] == "ok"

    def test_list_valued_argument_is_checked_element_wise(self, workspace: Path) -> None:
        """A parameter holding several paths yields one finding per element."""
        args = {"input_path": [str(workspace / "sub-01" / "t1.nii.gz"), "sub-01/missing.nii.gz"]}
        findings = pathcheck.check_tool_call_paths(args, workspace)
        assert [f["status"] for f in findings] == ["ok", "missing"]

    def test_non_path_arguments_are_skipped(self, workspace: Path) -> None:
        """Scalars and plain strings that name no path produce no findings."""
        args = {"transform_type": "affine", "skull_stripped": True, "threads": 8}
        assert pathcheck.check_tool_call_paths(args, workspace) == []

    def test_bare_name_output_dir_is_checked_despite_plain_value(self, workspace: Path) -> None:
        """The name says it is a path even though the value has no separator."""
        (finding,) = pathcheck.check_tool_call_paths({"output_dir": "derivatives"}, workspace)
        assert finding["param"] == "output_dir"

    @pytest.mark.parametrize("raw", [None, "not json", 42, [1, 2], ""])
    def test_uninterpretable_raw_input_yields_nothing(self, raw: object, workspace: Path) -> None:
        """Anything that is not an argument mapping is skipped, never raised on."""
        assert pathcheck.check_tool_call_paths(raw, workspace) == []

    def test_findings_are_capped(self, workspace: Path) -> None:
        """A pathological argument object cannot produce an unreadable dialog."""
        args = {f"input_{i}_path": f"/x/{i}.nii.gz" for i in range(60)}
        assert len(pathcheck.check_tool_call_paths(args, workspace)) <= 24
