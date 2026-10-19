"""The harness's tool routes translate to MedMCP's names once, at the edge."""

from __future__ import annotations

import pytest

from medmcp import harness


class TestCanonicalToolName:
    """Routes → the names provenance, distillation, replay and the UI key on."""

    @pytest.mark.parametrize(
        ("route", "expected"),
        [
            ("file_system.bash", "bash"),
            ("file_system.read_file", "read_file"),
            ("file_system.write_file", "write_file"),
            ("file_system.search_replace", "edit"),
            ("skill.read", "skill"),
            ("process.start", "bash"),
            ("process.stop", "bash"),
        ],
    )
    def test_builtin_routes(self, route: str, expected: str) -> None:
        """Each built-in route stands for one vibe tool; the process family is the shell."""
        assert harness.canonical_tool_name(route) == expected

    def test_mcp_route_uses_the_known_server_name(self) -> None:
        """``mcp_medmcp_neuro.x`` is ``medmcp-neuro_x`` when that server is installed."""
        name = harness.canonical_tool_name(
            "mcp_medmcp_neuro.skull_strip", ["medmcp-dicom", "medmcp-neuro"]
        )
        assert name == "medmcp-neuro_skull_strip"

    def test_mcp_route_for_an_unknown_server_keeps_the_squashed_spelling(self) -> None:
        """Nothing to map back to: the route's own spelling is still a usable name."""
        assert harness.canonical_tool_name("mcp_other_server.tool", []) == "other_server_tool"

    def test_vibe_group_routes_are_the_tool(self) -> None:
        """``vibe.todo`` is the todo tool; the plan card keys on that name."""
        assert harness.canonical_tool_name("vibe.todo") == "todo"

    def test_harness_tools_and_plain_names_pass_through(self) -> None:
        """A harness tool or an already-canonical name is left alone."""
        for name in ("search_tool_functions", "run_typescript", "todo", "medmcp-neuro_skull_strip"):
            assert harness.canonical_tool_name(name, ["medmcp-neuro"]) == name

    def test_group_squashes_every_non_identifier_character(self) -> None:
        """The harness's own rule: anything but [A-Za-z0-9_] becomes an underscore."""
        assert harness.mcp_group("my.server-2 x") == "mcp_my_server_2_x"
        assert harness.normalize_segment("") == "_"

    def test_harness_tools_are_known(self) -> None:
        """The three harness-owned tools are never workflow steps."""
        assert harness.is_harness_tool("run_typescript")
        assert not harness.is_harness_tool("bash")


class TestUnwrapResult:
    """The harness's result envelope comes off; older renderings pass through."""

    def test_envelope_with_json_text(self) -> None:
        """Text blocks join; a JSON object in them is the structured value."""
        raw = {"type": "success", "content": [{"type": "text", "text": '{"n": 3, "ok": true}'}]}
        text, value = harness.unwrap_result(raw)
        assert text == '{"n": 3, "ok": true}'
        assert value == {"n": 3, "ok": True}

    def test_envelope_with_plain_text(self) -> None:
        """Plain text stays text, as both halves."""
        raw = {
            "type": "success",
            "content": [{"type": "text", "text": "done"}, {"type": "text", "text": "ok"}],
        }
        assert harness.unwrap_result(raw) == ("done\nok", "done\nok")

    def test_legacy_values_pass_through(self) -> None:
        """A string or a structured object from older vibe is returned as it was."""
        assert harness.unwrap_result("ok: True") == ("ok: True", "ok: True")
        assert harness.unwrap_result({"path": "/x"}) == ('{"path": "/x"}', {"path": "/x"})
        assert harness.unwrap_result(None) == ("", None)

    def test_value_is_the_leading_object_when_blocks_follow(self) -> None:
        """A route-name block after the JSON result must not spoil the value."""
        raw = {
            "type": "success",
            "content": [
                {"type": "text", "text": '{"brain_path": "/ws/b.nii.gz"}'},
                {"type": "text", "text": "mcp_medmcp_neuro_core.skull_strip"},
            ],
        }
        text, value = harness.unwrap_result(raw)
        assert text.startswith('{"brain_path"') and text.endswith("skull_strip")
        assert value == {"brain_path": "/ws/b.nii.gz"}
