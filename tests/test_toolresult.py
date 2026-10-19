"""Tool results under both vibe generations read the same way."""

from __future__ import annotations

from medmcp.toolresult import looks_failed, parse_structured_blob


def test_structured_blob_from_the_legacy_rendering() -> None:
    """Older vibe rendered ``structured: {...}`` into the tool text."""
    text = "ok: True\nstructured: {'segmentation_path': '/ws/x_dseg.nii.gz', 'n': 3}"
    assert parse_structured_blob(text) == {"segmentation_path": "/ws/x_dseg.nii.gz", "n": 3}


def test_structured_blob_from_plain_json() -> None:
    """Newer vibe hands the MCP result over as JSON text; a non-object is not a blob."""
    assert parse_structured_blob('{\n  "parc": true,\n  "total_structures": 99\n}') == {
        "parc": True,
        "total_structures": 99,
    }
    assert parse_structured_blob("[1, 2]") == {}
    assert parse_structured_blob("{not json") == {}
    assert parse_structured_blob("plain text") == {}


def test_looks_failed_markers() -> None:
    """The failure markers of both generations, plus the path guard's refusal."""
    assert looks_failed("Error executing tool x: boom")
    assert looks_failed("ok: False\nstderr: bad")
    assert looks_failed("RETRY THIS CALL WITH A CORRECTED PATH.\n\nNobody refused this call")
    assert looks_failed("command failed (exit 2)")
    assert not looks_failed('{"ok": true}')


def test_structured_blob_ignores_text_after_the_object() -> None:
    """The harness may follow the JSON result with another block (the route it called)."""
    text = '{\n  "brain_path": "/ws/t1_brain.nii.gz"\n}\nmcp_medmcp_neuro_core.skull_strip'
    assert parse_structured_blob(text) == {"brain_path": "/ws/t1_brain.nii.gz"}
