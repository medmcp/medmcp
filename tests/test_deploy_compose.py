"""Guards on the published deploy compose.

`docker-compose.ghcr.yml` is what the documented one-command install actually
runs, and it has to be self-contained — so it inlines the Modelfile rather than
bind-mounting it. That duplication is exactly the kind that drifts silently: the
inlined values kept the previous model's sampling parameters for some time after
`Modelfile.muse` moved on, so every user of the documented install built a
differently tuned model than the one that was validated.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _modelfile_params() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / "Modelfile.muse").read_text().splitlines():
        if line.startswith("PARAMETER "):
            _, key, value = line.split(None, 2)
            out[key] = value.strip()
    return out


def _inlined_params() -> dict[str, str]:
    text = (ROOT / "docker-compose.ghcr.yml").read_text()
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"'PARAMETER (\w+) ([^']+)'", text)}


def test_inlined_modelfile_matches_the_source_of_truth() -> None:
    """The compose's inlined PARAMETERs must equal Modelfile.muse's."""
    expected, actual = _modelfile_params(), _inlined_params()
    assert expected, "Modelfile.muse declares no PARAMETERs — did it move?"
    assert actual, "docker-compose.ghcr.yml inlines no PARAMETERs — did the format change?"
    assert actual == expected


def test_inlined_base_model_matches_the_modelfile_from_line() -> None:
    """The base model the compose pulls must be the one the Modelfile derives from."""
    from_line = next(
        line.split(None, 1)[1].strip()
        for line in (ROOT / "Modelfile.muse").read_text().splitlines()
        if line.startswith("FROM ")
    )
    compose = (ROOT / "docker-compose.ghcr.yml").read_text()
    assert f"MEDMCP_BASE_MODEL:-{from_line}" in compose


def _vibe_mounts(filename: str) -> set[str]:
    """Return every ``name:/app/.vibe/<dir>`` volume entry in a compose file."""
    doc = cast("dict[str, Any]", yaml.safe_load((ROOT / filename).read_text()))
    services = cast("dict[str, Any]", doc["services"])
    found: set[str] = set()
    for service in services.values():
        for entry in cast("list[Any]", cast("dict[str, Any]", service).get("volumes") or []):
            if isinstance(entry, str) and ":/app/.vibe/" in entry:
                found.add(entry)
    return found


def test_both_compose_files_persist_the_same_vibe_state() -> None:
    """Whatever the local compose keeps across an image update, the published one keeps too.

    ``.vibe`` otherwise lives in the container's writable layer, so a directory
    mounted in one file and not the other means the documented install silently
    loses state a developer keeps — settings, chat history, or a configured
    external server and the consent that armed it.
    """
    local = _vibe_mounts("docker-compose.yml")
    assert local, "no .vibe volumes found — did the mount format change?"
    assert local == _vibe_mounts("docker-compose.ghcr.yml")
