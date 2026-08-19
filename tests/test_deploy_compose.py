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
