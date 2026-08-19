"""The security posture the README promises, asserted against both shipped configs.

The README tells users "No auto-approval… `web_search` is disabled and `web_fetch`
requires approval". Two configs have to deliver that — `docker/config.toml` baked
into the image and `.vibe/config.toml` for a host install — and they have drifted
apart before: one disabled `web_fetch` outright while the other left it on, and
both carried keys naming tools that no longer exist, so the intent was expressed
in a line vibe never read.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("docker/config.toml", ".vibe/config.toml")

# Tools that act on the machine or the network. None may be auto-approved.
MUST_ASK = ("bash", "edit", "write_file", "web_fetch")


def _load(rel: str) -> dict[str, Any]:
    with (ROOT / rel).open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize("rel", CONFIGS)
def test_no_auto_approval(rel: str) -> None:
    """The README's first security promise: nothing is approved for you."""
    assert _load(rel).get("auto_approve") is False


@pytest.mark.parametrize("rel", CONFIGS)
def test_acting_tools_require_approval(rel: str) -> None:
    """Every tool that touches the machine or the network asks first."""
    tools = _load(rel).get("tools", {})
    for name in MUST_ASK:
        assert name in tools, f"{rel}: no [tools.{name}] — the intent is unstated"
        assert tools[name].get("permission") == "ask", f"{rel}: {name} is not 'ask'"


@pytest.mark.parametrize("rel", CONFIGS)
def test_web_search_off_and_web_fetch_on_but_gated(rel: str) -> None:
    """The documented egress posture, exactly: search off, fetch on with approval."""
    cfg = _load(rel)
    disabled = cfg.get("disabled_tools", [])
    assert "web_search" in disabled
    assert "web_fetch" not in disabled, "web_fetch is meant to be available, but gated"


@pytest.mark.parametrize("rel", CONFIGS)
def test_no_allowlist_smuggles_an_approval(rel: str) -> None:
    """An allowlist entry is an approval nobody clicks; the bash one was emptied once."""
    for name, tool in _load(rel).get("tools", {}).items():
        assert not tool.get("allowlist"), f"{rel}: [tools.{name}] carries an allowlist"


@pytest.mark.parametrize("rel", CONFIGS)
def test_config_names_only_real_tools(rel: str) -> None:
    """A key naming a tool vibe does not have expresses nothing.

    Both configs have carried such keys (`read`, `search_replace`) after vibe
    renames, and stayed correct only because the defaults happened to agree.
    """
    tools_dir = ROOT / ".venv/lib/python3.12/site-packages/vibe/core/tools/builtins"
    builtins = {p.stem for p in tools_dir.glob("*.py")}
    if not builtins:  # not installed (e.g. a docs-only checkout)
        pytest.skip("vibe not installed")
    for name in _load(rel).get("tools", {}):
        assert name in builtins, f"{rel}: [tools.{name}] is not a vibe tool"


def test_both_install_paths_agree() -> None:
    """The container and host installs must not offer different postures."""
    image, host = (_load(c) for c in CONFIGS)
    assert image.get("disabled_tools") == host.get("disabled_tools")
    assert image.get("auto_approve") == host.get("auto_approve")
    assert {n: t.get("permission") for n, t in image.get("tools", {}).items()} == {
        n: t.get("permission") for n, t in host.get("tools", {}).items()
    }
