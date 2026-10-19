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
def test_config_names_only_real_tools(rel: str) -> None:
    """A key naming a tool vibe does not have expresses nothing.

    Both configs have carried such keys (`read`, `search_replace`) after vibe
    renames, and stayed correct only because the defaults happened to agree.
    """
    try:
        import vibe.core.tools.builtins as builtins_pkg  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
    except ImportError:  # not installed (e.g. a docs-only checkout)
        pytest.skip("vibe not installed")
    builtins = {p.stem for p in Path(builtins_pkg.__path__[0]).glob("*.py")}
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


@pytest.mark.parametrize("rel", CONFIGS)
def test_agent_profile_keeps_edits_gated(rel: str) -> None:
    """The agent profile is ours, and it leaves every acting tool behind the prompt.

    vibe 2.26 reads a custom system prompt only from the active profile, and a
    profile's keys sit *above* the project config — so the profile is where the
    posture lives. It is parsed with vibe's own loader, so a key vibe would not
    honour fails here too. Built-in profiles that approve on their own
    (``accept-edits``, ``smart-approve``, ``auto-approve``) must never be the default.
    """
    agent = _load(rel).get("default_agent")
    assert agent == "medmcp", f"{rel}: default_agent must be pinned to 'medmcp'"
    try:
        from vibe.core.agents.models import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
            AgentProfile,
            AgentSafety,
        )
    except ImportError:  # pragma: no cover - docs-only checkout
        pytest.skip("vibe not installed")
    profile = AgentProfile.from_toml(ROOT / ".vibe" / "agents" / f"{agent}.toml")
    assert profile.safety == AgentSafety.NEUTRAL
    overrides: dict[str, Any] = profile.overrides
    assert not overrides.get("bypass_tool_permissions")
    assert overrides.get("system_prompt_id") == "medmcp"
    assert overrides.get("enable_subagents") is False
    assert overrides.get("enable_background_processes") is False
    assert {"cron", "web_search"} <= set(overrides.get("disabled_tools", []))
    tool_overrides: dict[str, Any] = overrides.get("tools", {})
    for name in MUST_ASK:
        assert tool_overrides.get(name, {}).get("permission") in (None, "ask"), (
            f"{rel}: the {agent!r} profile auto-approves {name}"
        )


@pytest.mark.parametrize("rel", CONFIGS)
def test_bash_allowlist_survives_vibes_migration(rel: str) -> None:
    """The one-shot allowlist migration in vibe must not re-arm the emptied allowlist.

    ``[tools.bash] allowlist = []`` is deliberate: an allowlisted command runs
    without a prompt. vibe ships a one-shot migration that unions its read-only
    defaults into any existing allowlist unless the config records it as already
    applied — so ``applied_migrations`` is what keeps the list empty in practice.
    Run vibe's real migration over the shipped config to prove it.
    """
    cfg = _load(rel)
    assert cfg.get("tools", {}).get("bash", {}).get("allowlist") == []
    assert "bash_read_only_defaults_v1" in cfg.get("applied_migrations", [])
    try:
        from vibe.core.config._migration import (  # pyright: ignore[reportMissingTypeStubs, reportPrivateImportUsage]  # vibe ships no stubs
            migrate_config,
        )
    except ImportError:  # pragma: no cover - docs-only checkout
        pytest.skip("vibe not installed")
    migrate_config(cfg)
    # vibe adds `find` unconditionally (that one carries no one-shot id); the
    # read-only defaults it would otherwise union in must stay out.
    assert set(cfg["tools"]["bash"]["allowlist"]) <= {"find"}


@pytest.mark.parametrize("rel", CONFIGS)
def test_vibe_makes_no_unattended_requests(rel: str) -> None:
    """Every switch behind which vibe talks to Mistral is off, and vibe knows each key.

    The experiments client is gated on telemetry in vibe's code; the pin makes
    the posture independent of that gate.
    """
    from vibe.core.config.models import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
        ExperimentsConfig,
    )
    from vibe.core.config.vibe_schema import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
        VibeConfigSchema,
    )

    cfg = _load(rel)
    for key in ("enable_telemetry", "enable_update_checks", "enable_auto_update"):
        assert key in VibeConfigSchema.model_fields
        assert cfg[key] is False, key
    assert "experiments" in VibeConfigSchema.model_fields
    assert "enable" in ExperimentsConfig.model_fields
    assert cfg["experiments"]["enable"] is False


@pytest.mark.parametrize("rel", CONFIGS)
def test_stalled_stream_fails_the_turn(rel: str) -> None:
    """A model stream that goes silent fails instead of hanging the chat."""
    from vibe.core.config.vibe_schema import (  # pyright: ignore[reportMissingTypeStubs]  # vibe ships no stubs
        VibeConfigSchema,
    )

    cfg = _load(rel)
    assert "api_stream_idle_timeout" in VibeConfigSchema.model_fields
    assert cfg["api_stream_idle_timeout"] >= 60
