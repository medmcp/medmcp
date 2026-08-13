"""Tests for the pre_tool path guard (`medmcp.pathguard`)."""

# pyright: reportPrivateUsage=false

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

from medmcp import pathguard, settings


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace holding one real subject directory."""
    (tmp_path / "patient_05").mkdir()
    (tmp_path / "patient_05" / "t1.nii.gz").write_bytes(b"")
    return tmp_path


@pytest.fixture(autouse=True)
def isolate_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep each test's denial counts out of the real .vibe directory."""
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))


def _invocation(
    workspace: Path, tool_input: dict[str, Any], session_id: str = "sess-1"
) -> dict[str, Any]:
    return {
        "hook_event_name": "pre_tool",
        "session_id": session_id,
        "transcript_path": "/tmp/t",
        "cwd": str(workspace),
        "tool_name": "medmcp-neuro-core_register_to_template",
        "tool_call_id": "call-1",
        "tool_input": tool_input,
    }


class TestDecisions:
    """Allow a call whose paths resolve; deny one whose paths cannot."""

    def test_valid_paths_are_allowed(self, workspace: Path) -> None:
        """A resolvable call passes straight through to the approval dialog."""
        inv = _invocation(workspace, {"input_path": str(workspace / "patient_05" / "t1.nii.gz")})
        assert pathguard.decide(inv) == {"decision": "allow"}

    def test_missing_input_is_denied(self, workspace: Path) -> None:
        """The call never reaches the user; the model is told what was wrong."""
        inv = _invocation(workspace, {"input_path": str(workspace / "patient_09" / "t1.nii.gz")})
        out = pathguard.decide(inv)
        assert out["decision"] == "deny"
        assert "do not resolve" in out["reason"]
        assert "patient_05/" in out["reason"]  # the listing that enables the fix

    def test_reported_bug_reason_names_the_real_file(self, workspace: Path) -> None:
        """A hallucinated home directory comes back with the correction inline."""
        inv = _invocation(workspace, {"input_path": "/home/someone-else/data/patient_05/t1.nii.gz"})
        out = pathguard.decide(inv)
        assert out["decision"] == "deny"
        assert "did you mean patient_05/t1.nii.gz" in out["reason"]

    def test_warnings_alone_do_not_deny(self, workspace: Path) -> None:
        """An overwrite is the user's judgement call, so it must reach the dialog."""
        inv = _invocation(workspace, {"output_path": str(workspace / "patient_05" / "t1.nii.gz")})
        assert pathguard.decide(inv)["decision"] == "allow"

    def test_call_without_paths_is_allowed(self, workspace: Path) -> None:
        """Nothing to check, so nothing to say."""
        inv = _invocation(workspace, {"transform_type": "affine", "threads": 8})
        assert pathguard.decide(inv) == {"decision": "allow"}


class TestRetryCap:
    """A model that will not converge must not loop against the hook unseen."""

    def test_same_bad_call_is_allowed_through_after_the_cap(self, workspace: Path) -> None:
        """Better a human sees the stuck call than it retries off-screen forever."""
        inv = _invocation(workspace, {"input_path": str(workspace / "nope" / "t1.nii.gz")})
        verdicts = [pathguard.decide(inv)["decision"] for _ in range(4)]
        assert verdicts[: pathguard._MAX_DENIALS] == ["deny"] * pathguard._MAX_DENIALS
        assert verdicts[pathguard._MAX_DENIALS :] == ["allow", "allow"]

    def test_a_different_wrong_path_gets_its_own_budget(self, workspace: Path) -> None:
        """Correcting to another wrong path is progress, not repetition."""
        first = _invocation(workspace, {"input_path": str(workspace / "a" / "t1.nii.gz")})
        second = _invocation(workspace, {"input_path": str(workspace / "b" / "t1.nii.gz")})
        for _ in range(pathguard._MAX_DENIALS):
            pathguard.decide(first)
        assert pathguard.decide(first)["decision"] == "allow"
        assert pathguard.decide(second)["decision"] == "deny"

    def test_cap_is_per_session(self, workspace: Path) -> None:
        """One chat exhausting its budget must not disarm the guard in another."""
        args = {"input_path": str(workspace / "nope" / "t1.nii.gz")}
        for _ in range(pathguard._MAX_DENIALS):
            pathguard.decide(_invocation(workspace, args))
        assert pathguard.decide(_invocation(workspace, args))["decision"] == "allow"
        other = _invocation(workspace, args, session_id="sess-2")  # different chat
        assert pathguard.decide(other)["decision"] == "deny"


class TestFailOpen:
    """A guard that cannot do its job must never block work."""

    @pytest.mark.parametrize(
        "inv",
        [
            {},
            {"tool_name": "x", "tool_input": None, "cwd": ""},
            {"tool_name": "x", "tool_input": "not json", "cwd": ""},
        ],
    )
    def test_unusable_invocation_allows(self, inv: dict[str, Any]) -> None:
        """Missing or malformed fields resolve to allow, never to deny."""
        assert pathguard.decide(inv)["decision"] == "allow"

    def test_unwritable_ledger_still_denies(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Losing the retry cap must not cost the check itself."""

        def _no_save(path: Path, ledger: dict[str, int]) -> None:
            return None

        monkeypatch.setattr(pathguard, "_save_ledger", _no_save)
        inv = _invocation(workspace, {"input_path": str(workspace / "nope" / "t1.nii.gz")})
        assert pathguard.decide(inv)["decision"] == "deny"


class TestRegistration:
    """The guard is inert unless vibe is told about it — in the right file.

    vibe reads hooks from ``hooks.toml`` only; a ``[[hooks]]`` block in
    ``config.toml`` is silently ignored, which is how the first version of this
    shipped inert.
    """

    def _hooks(self, home: Path) -> list[dict[str, Any]]:
        with (home / "hooks.toml").open("rb") as fh:
            return cast("list[dict[str, Any]]", tomllib.load(fh)["hooks"])

    def test_hook_is_written_to_hooks_toml(self, tmp_path: Path) -> None:
        """Not config.toml — vibe never reads hooks from there."""
        settings._ensure_pathguard_hook(tmp_path)
        assert not (tmp_path / "config.toml").exists()
        (hook,) = self._hooks(tmp_path)
        assert hook["type"] == "pre_tool"
        assert hook["command"] == "medmcp-pathguard"
        assert "match" not in hook  # every tool: write_file and edit take paths too
        assert hook["strict"] is False  # a hook that cannot run must not block work

    def test_registration_is_idempotent_and_keeps_other_hooks(self, tmp_path: Path) -> None:
        """Re-syncing must not accumulate duplicates or drop a user's own hooks."""
        (tmp_path / "hooks.toml").write_text(
            '[[hooks]]\nname = "someone-elses"\ntype = "post_tool"\ncommand = "x"\n'
        )
        for _ in range(3):
            settings._ensure_pathguard_hook(tmp_path)
        names = [h["name"] for h in self._hooks(tmp_path)]
        assert names.count(settings.PATHGUARD_HOOK_NAME) == 1
        assert "someone-elses" in names

    def test_unparseable_file_is_rewritten(self, tmp_path: Path) -> None:
        """A corrupt hooks.toml must not leave the guard unregistered."""
        (tmp_path / "hooks.toml").write_text("this is not toml {{{")
        settings._ensure_pathguard_hook(tmp_path)
        assert [h["name"] for h in self._hooks(tmp_path)] == [settings.PATHGUARD_HOOK_NAME]

    def test_the_shipped_container_file_registers_the_hook(self) -> None:
        """The image bakes hooks.toml so the guard is live on the first prompt."""
        shipped = Path(__file__).resolve().parents[1] / "docker" / "hooks.toml"
        with shipped.open("rb") as fh:
            hooks = cast("list[dict[str, Any]]", tomllib.load(fh)["hooks"])
        assert [h["name"] for h in hooks] == [settings.PATHGUARD_HOOK_NAME]


class TestHookContract:
    """vibe's contract: exit 0, one JSON object on stdout."""

    def _run(self, payload: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "medmcp.pathguard"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_denies_over_stdio(self, workspace: Path) -> None:
        """End-to-end through the actual entry point vibe will invoke."""
        payload = json.dumps(
            _invocation(workspace, {"input_path": str(workspace / "gone" / "t1.nii.gz")})
        )
        proc = self._run(payload)
        assert proc.returncode == 0
        assert json.loads(proc.stdout)["decision"] == "deny"

    @pytest.mark.parametrize("payload", ["", "not json at all", "[]", "null"])
    def test_garbage_stdin_exits_zero_and_allows(self, payload: str) -> None:
        """Anything unparseable still yields a valid allow, never a crash."""
        proc = self._run(payload)
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {"decision": "allow"}
