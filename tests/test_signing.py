"""Tests for image signature verification (``medmcp.signing``).

cosign itself is not exercised here (a stub stands in for the binary); what is
asserted is the policy — who must have signed what — and the verifier's
discipline: verify by the pulled digest, accept only a signature that covers
that digest, fall back to the baked trust root only for trust-root trouble,
and fail closed on everything else.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

# pyright: reportPrivateUsage=false
from medmcp import settings, signing, update

DIGEST = "sha256:" + "ab" * 32
OTHER = "sha256:" + "cd" * 32
POLICY = signing.Policy(
    identity_regexp=r"^https://github\.com/medmcp/medmcp/\.github/workflows/release\.yml@refs/tags/v",
    issuer="https://token.actions.githubusercontent.com",
)


def _proc(code: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["cosign"], returncode=code, stdout=stdout, stderr=stderr
    )


def _verified(*digests: str) -> str:
    return json.dumps(
        [{"critical": {"image": {"docker-manifest-digest": d}}, "optional": {}} for d in digests]
    )


class TestPolicy:
    """The bundled policy: what it covers and which identities it accepts."""

    def test_bundled_policy_covers_the_published_images(self) -> None:
        """core, base and the compose artifact are all under the one identity."""
        policies = signing.load_policies()
        for repo in ("ghcr.io/medmcp/core", "ghcr.io/medmcp/base", "ghcr.io/medmcp/compose"):
            assert repo in policies, repo
        assert (
            policies["ghcr.io/medmcp/core"].issuer == "https://token.actions.githubusercontent.com"
        )

    @pytest.mark.parametrize(
        ("identity", "accepted"),
        [
            (
                "https://github.com/medmcp/medmcp/.github/workflows/release.yml@refs/tags/v0.3.0",
                True,
            ),
            (
                "https://github.com/medmcp/medmcp/.github/workflows/release.yml@refs/tags/v1.0.0-rc1",
                True,
            ),
            ("https://github.com/medmcp/medmcp/.github/workflows/images.yml@refs/heads/main", True),
            # A fork, another branch, another workflow, a lookalike domain.
            (
                "https://github.com/someone/medmcp/.github/workflows/release.yml@refs/tags/v0.3.0",
                False,
            ),
            (
                "https://github.com/medmcp/medmcp/.github/workflows/release.yml@refs/heads/feature",
                False,
            ),
            ("https://github.com/medmcp/medmcp/.github/workflows/ci.yml@refs/tags/v0.3.0", False),
            ("https://github.com/medmcp/medmcp/.github/workflows/release.yml@refs/tags/vX", False),
            (
                "https://githubXcom/medmcp/medmcp/.github/workflows/release.yml@refs/tags/v0.3.0",
                False,
            ),
            (
                "https://github.com/medmcp/medmcp/.github/workflows/release.yml@refs/tags/v0.3.0/x",
                False,
            ),
        ],
    )
    def test_identity_regexp(self, identity: str, accepted: bool) -> None:
        """Only this repository's release and main-build workflows are trusted."""
        import re

        regexp = signing.load_policies()["ghcr.io/medmcp/core"].identity_regexp
        assert bool(re.fullmatch(regexp, identity)) is accepted

    def test_policy_for_matches_the_repository_whatever_the_reference(self) -> None:
        """Tag or digest, the policy is keyed by repository."""
        table = {"ghcr.io/medmcp/core": POLICY}
        assert signing.policy_for("ghcr.io/medmcp/core:v0.3.0", table) is POLICY
        assert signing.policy_for(f"ghcr.io/medmcp/core@{DIGEST}", table) is POLICY
        assert signing.policy_for("ghcr.io/medmcp/core", table) is POLICY
        assert signing.policy_for("ghcr.io/medmcp/dicom:main", table) is None
        assert signing.policy_for("ghcr.io/evil/core:v0.3.0", table) is None

    def test_split_reference(self) -> None:
        """A registry port is not a tag."""
        assert signing.split_reference("host:5000/x/y:1.0") == ("host:5000/x/y", "1.0", None)
        assert signing.split_reference("host:5000/x/y") == ("host:5000/x/y", None, None)
        assert signing.split_reference(f"a/b:c@{DIGEST}") == ("a/b", "c", DIGEST)

    def test_unreadable_policy_is_empty_not_permissive(self, tmp_path: Path) -> None:
        """No policy file means nothing is covered — and nothing is waved through."""
        assert signing.load_policies(tmp_path / "missing.json") == {}
        bad = tmp_path / "bad.json"
        bad.write_text('{"policies": [{"images": ["x"], "identity_regexp": "("}]}')
        assert signing.load_policies(bad) == {}


class TestDigest:
    """The digest verified is the one docker pulled under."""

    def test_index_digest_from_repo_digests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RepoDigests carries the index digest for the repository in question."""

        def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
            assert args[:3] == ["docker", "image", "inspect"]
            return _proc(
                0,
                json.dumps(
                    [{"RepoDigests": [f"mirror/core@{OTHER}", f"ghcr.io/medmcp/core@{DIGEST}"]}]
                ),
            )

        monkeypatch.setattr(signing, "_run", _run)
        assert signing.image_digest("ghcr.io/medmcp/core:v0.3.0") == DIGEST

    def test_local_build_has_no_digest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An image that was never pulled cannot be verified."""

        def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
            return _proc(0, json.dumps([{"RepoDigests": []}]))

        monkeypatch.setattr(signing, "_run", _run)
        with pytest.raises(signing.SignatureError, match="no registry digest"):
            signing.image_digest("medmcp-core:dev")

    def test_explicit_digest_wins(self) -> None:
        """A reference that already names its digest needs no lookup."""
        assert signing.image_digest(f"ghcr.io/medmcp/core@{DIGEST}") == DIGEST


class TestVerify:
    """What the verifier accepts, and what it will not."""

    @pytest.fixture
    def harness(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
        """A cosign stub whose answers are scripted per attempt; records the calls."""
        calls: list[list[str]] = []
        answers: list[subprocess.CompletedProcess[str]] = []

        def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            return answers.pop(0)

        monkeypatch.setattr(signing, "_run", _run)
        root = tmp_path / "trusted_root.json"
        root.write_text("{}")
        return {"calls": calls, "answers": answers, "root": root, "no_root": tmp_path / "absent"}

    def _verify(self, harness: dict[str, Any], *, root: str = "root") -> signing.Verified:
        return signing.verify_image(
            "ghcr.io/medmcp/core:v0.3.0",
            digest=DIGEST,
            policy=POLICY,
            cosign="/usr/local/bin/cosign",
            trusted_root=harness[root],
        )

    def test_verified_by_digest_with_the_policy(self, harness: dict[str, Any]) -> None:
        """Cosign is asked about the digest, with the identity and issuer pinned."""
        harness["answers"].append(_proc(0, _verified(DIGEST)))
        verified = self._verify(harness)
        assert verified.ref == f"ghcr.io/medmcp/core@{DIGEST}"
        (call,) = harness["calls"]
        assert call[:3] == ["/usr/local/bin/cosign", "verify", f"ghcr.io/medmcp/core@{DIGEST}"]
        assert call[call.index("--certificate-identity-regexp") + 1] == POLICY.identity_regexp
        assert call[call.index("--certificate-oidc-issuer") + 1] == POLICY.issuer
        assert "--trusted-root" not in call
        assert "--insecure-ignore-tlog" not in call

    def test_signature_for_another_digest_is_refused(self, harness: dict[str, Any]) -> None:
        """A valid signature that covers a different image is not a pass."""
        harness["answers"].append(_proc(0, _verified(OTHER)))
        with pytest.raises(signing.SignatureError, match="not the pulled digest"):
            self._verify(harness)

    def test_failed_verification_is_a_verdict(self, harness: dict[str, Any]) -> None:
        """An identity mismatch is final; the baked root is not tried."""
        harness["answers"].append(_proc(1, "", "Error: no matching CertificateIdentity found"))
        with pytest.raises(signing.SignatureError, match="no matching CertificateIdentity"):
            self._verify(harness)
        assert len(harness["calls"]) == 1

    def test_trust_root_trouble_falls_back_to_the_baked_root(self, harness: dict[str, Any]) -> None:
        """Offline (no TUF) → retry with --trusted-root; that verdict counts."""
        harness["answers"] += [
            _proc(
                1,
                "",
                "Error: setting trusted material: tuf refresh failed: "
                'Get "https://tuf-repo-cdn.sigstore.dev/15.root.json": dial tcp: connect',
            ),
            _proc(0, _verified(DIGEST)),
        ]
        assert self._verify(harness).digest == DIGEST
        first, second = harness["calls"]
        assert "--trusted-root" not in first
        assert second[second.index("--trusted-root") + 1] == str(harness["root"])

    def test_no_baked_root_means_no_second_attempt(self, harness: dict[str, Any]) -> None:
        """Without a baked root, trust-root trouble is simply a failure."""
        harness["answers"].append(_proc(1, "", "Error: tuf refresh failed: dial tcp"))
        with pytest.raises(signing.SignatureError, match="tuf refresh failed"):
            self._verify(harness, root="no_root")
        assert len(harness["calls"]) == 1

    def test_no_policy_or_no_cosign_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Neither condition is an excuse to run the image."""
        with pytest.raises(signing.SignatureError, match="no signing policy"):
            signing.verify_image("ghcr.io/other/thing:1", digest=DIGEST, policy=None, cosign="/x")
        monkeypatch.setattr(signing, "cosign_path", lambda: None)
        with pytest.raises(signing.SignatureError, match="cosign is not installed"):
            signing.verify_image("ghcr.io/medmcp/core:v1", digest=DIGEST, policy=POLICY)


class TestCallSites:
    """Where verification is wired in."""

    def test_helper_is_started_by_digest(self) -> None:
        """After verification the helper runs repo@digest, and hands both digests over."""
        plan = update.ApplyPlan(
            container_id="c",
            image_repo="ghcr.io/medmcp/core",
            current_tag="v0.2.3",
            project="medmcp",
            project_dir="/d",
            socket_source="/s",
            state_volume="v",
            env={},
            current_digest=OTHER,
        )
        args = update.updater_command(plan, "v0.3.0", DIGEST)
        assert args[-2:] == [f"ghcr.io/medmcp/core@{DIGEST}", "apply"]
        assert f"MEDMCP_UPDATE_DIGEST={DIGEST}" in args
        assert f"MEDMCP_UPDATE_PREVIOUS_DIGEST={OTHER}" in args
        # Without a verified digest (never the case on the update path) the tag is used.
        assert update.updater_command(plan, "v0.3.0")[-2] == "ghcr.io/medmcp/core:v0.3.0"

    @staticmethod
    def _present(image: str) -> bool:
        return True

    @staticmethod
    def _no_arch_check(image: str) -> None:
        return None

    def test_stack_install_verifies_a_policed_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stack image under a policy must verify before its label is read."""
        seen: list[str] = []

        def _policy(
            image: str, policies: dict[str, signing.Policy] | None = None
        ) -> signing.Policy:
            return POLICY

        def _verify(image: str, **kwargs: object) -> signing.Verified:
            seen.append(image)
            raise signing.SignatureError("nope")

        def _label(image: str) -> dict[str, object]:
            pytest.fail("label read before verification")

        monkeypatch.setattr(settings, "_image_present", self._present)
        monkeypatch.setattr(signing, "policy_for", _policy)
        monkeypatch.setattr(signing, "verify_image", _verify)
        monkeypatch.setattr(settings, "read_stack_label", _label)
        with pytest.raises(signing.SignatureError, match="nope"):
            settings.install_stack_image("ghcr.io/medmcp/neuro:main")
        assert seen == ["ghcr.io/medmcp/neuro:main"]

    def test_stack_install_says_so_when_unpoliced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An image outside the policy installs, with the fact reported."""
        lines: list[str] = []

        def _policy(image: str, policies: dict[str, signing.Policy] | None = None) -> None:
            return None

        def _verify(image: str, **kwargs: object) -> signing.Verified:
            pytest.fail("verified without a policy")

        def _label(image: str) -> dict[str, object]:
            return {"name": "!bad"}

        monkeypatch.setattr(settings, "_image_present", self._present)
        monkeypatch.setattr(signing, "policy_for", _policy)
        monkeypatch.setattr(signing, "verify_image", _verify)
        monkeypatch.setattr(settings, "check_image_arch", self._no_arch_check)
        monkeypatch.setattr(settings, "read_stack_label", _label)
        with pytest.raises(ValueError, match="invalid stack name"):
            settings.install_stack_image("registry.local/x/stack:1", lines.append)
        assert any("unverified" in line for line in lines)
