"""Image signature verification (Sigstore keyless, via the cosign CLI).

Every image this project publishes is signed in CI without a private key:
the GitHub Actions run signs with its OIDC identity, and the signature is
bound to the exact repository, workflow file and ref that produced it. There
is nothing to store, leak or rotate. What makes that meaningful is the
*policy* applied at verification time — a signature from "someone on
Sigstore" is worthless; one from ``medmcp/medmcp``'s ``release.yml`` on a
``v*`` tag is the guarantee.

The policy is ``signing-policy.json`` next to this package: which image
repositories must be signed, and by which workflow identity. It ships inside
the core image, so each release carries the policy for the next one. Trust
starts at install time and is carried forward from there.

Two places verify, both failing closed:

- the update path, before the helper that swaps the install is started;
- stack install, before the image's label is even read.

Offline installs: keyless verification normally needs the Sigstore trust root
(fetched over TUF) and the transparency log. The signature bundle stored next
to the image already carries the log inclusion proof, and the trust root is
baked into the image at build time, so verification needs only the registry
the image came from. The live root is tried first (it may know a CA or log
shard rotated in since the image was built) and the baked one is the fallback.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from medmcp.acp import PROJECT_ROOT, JsonDict

log: logging.Logger = logging.getLogger("medmcp.signing")

POLICY_PATH: Path = Path(PROJECT_ROOT) / "signing-policy.json"
TRUSTED_ROOT_PATH: Path = Path(PROJECT_ROOT) / "sigstore" / "trusted_root.json"
VERIFY_TIMEOUT_SEC: float = 120.0

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
# Errors that mean "could not reach the Sigstore infrastructure", after which
# the baked trust root is worth a second attempt. Anything else is a verdict.
_TRUST_ROOT_ERROR_MARKERS: tuple[str, ...] = (
    "tuf",
    "trusted root",
    "trusted material",
    "dial tcp",
    "no such host",
    "proxyconnect",
)


class SignatureError(Exception):
    """The image could not be verified; ``str()`` says why, for the operator."""


@dataclass(frozen=True)
class Policy:
    """Who must have signed an image repository."""

    identity_regexp: str
    issuer: str


@dataclass(frozen=True)
class Verified:
    """A verified image, pinned to the digest the signature covers."""

    repo: str
    digest: str

    @property
    def ref(self) -> str:
        """``repo@sha256:…`` — the reference to run, immune to a moved tag."""
        return f"{self.repo}@{self.digest}"


def split_reference(image: str) -> tuple[str, str | None, str | None]:
    """``repo[:tag][@digest]`` → ``(repo, tag, digest)``.

    The registry host may carry a port (``host:5000/x``), which is the only
    other colon a reference can contain; a tag colon always follows the last
    slash.
    """
    ref, _, digest = image.partition("@")
    head, slash, last = ref.rpartition("/")
    if ":" in last:
        name, _, tag = last.partition(":")
        repo = f"{head}{slash}{name}"
    else:
        repo, tag = ref, ""
    return repo, tag or None, digest or None


def load_policies(path: Path | None = None) -> dict[str, Policy]:
    """The bundled policy file as ``{repo: Policy}``; empty if absent or unreadable."""
    target = path or POLICY_PATH
    try:
        doc = cast("JsonDict", json.loads(target.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("no usable signing policy at %s: %s", target, exc)
        return {}
    issuer = str(doc.get("issuer") or "")
    out: dict[str, Policy] = {}
    for entry in cast("list[JsonDict]", doc.get("policies") or []):
        regexp = str(entry.get("identity_regexp") or "")
        if not regexp:
            continue
        try:
            re.compile(regexp)
        except re.error as exc:
            log.warning("skipping signing policy with a bad regexp %r: %s", regexp, exc)
            continue
        policy = Policy(identity_regexp=regexp, issuer=str(entry.get("issuer") or issuer))
        for repo in cast("list[str]", entry.get("images") or []):
            out[str(repo)] = policy
    return out


def policy_for(image: str, policies: dict[str, Policy] | None = None) -> Policy | None:
    """The policy covering *image*'s repository, if any."""
    repo, _, _ = split_reference(image)
    table = policies if policies is not None else load_policies()
    return table.get(repo)


def cosign_path() -> str | None:
    """The cosign binary, if installed."""
    return shutil.which("cosign")


def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def image_digest(image: str) -> str:
    """The registry digest a locally pulled *image* was fetched under.

    For a multi-arch tag this is the index digest — the digest CI signs — not
    the platform manifest's. Raises :class:`SignatureError` if docker does not
    know the image or it carries no repository digest (a local build).
    """
    repo, _, digest = split_reference(image)
    if digest:
        return digest
    try:
        proc = _run(["docker", "image", "inspect", image], timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SignatureError(f"could not inspect {image}: {exc}") from exc
    if proc.returncode != 0:
        raise SignatureError(f"could not inspect {image}: {proc.stderr.strip()}")
    try:
        items = cast("list[JsonDict]", json.loads(proc.stdout))
    except json.JSONDecodeError as exc:
        raise SignatureError(f"unreadable inspect output for {image}") from exc
    repo_digests = cast("list[str]", (items[0] if items else {}).get("RepoDigests") or [])
    for entry in repo_digests:
        entry_repo, _, entry_digest = entry.partition("@")
        if entry_repo == repo and _DIGEST_RE.fullmatch(entry_digest):
            return entry_digest
    raise SignatureError(f"{image} carries no registry digest for {repo}; was it pulled?")


def _parse_verified_digests(stdout: str) -> set[str]:
    try:
        items = cast("list[JsonDict]", json.loads(stdout or "[]"))
    except json.JSONDecodeError:
        return set()
    found: set[str] = set()
    for item in items:
        critical = cast("JsonDict", item.get("critical") or {})
        image = cast("JsonDict", critical.get("image") or {})
        digest = str(image.get("docker-manifest-digest") or "")
        if digest:
            found.add(digest)
    return found


def _looks_like_trust_root_trouble(stderr: str) -> bool:
    text = stderr.lower()
    return any(marker in text for marker in _TRUST_ROOT_ERROR_MARKERS)


def verify_image(
    image: str,
    *,
    digest: str | None = None,
    policy: Policy | None = None,
    cosign: str | None = None,
    trusted_root: Path | None = None,
) -> Verified:
    """Verify that *image* is signed by the identity its policy names.

    Verification is by digest, never by tag: *digest* defaults to what docker
    recorded when the image was pulled. Returns the pinned reference to run.
    Raises :class:`SignatureError` — with the reason — on a missing policy,
    a missing cosign, a failed verification, or a signature that covers a
    different digest than the one pulled.
    """
    repo, _, _ = split_reference(image)
    policy = policy or policy_for(image)
    if policy is None:
        raise SignatureError(f"no signing policy covers {repo}; refusing to run it unverified.")
    binary = cosign or cosign_path()
    if binary is None:
        raise SignatureError("cosign is not installed here, so the image cannot be verified.")
    digest = digest or image_digest(image)
    ref = f"{repo}@{digest}"
    base = [
        binary,
        "verify",
        ref,
        "--certificate-identity-regexp",
        policy.identity_regexp,
        "--certificate-oidc-issuer",
        policy.issuer,
        "-o",
        "json",
    ]
    root = trusted_root if trusted_root is not None else TRUSTED_ROOT_PATH
    attempts: list[list[str]] = [base]
    if root.exists():
        attempts.append([*base, "--trusted-root", str(root)])

    last_error = ""
    for index, args in enumerate(attempts):
        try:
            proc = _run(args, timeout=VERIFY_TIMEOUT_SEC)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SignatureError(f"cosign could not run: {exc}") from exc
        if proc.returncode == 0:
            verified = _parse_verified_digests(proc.stdout)
            if digest not in verified:
                raise SignatureError(
                    f"the signature on {ref} covers {sorted(verified) or 'nothing'}, "
                    f"not the pulled digest."
                )
            log.info("verified %s (%s)", ref, "baked trust root" if index else "live trust root")
            return Verified(repo=repo, digest=digest)
        last_error = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown"
        if index == 0 and len(attempts) > 1 and _looks_like_trust_root_trouble(proc.stderr):
            log.info("live trust root unavailable (%s); retrying with the baked one", last_error)
            continue
        break
    raise SignatureError(f"signature verification failed for {ref}: {last_error}")


def describe(verified: Verified) -> str:
    """One line for a progress log."""
    return f"Signature verified: {verified.ref}"


__all__: list[str] = [
    "Policy",
    "SignatureError",
    "Verified",
    "cosign_path",
    "describe",
    "image_digest",
    "load_policies",
    "policy_for",
    "split_reference",
    "verify_image",
]
