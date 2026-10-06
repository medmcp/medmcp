# Security Policy

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email **julian.mcginnis@tum.de** with:

- A description of the issue
- Steps to reproduce (anonymized — never include real patient data)
- Affected version(s) and commit SHA if known
- Any suggested mitigation

You should receive an acknowledgement within a few business days. We will work with you on disclosure timing and credit.

## Scope

MedMCP and its ecosystem packages are **research software**. They are not medical devices and **must not be used for clinical decision-making**. Security reports should focus on:

- Code-execution vulnerabilities in parsing/processing pipelines (DICOM, NIfTI, etc.)
- Path traversal, SSRF, or injection in tool handlers
- Dependency vulnerabilities we haven't picked up
- Leaking of user data through logs or telemetry

Out of scope:

- Issues that only apply to clinical use (MedMCP is not approved for it — see the warning in the README)
- Denial of service on deliberately malformed inputs where the fix is "don't feed it malformed inputs"

## Network posture

The workspace server listens on loopback and has **no authentication**. Do not
expose the port. In a container it binds `0.0.0.0`, and compose publishes it only
to the host's loopback (`127.0.0.1:8100:8100`) — keep it that way.

Binding loopback does not by itself keep a browser out: any page you visit can
send requests to `127.0.0.1`, and a WebSocket upgrade is exempt from the
same-origin policy entirely. The server therefore refuses requests and
connections whose `Origin` is not the workspace's own page, and requests whose
`Host` is not a loopback address (which is what stops DNS rebinding, where a
hostname the attacker controls resolves to `127.0.0.1`).

If you front MedMCP with a reverse proxy or reach it by a hostname, list it:

```
MEDMCP_ALLOWED_HOSTS=medmcp.example.internal
MEDMCP_ALLOWED_ORIGINS=https://medmcp.example.internal
```

Both default to empty — unlisted means refused. Neither replaces authentication,
which the workspace does not have; they keep a browser from acting as one.

## How the workspace protects your data

MedMCP assumes the model can be steered by prompt injection (for example by text
pasted from an untrusted document), so its safety model is built around explicit
user control:

- **Nothing is changed or sent without your approval.** Writing a file, editing
  one, fetching a URL, or running a command with side effects each require an
  explicit click. There is no "always allow" and no session-wide approval; each
  call is approved on its own. Read-only shell commands (`ls`, `cat`, `grep`, …)
  run without a prompt *inside your workspace*; pointed outside it, they ask
  first, and `find -exec` is treated as execution, not reading.
- **No data egress by default.** `web_search` is disabled and `web_fetch`
  requires approval.
- **Isolated tool stacks.** Stack containers run with `--network none`, all
  capabilities dropped, and no privilege escalation. They bake their models at
  build time, so a tool call cannot reach the network even if the agent is
  steered into making one. A stack that genuinely needs egress must declare it
  in its image label, and installing it asks for consent.

## What contacts the internet

- **The release check**: one anonymous request to github.com a day that carries
  nothing about the workspace, the machine, or its use. Switch it off in
  Settings → Advanced, or for a whole deployment with `MEDMCP_UPDATE_CHECK=0`
  (which also disables manual checks). `MEDMCP_UPDATE_URL` reads releases from a
  mirror instead.
- **Downloads you start**: installing a tool stack, updating MedMCP, or choosing
  a local model you have not downloaded yet. None of them sends workspace data.
- **Outside services you switch on** (Settings → Advanced, both off by default,
  each behind a consent dialog and a standing banner while in use):
  - *External MCP servers* receive whatever the agent passes to their tools;
    each call still needs your approval.
  - *A cloud model* receives the whole chat on every turn: what you type, the
    files the agent reads, and tool results. No approval prompt covers this. Its
    API key is stored on the machine and never written to the agent's config.

Nothing else in MedMCP contacts the internet.

## Signed images

Every published image and the compose artifact are signed in CI with
[Sigstore](https://www.sigstore.dev/) keyless signing: the signature is bound to
this repository's release workflow, and there is no signing key to protect. The
workspace verifies a signature before it applies an update or installs a stack,
and refuses anything that does not verify. To check a release yourself, replace
`<tag>` with a release tag such as `v0.3.0`:

```bash
cosign verify ghcr.io/medmcp/core:<tag> \
  --certificate-identity-regexp '^https://github\.com/medmcp/medmcp/\.github/workflows/release\.yml@refs/tags/v' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

When mirroring images into a private registry, copy the signatures along with
them (`cosign copy` does; a plain image copy does not).

## Patient data

If a security issue only reproduces with data you cannot share, describe the file characteristics (modality, vendor, transfer syntax, dimensions) and we will reproduce with synthetic data. **Never** attach PHI to emails, logs, or issues.
