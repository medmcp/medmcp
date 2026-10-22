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

## Threat model

What MedMCP protects is the imaging data in the workspace and the operator's
machine. It assumes:

- **The model can be steered.** Anything it reads — a document, a DICOM tag, a
  tool's output, a shared workflow — may carry instructions. So the model gets
  no authority of its own: every action waits for a person, and the agent runs
  in a container that can reach the workspace and nothing else.
- **Other accounts and processes on the machine are not trusted.** The port is
  loopback-only and the API needs a token.
- **Web pages are not trusted.** Requests from another origin are refused.
- **Stack images are trusted once installed.** They run sandboxed (no network,
  no capabilities, the workspace only); a stack that needs the network says so
  and asks for consent at install.

Trust boundaries in a `docker compose` install: the browser ↔ the workspace
server (token, origin check); the server ↔ the agent container (stdio, a
read-only config, the broker that only runs approved stack calls, the egress
gateway that holds every credential); the server ↔ Docker (the guard); the
server ↔ stacks (sandboxed containers).

Not covered: a person approving a harmful call (the dialog shows the command,
rule-based risk labels and every path, but the decision is theirs); a
compromised stack image or model file; and a host-native install, where the
agent runs with the operator's rights.

## Network posture

The workspace server listens on loopback. Do not expose the port. In a container
it binds `0.0.0.0`, and compose publishes it only to the host's loopback
(`127.0.0.1:8100:8100`) — keep it that way.

Loopback is shared by every account and process on the machine, so every API
request and WebSocket also needs the workspace's **access token**. A browser
receives it once, from the link the server logs at start and `medmcp url` prints
(`docker exec medmcp-medmcp-1 medmcp url`), and keeps it as an `HttpOnly`,
`SameSite=Strict` cookie; a script sends `Authorization: Bearer <token>`. The
token is generated once and stored `0600` with the workspace state;
`MEDMCP_UI_TOKEN` sets it instead.

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

Both default to empty — unlisted means refused. The token still applies.

## How the workspace protects your data

MedMCP assumes the model can be steered by prompt injection (for example by text
pasted from an untrusted document), so its safety model is built around explicit
user control:

- **Nothing is changed or sent without your approval.** Writing a file, editing
  one, fetching a URL, running a command or a tool each require an explicit
  click. The one approval that lasts beyond a call is "Allow `ls` for this chat"
  for a read-only command (`ls`, `cat`, `grep`, …) in your workspace; it ends
  with the chat. Reading a file inside the workspace needs no approval; outside
  it, it asks first.
- **What approving a command grants.** A shell command runs with the agent's
  rights. In a `docker compose` install the agent has its own container: the
  workspace, its own configuration read-only, and nothing else — no Docker
  socket, no credentials, no capabilities beyond writing your files, and no
  network: the model, a cloud model and external MCP servers are reached
  through a gateway in the core, which holds their keys and forwards only to
  what you configured. A stack tool runs only when you approved that exact call. Run
  host-native (`just workspace`), the agent is you: an approved command can do
  whatever your account can, so read each one before you allow it.
- **The workspace server cannot reach the host either.** It starts containers
  through a guard that holds the Docker socket and refuses privileged
  containers, host namespaces, added capabilities and mounts other than the
  workspace and MedMCP's own volumes.
- **The agent cannot loosen its own rules.** Its configuration is read-only in
  the container install, and every new chat restores the approval settings,
  hooks and stack definitions to their pinned values (audit-logged when it had
  to).
- **No data egress by default.** `web_search` is disabled and `web_fetch`
  requires approval.
- **Isolated tool stacks.** Stack containers run with `--network none`, all
  capabilities dropped but writing your files, and no privilege escalation; a
  stack definition that adds a mount, a host namespace or `--privileged` is
  refused. They bake their models at
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
