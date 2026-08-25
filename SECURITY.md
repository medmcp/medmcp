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

## Patient data

If a security issue only reproduces with data you cannot share, describe the file characteristics (modality, vendor, transfer syntax, dimensions) and we will reproduce with synthetic data. **Never** attach PHI to emails, logs, or issues.
