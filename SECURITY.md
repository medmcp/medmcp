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

medmcp and its ecosystem packages are **research software**. They are not medical devices and **must not be used for clinical decision-making**. Security reports should focus on:

- Code-execution vulnerabilities in parsing/processing pipelines (DICOM, NIfTI, etc.)
- Path traversal, SSRF, or injection in tool handlers
- Dependency vulnerabilities we haven't picked up
- Leaking of user data through logs or telemetry

Out of scope:

- Issues that only apply to clinical use (medmcp is not approved for it — see the warning in the README)
- Denial of service on deliberately malformed inputs where the fix is "don't feed it malformed inputs"

## Patient data

If a security issue only reproduces with data you cannot share, describe the file characteristics (modality, vendor, transfer syntax, dimensions) and we will reproduce with synthetic data. **Never** attach PHI to emails, logs, or issues.
