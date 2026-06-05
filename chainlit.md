# Welcome to MedMCP

MedMCP lets you run validated medical-imaging tools through plain language — no command line, no Python environments, no library APIs. Everything runs **locally**: your data never leaves this machine.

> ⚠️ MedMCP is under active development and **not licensed for clinical use**.

## Getting started

1. **Just ask.** Describe what you want in natural language, e.g. *"Skull-strip the T1 in `data/patient01/t1.nii.gz` and register it to MNI."*
2. **Approve each step.** Before any tool runs (file writes, conversions, processing), MedMCP shows you what it will do and waits for your **Approve / Reject**. Nothing happens without your click.
3. **Follow along.** Each tool call is summarized with a plain-language explanation and the result.

## Choosing your tools

Open the **settings** (gear icon) to:

- **Toggle imaging stacks** — enable only the domains you need (e.g. `medmcp-neuro`). Changes apply to your next message.
- **Explain tool calls** — add a short plain-language explanation to each step.
- **Record provenance** — keep a reproducible log of the session.

## Saving & reusing workflows

Found a sequence that works? Turn it into a repeatable pipeline:

- **Save workflow** (button in the message box) — distills the current chat into a reusable workflow, keeping just the steps that mattered.
- **Manage workflows** — review, rename, refine, or promote your saved workflows.
- **Run** — replay a saved workflow on **new data**, deterministically and **without the LLM**. MedMCP asks for the new inputs (each labelled with what it is), shows you the exact steps, and runs them on your confirmation.

## Your data stays yours

Every session is recorded locally under `.vibe/provenance/` (toggle it off in settings if you prefer). Deleting a chat removes its logs. Web search is disabled and the app is reachable only from this machine.

Happy analyzing! 🧠
