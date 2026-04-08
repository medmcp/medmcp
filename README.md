# MedMCP: An Agentic Framework for Democratizing Medical Imaging Pipelines

MedMCP is an open, community-driven agentic framework that exposes validated medical imaging tools through a natural-language interface.
It is designed to enable clinicians, radiologists, and domain researchers to apply state-of-the-art image analysis methods without requiring expertise in command-line interfaces, Python environment management, or library-specific APIs.

The framework enforces a strict separation between **orchestration** and **execution**: a locally served language model plans and sequences operations by invoking pre-registered tools, while all computational work is delegated to validated, tested implementations.
MedMCP runs entirely on-premise and is designed to meet the data governance and privacy requirements of clinical and translational research environments.

> [!WARNING]
> MedMCP is under active development and not licensed for clinical use!

---

## Vision

MedMCP aims to become a **community-driven framework of tested medical imaging skills** for local AI agents. The core idea:

- **Accessibility** - MedMCP exists to make validated imaging tools reachable, not to produce new ones. Every architectural decision is evaluated against one question: *"Does this reduce the barrier between a working method and the practitioner who needs it?"*
- **Local-First** - MedMCP runs entirely on-premise: LLM inference is handled by a locally served model (in our case via Ollama), and no imaging data, patient metadata, or intermediate results leave the institution's infrastructure. This is a hard architectural constraint, not an optional feature.
- **Community Driven** - MedMCP is designed to grow through community contributions: a shared schema for tool and skill metadata, CI-based testing, and a central registry for discovery ensure that each new contribution is immediately available to all users. Accessibility scales with the community, not with any single team's capacity. [Learn how to contibute!](CONTRIBUTING.md)

---

## Table of Contents

- [Installation](#Installation)
- [Contributing](#Contributing)

---

## Installation

### Prerequisites

Linux, Python ≥3.12, [Ollama](https://ollama.com/), and [just](https://github.com/casey/just).

### Quickstart

```bash
just setup        # installs uv, ollama, syncs Python deps
just pull_model   # builds the local Devstral model (~15 GB, one-time)
just ui           # launches the web UI at http://localhost:8000
```

Every tool call in the UI (bash, file writes, web fetches) requires an interactive Approve / Reject click. See the `SECURITY MODEL` section in `src/medmcp/app.py` for the full threat model.

---

## Contributing
