# MedMCP: An Agentic Framework for Democratizing Medical Imaging Pipelines

MedMCP is an open, community-driven agentic framework that exposes validated medical imaging tools through a natural-language interface.
It is designed to enable clinicians, radiologists, and domain researchers to apply state-of-the-art image analysis methods without requiring expertise in command-line interfaces, Python environment management, or library-specific APIs.

The framework enforces a strict separation between **orchestration** and **execution**: a locally served language model plans and sequences operations by invoking pre-registered tools, while all computational work is delegated to validated, tested implementations.
MedMCP runs entirely on-premise and is designed to meet the data governance and privacy requirements of clinical and translational research environments.

> [!WARNING]
> MedMCP is under active development and not licensed for clinical use!

---

## Table of Contents

- [Installation](#installation)
- [Vision](#vision)
- [Architecture](#architecture)
- [Imaging Stacks](#imaging-stacks)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

### Prerequisites

- **OS:** Linux (tested on Ubuntu)
- **Hardware:** A GPU with at least 24 GB VRAM is recommended for running the local Gemma 4 26B model (~18 GB loaded). CPU-only inference works but will be slow.
- **Tools:** [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (we recommend installing `rust-just` with `uv tool install rust-just`)

### Quickstart

The fastest way to get started is a single command that handles the full setup
(uv, Ollama, Python deps, model pull) and then launches the UI:

```bash
just medmcp       # install everything, pull model, start Ollama, launch UI
```

If you prefer to run each step separately:

```bash
just setup        # installs uv, ollama, and syncs Python deps
just pull-model   # pulls Gemma 4 26B and builds gemma4-medmcp (~18 GB, one-time)
just serve-ollama # start the Ollama server (foreground, Ctrl-C to stop)
just ui           # launches the web UI at http://localhost:8000
```

---

## Vision

MedMCP aims to become a **community-driven framework of tested medical imaging skills** for local AI agents. The core idea:

- **Accessibility** — MedMCP exists to make validated imaging tools reachable, not to produce new ones. Every architectural decision is evaluated against one question: *"Does this reduce the barrier between a working method and the practitioner who needs it?"*
- **Local-First** — MedMCP runs entirely on-premise: LLM inference is handled by a locally served model (in our case via Ollama), and no imaging data, patient metadata, or intermediate results leave the institution's infrastructure. This is a hard architectural constraint, not an optional feature.
- **Community Driven** — MedMCP is designed to grow through community contributions: a shared schema for tool and skill metadata, CI-based testing, and a central registry for discovery ensure that each new contribution is immediately available to all users. Accessibility scales with the community, not with any single team's capacity. [Learn how to contribute!](CONTRIBUTING.md)

---

## Architecture

MedMCP is built as a three-layer stack:

1. **Chainlit web UI** (`src/medmcp/app.py`) — the user-facing chat interface served at `http://localhost:8000`.
2. **vibe-acp subprocess** — an agent orchestrator that receives JSON-RPC 2.0 messages from the UI, manages tool execution, and communicates with the local LLM.
3. **Ollama** — serves the local Gemma 4 model (`Modelfile.gemma4`) with 128k context, top-p/top-k sampling, and repeat-penalty guards. The model runs at temperature 0.3 for deterministic instruction-following, configured in `Modelfile.gemma4`.

The UI spawns a single vibe-acp subprocess and demultiplexes sessions over it. Every tool call (bash, file writes, web fetches) is gated by an interactive Approve/Reject prompt — the user must explicitly approve each action before any side effect occurs.

---

## Imaging Stacks

MedMCP's imaging capabilities are provided by optional **stack** packages. Each stack bundles domain-specific tools and their foundation dependencies into a single MCP server.

```
┌─ medmcp (core) ──────────────────────────────────┐
│  Chainlit UI, agent loop, prompts, config        │
└──────────────────────────────────────────────────┘
           │ discovers via uv tool environments
           ▼
┌─ stack layer (domain-specific) ──────────────────┐
│  medmcp-neuro       brain extraction, seg, reg   │
│  medmcp-cardiac     (planned)                    │
│  medmcp-microscopy  (planned)                    │
└──────────────────────────────────────────────────┘
           │ depends on
           ▼
┌─ foundation layer (shared I/O) ──────────────────┐
│  medmcp-dicom       DICOM inspection + conversion│
└──────────────────────────────────────────────────┘
```

Each stack runs in its own isolated uv tool environment. Install a stack with:

```bash
just install-stack "git+ssh://git@github.com/medmcp/medmcp-neuro.git"
```

Once installed, the stack is auto-discovered via its `[medmcp.stacks]` entry point and appears as a toggle in the UI's **ChatSettings panel** — no manual edits to `.vibe/config.toml` needed. Toggle changes take effect on the next conversation. Restart the UI after installing or removing a stack.

---

## Security

MedMCP's security model is designed around the assumption that the local model may be steered by prompt injection (e.g. content pasted from untrusted documents). Key constraints:

- **No auto-approval** — every tool call requires an explicit user click. There is no auto-approval path.
- **Localhost only** — the Chainlit server binds to localhost. Do not expose port 8000 over a network without adding real authentication.
- **No data exfiltration** — `web_search` is disabled; `web_fetch` requires approval. No data leaves the institution's infrastructure by default.

For vulnerability reporting, see [SECURITY.md](SECURITY.md).

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on forking, setting up a development environment, running checks, and submitting pull requests.

---

## License

MedMCP is released under the [Apache License 2.0](LICENSE).
