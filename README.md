![](https://capsule-render.vercel.app/api?type=waving&height=200&color=0:D22229,100:2B4FA3&text=MedMCP&reversal=false&fontSize=46&fontAlignY=28&desc=An%20Agentic%20Framework%20for%20Democratizing%20Medical%20Imaging%20Pipelines&descSize=24&descAlignY=55&fontColor=FFFFFF)

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
- [Workspace UI](#workspace-ui)
- [Imaging Stacks](#imaging-stacks)
- [Provenance & Reusable Workflows](#provenance--reusable-workflows)
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
just ui           # launches the chat web UI at http://localhost:8000
```

MedMCP also ships an experimental **workspace UI** — a three-panel interface
(file explorer, medical-image viewer, chat) for working with imaging data
alongside the agent. See [Workspace UI](#workspace-ui).

```bash
just workspace-build  # build the workspace frontend (one-time / after updates; needs node + npm)
just workspace        # launches the workspace UI at http://localhost:8100
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

1. **Web UI** — the user-facing frontend. Two are available on top of the same backend: the **Chainlit chat UI** (`src/medmcp/app.py`, `http://localhost:8000`) and the **workspace UI** (`src/medmcp/server.py` + `frontend/`, `http://localhost:8100`; see [Workspace UI](#workspace-ui)).
2. **vibe-acp subprocess** — an agent orchestrator that receives JSON-RPC 2.0 messages from the UI, manages tool execution, and communicates with the local LLM.
3. **Ollama** — serves the local Gemma 4 model (`Modelfile.gemma4`) with 128k context, top-p/top-k sampling, and repeat-penalty guards. The model runs at temperature 0.3 for deterministic instruction-following, configured in `Modelfile.gemma4`.

The UI spawns a single vibe-acp subprocess and demultiplexes sessions over it. Every tool call (bash, file writes, web fetches) is gated by an interactive Approve/Reject prompt — the user must explicitly approve each action before any side effect occurs.

---

## Workspace UI

In addition to the chat-only Chainlit interface, MedMCP ships an experimental **workspace UI** — a four-panel layout built for working *with* imaging data, not just chatting about it:

- **File explorer** (top-left) — browse, rename, move, delete, and upload files in your workspace.
- **Image viewer** (top-right) — view medical images directly in the browser: NIfTI/NRRD/MGZ volumes render with multiplanar slices and a 3D view (scroll to move through slices), and PDFs, images, and text files open inline. **Drag a segmentation from the explorer onto an image** to overlay it — each label is drawn in a distinct color over the anatomy, with an adjustable opacity, and the background stays transparent in both the slices and the 3D render.
- **Workflows** (bottom-left) — save the current chat as a reusable workflow (the bookmark button distills it into a recipe + skill), then review, rename, refine, promote, or delete it. **Run** replays a saved recipe deterministically on new inputs — no LLM involved: fill in the inputs (drag files in from the explorer), review the resolved steps, and watch each step stream its result.
- **Chat** (bottom-right) — the same agent as the Chainlit UI, with streamed responses, per-tool-call approval prompts (with plain-language explanations and risk tags), and a settings drawer for the stack/workflow/feature toggles.

The workspace UI is served by a separate local server on `http://localhost:8100` and shares all of MedMCP's machinery below the interface (agent loop, tool approval, provenance) with the Chainlit UI — the two are parallel frontends.

```bash
just workspace-build  # build the React frontend (requires node + npm; one-time / after updates)
just workspace        # launch the workspace UI at http://localhost:8100
```

The explorer and the agent's working directory are rooted at the repository's `data/` directory by default (created on first launch); point them at another folder with `MEDMCP_WORKSPACE=/path/to/data just workspace`. Like the chat UI, the server binds to localhost only and gates every tool call behind explicit approval.

> [!NOTE]
> The workspace UI is under active development. Chat history/resume is not yet available there — use the [Chainlit UI](#architecture) (`just ui`) for that.

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

## Provenance & Reusable Workflows

Beyond running one-off analyses, MedMCP records what it does and lets you turn a successful session into a repeatable pipeline.

- **Provenance** — every session is recorded to `.vibe/provenance/<session>/`: the environment (git commit, active stacks + versions, model), one normalized entry per tool call (resolved arguments, structured outputs, permission decision, duration), a mirror of the approval log, and a documentation-grade `report.md`. Recording is on by default and can be turned off per session with the **Record provenance** switch in settings. Deleting a chat in the UI removes its logs; you never need the CLI for cleanup.

- **Save a workflow** — the **Save workflow** button in the message box distills the current chat into a reusable workflow. MedMCP keeps only the steps that mattered (dropping exploratory, failed, and rejected tool calls) and lifts concrete file paths into named inputs, producing a human-readable `SKILL.md` and a machine-readable `recipe.yaml`. Review, rename, refine, and **Promote** it to keep it as a permanent skill.

- **Replay on new data (no LLM)** — the **Run** button replays a saved workflow deterministically: it asks for the new inputs (each labelled with what it is, e.g. *the input_path for `medmcp-neuro:skull_strip`*), shows you the exact steps it will run, and on confirmation calls the same tools in the same order — no model reasoning involved. Step outputs are fed forward automatically, and a failed step aborts the run.

Personal workflows can be toggled on/off individually, or disabled entirely with the **Personal workflows** master switch in settings. The same operations are available from the `medmcp` CLI (`medmcp list`, `report`, `distill`, `promote`, …) and the matching `just` recipes for scripted use.

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
