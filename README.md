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
- **Hardware:** an NVIDIA GPU; ≥ 24 GB VRAM recommended for the local Gemma 4 26B model (~18 GB loaded). CPU-only inference works but is slow.
- **To run with Docker (recommended):** Docker (rootless is supported) + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/) with a CDI spec (`nvidia-ctk cdi generate`); host driver ≥ R570 (CUDA 12.8).
- **To run natively:** [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (we recommend `uv tool install rust-just`).

### Run with Docker (recommended)

The recommended way to run MedMCP is with **Docker** — a reproducible, GPU-ready
setup: a CPU-only **core** (workspace UI + agent) plus a bundled **Ollama** GPU
service, with imaging stacks launched as their own GPU containers. (See
[Prerequisites](#prerequisites) for the Docker/NVIDIA requirements.)

```bash
just compose-up    # build images, ensure the gemma4-medmcp model, start core + Ollama
# open http://localhost:8100  —  tear down with: just compose-down
```

By default the workspace is `./data` and models are reused from the host's
`~/.ollama` (override with `MEDMCP_WORKSPACE` / `OLLAMA_MODELS_DIR`; a fresh
named-volume deploy pulls the ~18 GB model on first run). Imaging stacks are
declared as container manifests under `stacks.d/` — see [Imaging Stacks](#imaging-stacks).

### Run natively with `just` (alternative)

Prefer a host install? A single command handles the full setup (uv, Ollama, Python
deps, model pull) and launches the UI:

```bash
just medmcp       # install everything, pull model, start Ollama, launch UI
```

If you prefer to run each step separately:

```bash
just setup        # installs uv, ollama, and syncs Python deps
just pull-model   # pulls Gemma 4 26B and builds gemma4-medmcp (~18 GB, one-time)
just serve-ollama # start the Ollama server (foreground, Ctrl-C to stop)
just workspace-build  # build the workspace frontend (one-time; needs node + npm)
just workspace    # launches the workspace UI at http://localhost:8100
```

MedMCP's primary interface is the **workspace UI** — a four-panel interface
(file explorer, medical-image viewer, workflows, chat) for working with imaging
data alongside the agent. See [Workspace UI](#workspace-ui).

> **Developing MedMCP?** The recommended setup is the **dev container** (works with
> PyCharm and VS Code) — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Vision

MedMCP aims to become a **community-driven framework of tested medical imaging skills** for local AI agents. The core idea:

- **Accessibility** — MedMCP exists to make validated imaging tools reachable, not to produce new ones. Every architectural decision is evaluated against one question: *"Does this reduce the barrier between a working method and the practitioner who needs it?"*
- **Local-First** — MedMCP runs entirely on-premise: LLM inference is handled by a locally served model (in our case via Ollama), and no imaging data, patient metadata, or intermediate results leave the institution's infrastructure. This is a hard architectural constraint, not an optional feature.
- **Community Driven** — MedMCP is designed to grow through community contributions: a shared schema for tool and skill metadata, CI-based testing, and a central registry for discovery ensure that each new contribution is immediately available to all users. Accessibility scales with the community, not with any single team's capacity. [Learn how to contribute!](CONTRIBUTING.md)

---

## Architecture

MedMCP is built as a three-layer stack:

1. **Web UI** — the user-facing frontend: the **workspace UI** (`src/medmcp/server.py` + `frontend/`, `http://localhost:8100`; see [Workspace UI](#workspace-ui)).
2. **vibe-acp subprocess** — an agent orchestrator that receives JSON-RPC 2.0 messages from the UI, manages tool execution, and communicates with the local LLM.
3. **Ollama** — serves the local Gemma 4 model (`Modelfile.gemma4`) with 128k context, top-p/top-k sampling, and repeat-penalty guards. The model runs at temperature 0.3 for deterministic instruction-following, configured in `Modelfile.gemma4`.

The UI spawns a single vibe-acp subprocess and demultiplexes sessions over it. Every tool call (bash, file writes, web fetches) is gated by an interactive Approve/Reject prompt — the user must explicitly approve each action before any side effect occurs.

MedMCP runs **host-native** (the `just` recipes above) or **fully containerized** for deployment: a CPU-only core image plus a bundled Ollama GPU service, with imaging stacks launched as their own GPU containers. The same on-premise, no-egress, per-tool-approval posture holds in both modes. See [Run with Docker](#run-with-docker-recommended).

---

## Workspace UI

The **workspace UI** is MedMCP's primary interface — a four-panel layout built for working *with* imaging data, not just chatting about it:

- **File explorer** (top-left) — browse, rename, move, delete, and upload files in your workspace.
- **Image viewer** (top-right) — view medical images directly in the browser: NIfTI/NRRD/MGZ volumes render with multiplanar slices and a 3D view (scroll to move through slices), and PDFs, images, and text files open inline. **Drag a segmentation from the explorer onto an image** to overlay it — each label is drawn in a distinct color over the anatomy, with an adjustable opacity, and the background stays transparent in both the slices and the 3D render.
- **Workflows** (bottom-left) — save the current chat as a reusable workflow (the bookmark button distills it into a recipe + skill), then review, rename, refine, promote, or delete it. **Run** replays a saved recipe deterministically on new inputs — no LLM involved: fill in the inputs (drag files in from the explorer), review the resolved steps, and watch each step stream its result.
- **Chat** (bottom-right) — the MedMCP agent, with streamed responses, per-tool-call approval prompts (with plain-language explanations and risk tags), and a settings drawer for the stack/workflow/feature toggles. The agent knows which file is open in the viewer, so "this image" means what you're looking at. Conversations persist: a refresh resumes your last session, and a **Chats menu** lists, renames, archives, or deletes this workspace's past sessions.

The workspace UI is served by a local server on `http://localhost:8100`.

```bash
just workspace-build  # build the React frontend (requires node + npm; one-time / after updates)
just workspace        # launch the workspace UI at http://localhost:8100
```

The explorer and the agent's working directory are rooted at the repository's `data/` directory by default (created on first launch); point them at another folder with `MEDMCP_WORKSPACE=/path/to/data just workspace`. The server binds to localhost only and gates every tool call behind explicit approval.

---

## Imaging Stacks

MedMCP's imaging capabilities are provided by optional **stack** packages. Each stack bundles domain-specific tools and their foundation dependencies into a single MCP server.

```
┌─ medmcp (core) ──────────────────────────────────┐
│  workspace UI, agent loop, prompts, config       │
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

**Containerized stacks (deployment).** A stack can instead ship as a container image and be declared in a `stacks.d/<name>.toml` manifest (`command = "docker"`, `args = [...]`); the core then launches it over stdio with `docker run -i` (GPU stacks add `--device nvidia.com/gpu=all`). This is a second discovery source alongside `uv tool` installs — a local `uv tool` install of the same name takes precedence, so you can develop against a local checkout while the fleet runs the pinned image. Each containerized stack pins its own CUDA build, so it runs on any host with driver ≥ R570 (CUDA backward-compatibility covers newer drivers).

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
- **Localhost only** — the workspace server binds to localhost. Do not expose port 8100 over a network without adding real authentication.
- **No data exfiltration** — `web_search` is disabled; `web_fetch` requires approval. No data leaves the institution's infrastructure by default.

For vulnerability reporting, see [SECURITY.md](SECURITY.md).

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on forking, setting up a development environment, running checks, and submitting pull requests.

---

## License

MedMCP is released under the [Apache License 2.0](LICENSE).
