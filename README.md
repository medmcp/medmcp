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
- **To run (recommended):** Docker (rootless is supported) + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/) with a CDI spec (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) and CDI enabled in the daemon (`"features": { "cdi": true }` in `/etc/docker/daemon.json`, then restart Docker) so `nvidia.com/gpu=all` resolves; host driver ≥ R570 (CUDA 12.8); and access to the private image registry — `docker login ghcr.io` with a `read:packages` token. On rootful Docker, set `MEDMCP_DOCKER_SOCK=/var/run/docker.sock`. Behind a corporate proxy, see [Running behind a proxy](#running-behind-a-proxy).
- **To develop (build from source):** [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (we recommend `uv tool install rust-just`); Node 24 for the frontend.

### Run with Docker (recommended)

MedMCP ships as prebuilt images on the private registry `ghcr.io/medmcp` — a node
**pulls** them, nothing is built locally. You only need two files from this repo:
`docker-compose.ghcr.yml` and `Modelfile.gemma4` (clone the repo, or copy just those).

1. **Authenticate** to the registry (once per node — see [Prerequisites](#prerequisites)):

   ```bash
   docker login ghcr.io   # username: your GitHub user · password: a read:packages token
   ```

2. **Start it** — set `MEDMCP_WORKSPACE` to the absolute host folder holding your imaging data:

   ```bash
   MEDMCP_WORKSPACE="$PWD/data" docker compose -f docker-compose.ghcr.yml up -d
   # from a repo checkout you can use:  just compose-up-ghcr
   ```

   This pulls the CPU-only **core** plus a bundled **Ollama** GPU service and, on first
   run, builds the `gemma4-medmcp` model (~18 GB pull). Open **http://localhost:8100**.

3. **Add tools** — in the UI, open **Settings → Stacks → Available** and install the
   imaging stacks you need (e.g. `dicom`, `neuro`); each pulls its GPU image on demand
   and is then launched by the core when the agent calls it.

The workspace folder is bind-mounted at the **same absolute path** inside the container,
so the agent's outputs land back on your host. Pin a specific release with
`MEDMCP_TAG=<git-sha>` (default `:main`). Tear down with
`docker compose -f docker-compose.ghcr.yml down`. Air-gapped sites mirror the images
into an internal registry — see [Imaging Stacks](#imaging-stacks).

### Run from source (development)

To build and run everything locally instead of pulling images:

```bash
just compose-up   # build the images, then start core + Ollama (http://localhost:8100)
```

Or run host-native (uv + Ollama, no containers):

```bash
just medmcp       # install everything, pull the model, start Ollama, launch the UI
```

MedMCP's primary interface is the **workspace UI** — a four-panel layout (file
explorer, medical-image viewer, workflows, chat). See [Workspace UI](#workspace-ui).

> **Developing MedMCP?** The recommended setup is the **dev container** (PyCharm or
> VS Code) — see [CONTRIBUTING.md](CONTRIBUTING.md).

### Running behind a proxy

Corporate proxies touch MedMCP at three independent layers — configuring only one
is the usual cause of `TLS handshake timeout` or `certificate signed by unknown
authority` during install:

1. **Docker daemon** (pulls the MedMCP/Ollama images). The daemon does *not* read
   your shell's proxy vars — configure it explicitly, e.g. a systemd drop-in
   `/etc/systemd/system/docker.service.d/http-proxy.conf` with
   `Environment="HTTPS_PROXY=http://user:pass@proxy:port"` then
   `systemctl daemon-reload && systemctl restart docker`. (In a systemd unit, a
   literal `%` in the password must be escaped as `%%`.)

2. **Containers** (the first-run `ollama pull` of `gemma4:26b` reaches out to
   `registry.ollama.ai`). Set proxy env for containers via `~/.docker/config.json`
   `"proxies"`, and **add the compose service names to `noProxy`** so
   container-to-container calls don't get sent to the proxy:

   ```json
   { "proxies": { "default": {
       "httpProxy":  "http://user:pass@proxy:port",
       "httpsProxy": "http://user:pass@proxy:port",
       "noProxy": "localhost,127.0.0.1,llm,llm-init,medmcp"
   } } }
   ```

3. **TLS interception (MITM).** If the proxy re-signs TLS, add the proxy overlay so
   containers trust its CA — point `MEDMCP_CA_BUNDLE` at a bundle that includes the
   proxy root (the host's own bundle usually already does):

   ```bash
   MEDMCP_WORKSPACE="$PWD/data" MEDMCP_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
     docker compose -f docker-compose.ghcr.yml -f docker-compose.proxy.yml up -d
   ```

**Simplest escape hatch:** pull the model on the host once
(`ollama pull gemma4:26b`) and set `OLLAMA_MODELS_DIR` to your host store (e.g.
`~/.ollama`). The container then reuses it and skips the ~18 GB in-container pull
entirely — sidestepping layers 2 and 3 for the model download.

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

**Private images (GHCR).** CI (`.github/workflows/images.yml`) builds and pushes the base, core, and stack images to the **private** registry `ghcr.io/medmcp/*` (packages are private by default; the base package must grant the stack repos read access). For a private fleet:

1. `docker login ghcr.io` on each node (a `read:packages` token); compose mounts the host's docker credentials so the in-UI install can pull.
2. Run the **published** images (pulls the core, builds nothing) with the deploy compose — `just compose-up-ghcr`, i.e. `docker compose -f docker-compose.ghcr.yml up -d`. It defaults `MEDMCP_CATALOG_URL` to the bundled `/app/catalog.ghcr.json`; pin a release with `MEDMCP_TAG=<git-sha>`. A node needs only this file + `Modelfile.gemma4`.

Air-gapped sites mirror the images into an internal registry and point the catalog there instead. No image data leaves the machine either way — these are inbound pulls.

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
