# Contributing to MedMCP

Thank you for your interest in contributing to MedMCP!
This guide will help you get set up and familiar with our development workflow.

## Get started!

Ready to contribute? Here's how to set up `medmcp` for local development.

### 1) Create an issue on the GitHub repository

It's good practice to first discuss the proposed changes as the feature might
already be implemented.


### 2) Fork the `medmcp` repository on GitHub

Click [here](https://github.com/medmcp/medmcp/fork) to create your fork.

### 3) Clone your fork locally

```bash
git clone https://github.com/<your-username>/medmcp.git
cd medmcp
```

### 4) Develop in the dev container (recommended)

We **recommend developing and building inside the dev container** (`.devcontainer/`):
it gives everyone the same toolchain and matches the deployment images, so "works
on my machine" problems and host setup drift disappear. It derives from the shared
`medmcp-base` image and bundles Python 3.12 + uv, Node 24, `just`, the Docker CLI,
and the Compose plugin.

**Requirements:** Docker (rootless is supported — point your IDE's Docker
connection at the rootless socket, e.g. `unix:///run/user/$(id -u)/docker.sock`).
Build the base image once first, since the dev image derives from it:

```bash
just docker-base
```

**Open it in your IDE:**

- **PyCharm** (the unified PyCharm, 2024.2+; Docker integration is a Professional
  feature). Open this project, open `.devcontainer/devcontainer.json`, and use the
  **Dev Container** gutter action — PyCharm builds the container and runs an IDE
  backend inside it (first start is slower while it provisions). Ensure Docker
  integration is enabled and pointed at your rootless Docker. JetBrains supports
  the build/general/Compose/lifecycle/variables fields this config uses; see the
  [JetBrains Dev Containers docs](https://www.jetbrains.com/help/pycharm/connect-to-devcontainer.html).
- **VS Code:** **Dev Containers: Reopen in Container** (needs the
  [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers)).
- **CLI:** `devcontainer up --workspace-folder .` (the
  [`devcontainer` CLI](https://github.com/devcontainers/cli)).

On first start it runs `uv sync` and `npm install` automatically. Inside the
container the normal recipes work:

```bash
just check            # lint + format + typecheck + tests
just workspace-build  # build the frontend bundle
just workspace        # run the workspace UI (forwarded to http://localhost:8100)
```

What the dev container wires up for you:

- **Your repo is mounted** for live editing — host and container see the same files.
- **The host's rootless Docker socket is mounted**, so `docker`, `docker compose`,
  and `just compose-up` work *from inside* the container, including launching the
  containerized imaging stacks.
- **Ports 8100 (workspace) and 5173 (Vite dev server) are forwarded** to your host.
- **`OLLAMA_BASE_URL` points at the host's Ollama** (`host.docker.internal:11434`),
  so to use the live agent run `just serve-ollama` (and `just pull-model` once) on
  the **host**; tests and frontend builds don't need it.

### 5) Local install (alternative)

[uv](https://docs.astral.sh/uv/) is recommended for development. You can simply install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

We provide simple [just](https://just.systems/) commands to set up the rest of the development environment. Just can be installed with:
```bash
uv tool install rust-just
```

You can now install `medmcp` and all its dependencies with:
```bash
just setup
```
The command (1) checks if [uv](https://docs.astral.sh/uv/) is available if necessary installs it;
(2) checks if [Ollama](https://ollama.com/) is available and if necessary installs it;
(3) installs all other dependencies.


### 6) Running checks locally

Before pushing, make sure all checks pass - these are the same checks that run
in CI:

```bash
uv run ruff check          # lint
uv run ruff format --check # formatting
uv run pyright             # type-checking (strict mode)
uv run pytest              # tests
```

To auto-fix lint and formatting issues:

```bash
uv run ruff check --fix
uv run ruff format
```

### Building the container images

The deployable images are built with `just` (rootless Docker + the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/) with a
CDI spec for GPU; host driver ≥ R570):

```bash
just docker-base                                    # shared base image (medmcp-base)
just docker-build                                   # core image (medmcp-core:dev)
just docker-build-stack ../medmcp-dicom-dev medmcp-dicom:dev   # a stack image
just compose-up                                     # core + bundled Ollama; http://localhost:8100
just compose-down                                   # tear it down
```

This uses a single shared CUDA base, `stacks.d` manifests, a CUDA-12.8 / driver-R570
floor, and is buildx-ready for multi-arch.

## Code style

- **Formatter/Linter:** [Ruff](https://docs.astral.sh/ruff/) - line length 100, targeting Python 3.12.
- **Type checking:** [Pyright](https://github.com/microsoft/pyright) in strict mode. All code must be fully typed.
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Pre-commit hooks enforce formatting, trailing whitespace, and valid YAML/TOML automatically on each commit.

## Submitting changes

1. Create a feature branch from `main`:

   ```bash
   git checkout -b my-feature
   ```

2. Make your changes and commit with a clear, descriptive message.
3. Push your branch and open a pull request against `main`.
4. CI runs on all pull requests — ensure all checks are green before requesting review.

## Reporting issues

Open a GitHub issue with:

- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Python version and OS
