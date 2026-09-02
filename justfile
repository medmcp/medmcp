set quiet := true

alias help := default

# Defaults to just --list
default:
    @just --list

# Remove caches and build artifacts
clean:
    rm -rf .mypy_cache
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf .tox
    rm -rf .venv
    rm -rf dist
    rm -rf build
    rm -rf **/__pycache__
    rm -rf src/*.egg-info
    rm -f .coverage
    rm -f coverage.*

# Promoted workflows under .vibe/workflows/active/ are preserved; use
# clean-workflows to remove those too.

# Remove local chat/session state under .vibe/ (logs, history, provenance, drafts)
clean-chats:
    rm -rf .vibe/logs
    rm -f .vibe/trusted_folders.toml
    rm -rf .vibe/vibehistory
    rm -rf .vibe/provenance
    rm -rf .vibe/workflows/draft

# Remove ALL distilled workflows
clean-workflows:
    rm -rf .vibe/workflows

# Install uv (only if just is installed via package manager)
@install-uv:
    if ! command -v uv >/dev/null 2>&1; then \
        echo "uv is not installed. Installing..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    else \
        echo "uv is available and ready to use..."; \
    fi

# Install Ollama via bundled script
install-ollama:
    @./scripts/install_ollama.sh

# Install uv + Ollama, sync dev environment, register pre-commit hooks
setup: install-uv install-ollama
    uv sync
    uv run pre-commit install
    # .vibe/config.toml is tracked in git but also written at runtime by
    # sync_servers_to_vibe_config (resolves MCP server command paths).
    # skip-worktree prevents those local writes from showing up as dirty.
    git update-index --skip-worktree .vibe/config.toml || true

# Pull upstream changes to .vibe/config.toml (e.g. new tool permissions).
# Run this after pulling a commit that intentionally changes config.toml.
pull-config:
    git update-index --no-skip-worktree .vibe/config.toml
    git checkout .vibe/config.toml
    git update-index --skip-worktree .vibe/config.toml

# Install a stack package into its own isolated uv environment.
# load_mcp_servers() discovers it automatically by scanning uv tool envs for
# [medmcp.stacks] entry points — no changes to config files needed.
# Resolution is deterministic from the stack's committed lock: stacks pin their
# own CUDA build in pyproject (e.g. medmcp-neuro pins the cu128 PyTorch index), so
# installs match the containers. (Don't add --torch-backend=auto — it picks a
# wheel from the host driver and would override that pin; host floor is R570.)
# Usage: just install-stack ../medmcp-neuro
#        just install-stack "git+ssh://git@github.com/medmcp/medmcp-neuro.git"
install-stack STACK:
    uv tool install {{STACK}}

# Uninstall a stack package from its isolated uv environment.
# Usage: just uninstall-stack medmcp-neuro
uninstall-stack STACK:
    uv tool uninstall {{STACK}}

# Run every CI check locally (lint, format, typecheck, tests)
check: lint format-check typecheck test

# Lint with ruff
lint:
    uv run ruff check

# Format code with ruff
format:
    uv run ruff format

# Check formatting without writing changes
format-check:
    uv run ruff format --check

# Strict type-checking with pyright
typecheck:
    uv run pyright

# Run the pytest suite
test *ARGS:
    uv run pytest {{ARGS}}

# Auto-fix lint findings and format
fix:
    uv run ruff check --fix
    uv run ruff format

# Regenerate THIRD_PARTY_NOTICES.md from the resolved deps (offline; run after a
# runtime dependency change). Reads uv.lock + node_modules, no network.
notices:
    uv run python scripts/gen_third_party_notices.py

# Verify THIRD_PARTY_NOTICES.md matches the installed deps (the CI gate; needs
# both .venv synced and frontend node_modules installed). Regenerates, then fails
# if it drifted — re-stage the file and commit.
notices-check:
    uv run python scripts/gen_third_party_notices.py
    git diff --exit-code THIRD_PARTY_NOTICES.md

# Pull Muse Glimmer 30B and build the custom muse-medmcp model.
# Needs Ollama >= 0.32.15: 0.32.9 was the first build to accept the manifest
# (older ones reject it with a 412), and 0.32.15 the first where a tool-call
# parser error mid-stream no longer wedges the chat.
pull-model:
    ollama pull muse-glimmer:30b
    ollama create muse-medmcp -f Modelfile.muse

# Launch mistral-vibe CLI (reads .vibe/config.toml)
vibe *ARGS:
    uv run vibe {{ARGS}}

# List sessions that have a provenance record
provenance-list:
    uv run medmcp list

# Render the human-readable provenance report for a session
# Usage: just report <session-id>
report SESSION:
    uv run medmcp report {{SESSION}}

# Distill a replayable workflow (recipe.yaml) from a session's raw log
# Usage: just distill <session-id>
distill SESSION:
    uv run medmcp distill {{SESSION}}

# List personal workflows
workflows:
    uv run medmcp workflows

# Delete a personal workflow by name
# Usage: just delete-workflow <workflow-name>
delete-workflow NAME:
    uv run medmcp delete {{NAME}}

# Export a workflow to a shareable <name>.workflow.yaml (add --out PATH to override)
# Usage: just export-workflow <workflow-name>
export-workflow NAME *ARGS:
    uv run medmcp export {{NAME}} {{ARGS}}

# Import a shared workflow file as a new workflow
# Usage: just import-workflow <file.workflow.yaml>
import-workflow FILE:
    uv run medmcp import {{FILE}}

# Launch the workspace UI (explorer + viewer + workflows + chat) at http://localhost:8100
workspace:
    uv run medmcp-workspace

# Build the workspace frontend bundle (requires node/npm)
workspace-build:
    cd frontend && npm install && npm run build

# Run the frontend dev server with hot reload (proxies to medmcp-workspace)
workspace-dev:
    cd frontend && npm run dev

# Start the Ollama server (blocks until stopped)
serve-ollama:
    ollama serve

# One-shot: install everything, pull the model, start Ollama, launch the UI
medmcp: setup pull-model workspace-build
    @echo "Starting Ollama server..."
    @ollama serve &
    @sleep 2
    @echo "Launching the MedMCP workspace..."
    uv run medmcp-workspace

# ── Containers ────────────────────────────────────────────────────────────────

# Build the shared base image (medmcp-base) locally for the host arch.
# CUDA_TAG aligns the base CUDA runtime with the deployment target (see Dockerfile.base).
docker-base CUDA_TAG="12.8.1":
    docker build -f Dockerfile.base --build-arg CUDA_TAG={{CUDA_TAG}} -t medmcp-base:dev .

# Build the core image locally (depends on medmcp-base).
docker-build: docker-base
    docker build -t medmcp-core:dev .

# Build a stack image from a sibling repo. Usage: just docker-build-stack ../medmcp-dicom-dev ghcr.io/medmcp/dicom:dev
docker-build-stack DIR TAG:
    docker build -t {{TAG}} {{DIR}}

# Build the multi-arch base + push (needs a docker-container builder + registry login).
# Usage: just docker-base-multiarch ghcr.io/medmcp/base:dev "linux/amd64,linux/arm64"
docker-base-multiarch TAG PLATFORMS="linux/amd64,linux/arm64" CUDA_TAG="12.8.1":
    docker buildx build -f Dockerfile.base --build-arg CUDA_TAG={{CUDA_TAG}} \
        --platform {{PLATFORMS}} -t {{TAG}} --push .

# Bring up the full stack (llm + core). Defaults: workspace=./data, models=host ~/.ollama
# (so the bundled Ollama reuses an already-pulled muse-medmcp instead of an 18 GB download).
compose-up:
    MEDMCP_WORKSPACE="${MEDMCP_WORKSPACE:-$(pwd)/data}" \
    OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$HOME/.ollama}" \
    docker compose up -d --build

# Bring up the stack from the PUBLISHED GHCR images (pulls core, builds nothing).
# Pin a release with MEDMCP_TAG=<git-sha>.
compose-up-ghcr:
    MEDMCP_WORKSPACE="${MEDMCP_WORKSPACE:-$(pwd)/data}" \
    OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$HOME/.ollama}" \
    docker compose -f docker-compose.ghcr.yml up -d --pull always

# Tear down the stack.
compose-down:
    MEDMCP_WORKSPACE="${MEDMCP_WORKSPACE:-$(pwd)/data}" docker compose down

# Tail core + llm logs.
compose-logs:
    MEDMCP_WORKSPACE="${MEDMCP_WORKSPACE:-$(pwd)/data}" docker compose logs -f --tail=100
