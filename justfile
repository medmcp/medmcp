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

# Remove ALL distilled workflows, including promoted ones under active/
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
    # _sync_servers_to_vibe_config (resolves MCP server command paths).
    # skip-worktree prevents those local writes from showing up as dirty.
    git update-index --skip-worktree .vibe/config.toml || true

# Pull upstream changes to .vibe/config.toml (e.g. new tool permissions).
# Run this after pulling a commit that intentionally changes config.toml.
pull-config:
    git update-index --no-skip-worktree .vibe/config.toml
    git checkout .vibe/config.toml
    git update-index --skip-worktree .vibe/config.toml

# Install a stack package into its own isolated uv environment.
# _load_mcp_servers() discovers it automatically by scanning uv tool envs for
# [medmcp.stacks] entry points — no changes to config files needed.
# --torch-backend=auto detects the host NVIDIA driver at install time and picks
# the matching CUDA wheel for any PyTorch-ecosystem deps (or CPU wheels if no GPU),
# so GPU support works on each user's machine regardless of their driver version.
# Usage: just install-stack ../medmcp-neuro
#        just install-stack "git+ssh://git@github.com/medmcp/medmcp-neuro.git"
install-stack STACK:
    uv tool install --torch-backend=auto {{STACK}}

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

# Pull Gemma 4 26B and build custom gemma4-medmcp
pull-model:
    ollama pull gemma4:26b
    ollama create gemma4-medmcp -f Modelfile.gemma4

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

# Distill a reusable workflow (recipe.yaml + SKILL.md) from a session's raw log
# Usage: just distill <session-id>   (add --no-llm to skip the narrative pass)
distill SESSION *ARGS:
    uv run medmcp distill {{SESSION}} {{ARGS}}

# Promote a reviewed draft workflow into active/ so it loads as a skill
# Usage: just promote <workflow-name>
promote NAME:
    uv run medmcp promote {{NAME}}

# List personal workflows (draft + promoted)
workflows:
    uv run medmcp workflows

# Delete a personal workflow (draft or active) by name
# Usage: just delete-workflow <workflow-name>
delete-workflow NAME:
    uv run medmcp delete {{NAME}}

# Launch the workspace UI (explorer + viewer + chat) at http://localhost:8100
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

# One-shot: install everything, pull Gemma 4 model, start Ollama, launch the UI
medmcp: setup pull-model workspace-build
    @echo "Starting Ollama server..."
    @ollama serve &
    @sleep 2
    @echo "Launching the MedMCP workspace..."
    uv run medmcp-workspace
