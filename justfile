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

# Remove local chat/session state under .vibe/
clean-chats:
    rm -rf .vibe/logs
    rm -f .vibe/medmcp_threads.db
    rm -f .vibe/trusted_folders.toml
    rm -rf .vibe/vibehistory

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

# Pull Gemma 4 26B and build custom gemma4-medmcp (32k context)
pull-model:
    ollama pull gemma4:26b
    ollama create gemma4-medmcp -f Modelfile.gemma4

# Launch mistral-vibe CLI (reads .vibe/config.toml)
vibe *ARGS:
    uv run vibe {{ARGS}}

# Launch the Chainlit web UI
ui:
    @./scripts/run_ui.sh

# Start the Ollama server (blocks until stopped)
serve-ollama:
    ollama serve

# One-shot: install everything, pull Gemma 4 model, start Ollama, launch the UI
medmcp: setup pull-model
    @echo "Starting Ollama server..."
    @ollama serve &
    @sleep 2
    @echo "Launching MedMCP UI..."
    @./scripts/run_ui.sh
