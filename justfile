set quiet := true

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
@install_uv:
    if ! command -v uv >/dev/null 2>&1; then \
        echo "uv is not installed. Installing..."; \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    else \
        echo "uv is available and ready to use..."; \
    fi

# Install Ollama via bundled script
install_ollama:
    @./scripts/install_ollama.sh

# Install uv + Ollama, sync dev environment, register pre-commit hooks
setup: install_uv install_ollama
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

# Pull base model and build custom devstral-medmcp (32k context)
pull_model:
    ollama pull devstral-small-2:latest
    ollama create devstral-medmcp -f Modelfile.devstral

# Launch mistral-vibe CLI (reads .vibe/config.toml)
vibe *ARGS:
    uv run vibe {{ARGS}}

# Launch the Chainlit web UI
ui:
    @./scripts/run_ui.sh
