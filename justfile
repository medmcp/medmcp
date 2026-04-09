set quiet := true

default:
    @just --list

clean:
    rm -rf .mypy_cache
    rm -rf .pytest_cache
    rm -rf .tox
    rm -rf .venv
    rm -rf dist
    rm -rf **/__pycache__
    rm -rf src/*.egg-info
    rm -f .coverage
    rm -f coverage.*

clean-chats:
    rm -rf .vibe/logs
    rm -f .vibe/medmcp_threads.db
    rm -f .vibe/trusted_folders.toml
    rm -rf .vibe/vibehistory

@install_uv:
	if ! command -v uv >/dev/null 2>&1; then \
		echo "uv is not installed. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
	  echo "uv is available and ready to use..."; \
	fi

install_ollama:
    @./scripts/install_ollama.sh

setup: install_uv install_ollama
    uv sync
    uv run pre-commit install

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
