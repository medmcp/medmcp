#!/usr/bin/env bash
# Core container entrypoint: point vibe-acp's model provider at the LLM service
# (config.toml has no env expansion), then launch the workspace server.
set -euo pipefail

: "${OLLAMA_BASE_URL:=http://llm:11434}"
CONFIG="${VIBE_HOME:-/app/.vibe}/config.toml"

# Rewrite the single [[providers]] api_base from $OLLAMA_BASE_URL. Idempotent —
# runs on every start, before sync_servers_to_vibe_config (which leaves providers
# untouched). Trailing slashes are stripped so we always emit "<base>/v1".
if [ -f "$CONFIG" ]; then
    base="${OLLAMA_BASE_URL%/}"
    sed -i -E "s#^api_base = \".*\"#api_base = \"${base}/v1\"#" "$CONFIG"
fi

exec medmcp-workspace
