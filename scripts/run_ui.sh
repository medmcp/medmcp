#!/usr/bin/env bash
set -euo pipefail

if ! grep -q '^CHAINLIT_AUTH_SECRET=' .env 2>/dev/null; then
    secret=$(uv run --quiet chainlit create-secret | grep '^CHAINLIT_AUTH_SECRET=')
    [ -n "$secret" ] || { echo "error: chainlit create-secret produced no CHAINLIT_AUTH_SECRET line" >&2; exit 1; }
    printf '%s\n' "$secret" >> .env
fi

uv run --env-file .env chainlit run src/medmcp/app.py
