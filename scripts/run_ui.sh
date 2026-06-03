#!/usr/bin/env bash
set -euo pipefail

# Anchor to the repo root so `.env` lookups work regardless of where the
# script is invoked from.
cd "$(dirname "$0")/.."

if ! grep -q '^CHAINLIT_AUTH_SECRET=' .env 2>/dev/null; then
    secret=$(uv run --quiet chainlit create-secret | grep '^CHAINLIT_AUTH_SECRET=')
    [ -n "$secret" ] || { echo "error: chainlit create-secret produced no CHAINLIT_AUTH_SECRET line" >&2; exit 1; }
    # Ensure .env ends with a newline before appending so the new line doesn't
    # concatenate onto an existing one. `tail -c1` + command substitution
    # strips a trailing newline — if the result is non-empty, the last byte
    # was not a newline, so we add one.
    if [ -s .env ] && [ "$(tail -c1 .env)" ]; then
        printf '\n' >> .env
    fi
    printf '%s\n' "$secret" >> .env
fi

uv run --env-file .env chainlit run src/medmcp/app.py --host 127.0.0.1 --port "${PORT:-8000}"
