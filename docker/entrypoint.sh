#!/usr/bin/env bash
# Core container entrypoint: point vibe-acp's model provider at the LLM service
# (config.toml has no env expansion), sanitize the mounted docker credentials,
# then launch the workspace server.
set -euo pipefail

: "${OLLAMA_BASE_URL:=http://llm:11434}"

# GPU selector for the LLM service and GPU stacks (CDI device id). Stack manifests
# reference ${MEDMCP_GPU} and are expanded with os.path.expandvars, which has no
# ":-" default — so export a default here (and in settings) to keep an unset value
# from leaking the literal placeholder. "all" = every GPU; pin with an index/UUID.
export MEDMCP_GPU="${MEDMCP_GPU:-all}"

# Talk to the mounted (rootless) daemon directly so the in-container docker CLI
# never tries to resolve the host's docker context.
export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"

CONFIG="${VIBE_HOME:-/app/.vibe}/config.toml"

# Optionally interpose the tool-call repair shim between vibe and the LLM service.
# Ollama's Glimmer function-calling parser drops roughly half of all tool calls and
# truncates the stream with no finish_reason and no error — HTTP 200, empty turn,
# which reads as a frozen chat. The shim renames the colliding tool, retries
# truncated turns, and sanitizes malformed arguments. Opt-in, so the direct path
# stays the default; if it fails to come up we fall back rather than strand vibe on
# a dead port. Only vibe is routed through it: the auxiliary calls (explanations,
# distillation prose) read OLLAMA_BASE_URL and keep talking to Ollama directly.
LLM_TARGET_URL="$OLLAMA_BASE_URL"
if [ -n "${MEDMCP_LLM_SHIM:-}" ] && [ "${MEDMCP_LLM_SHIM}" != "0" ]; then
    shim_port="${MEDMCP_SHIM_PORT:-11435}"
    MEDMCP_SHIM_UPSTREAM="$OLLAMA_BASE_URL" \
    MEDMCP_SHIM_HOST=127.0.0.1 \
    MEDMCP_SHIM_PORT="$shim_port" \
        medmcp-llm-shim &

    # vibe-acp starts lazily, but wait anyway so the first prompt cannot race it.
    shim_ready=""
    for _ in $(seq 1 50); do
        if python3 -c "
import socket, sys
s = socket.socket(); s.settimeout(0.2)
sys.exit(0 if s.connect_ex(('127.0.0.1', $shim_port)) == 0 else 1)
" 2>/dev/null; then
            shim_ready=1
            break
        fi
        sleep 0.2
    done

    if [ -n "$shim_ready" ]; then
        LLM_TARGET_URL="http://127.0.0.1:${shim_port}"
        echo "medmcp-entrypoint: tool-call shim on :${shim_port} -> ${OLLAMA_BASE_URL}"
    else
        echo "medmcp-entrypoint: WARNING: shim did not start; using ${OLLAMA_BASE_URL} directly" >&2
    fi
fi

# Rewrite the single [[providers]] api_base from $LLM_TARGET_URL. Idempotent —
# runs on every start, before sync_servers_to_vibe_config (which leaves providers
# untouched). Trailing slashes are stripped so we always emit "<base>/v1".
if [ -f "$CONFIG" ]; then
    base="${LLM_TARGET_URL%/}"
    sed -i -E "s#^api_base = \".*\"#api_base = \"${base}/v1\"#" "$CONFIG"

    # Point the single [[models]] entry at $OLLAMA_MODEL for the same reason:
    # config.toml has no env expansion, and the model the agent chats with is
    # named there while the auxiliary calls (explanations, distillation prose)
    # read the env var. Rewriting here keeps the two from drifting apart, and
    # makes swapping models a one-variable change with no image rebuild.
    # Scoped to the [[models]] block: [[providers]] carries its own `name` and
    # comes first in the file, so an unanchored substitution renames the provider
    # and the agent loses its backend entirely.
    if [ -n "${OLLAMA_MODEL:-}" ]; then
        sed -i -E "/^\[\[models\]\]/,/^name = / s#^name = \".*\"#name = \"${OLLAMA_MODEL}\"#" "$CONFIG"
    fi
fi

# Sanitize the mounted host docker config: keep only `auths`, dropping the host's
# `currentContext`/`contexts`/`credsStore`/`credHelpers`. Those don't resolve
# inside the container — a leaked `currentContext` (the standard rootless setup
# selects one) makes every stack `docker pull`/`docker run` fail with
# "context not found". The host config is mounted read-only at a staging path.
HOST_DOCKER_CONFIG="${HOST_DOCKER_CONFIG:-/run/host-docker-config.json}"
if [ -f "$HOST_DOCKER_CONFIG" ]; then
    mkdir -p /root/.docker
    python3 - "$HOST_DOCKER_CONFIG" /root/.docker/config.json <<'PY'
import json, sys

src, dst = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(src))
except Exception:
    data = {}
auths = data.get("auths", {})
json.dump({"auths": auths}, open(dst, "w"))
if not auths:
    sys.stderr.write(
        "medmcp-entrypoint: note: mounted docker config has no plaintext auths "
        "(host likely uses a credential helper). Published stack images pull "
        "anonymously; only non-public ones need a token or a host-side pre-pull.\n"
    )
PY
fi

# If a registry token is supplied, log in inside the core. This covers hosts
# whose docker config uses a credential helper (so the mounted config carries no
# plaintext auths) — docker login writes the auth into /root/.docker/config.json.
if [ -n "${GHCR_TOKEN:-}" ]; then
    if printf '%s' "$GHCR_TOKEN" | docker login "${GHCR_REGISTRY:-ghcr.io}" \
        -u "${GHCR_USER:-medmcp}" --password-stdin >/dev/null 2>&1; then
        echo "medmcp-entrypoint: logged in to ${GHCR_REGISTRY:-ghcr.io}"
    else
        echo "medmcp-entrypoint: WARNING: docker login to ${GHCR_REGISTRY:-ghcr.io} failed" >&2
    fi
fi

exec medmcp-workspace
