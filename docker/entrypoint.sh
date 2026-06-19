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

# Rewrite the single [[providers]] api_base from $OLLAMA_BASE_URL. Idempotent —
# runs on every start, before sync_servers_to_vibe_config (which leaves providers
# untouched). Trailing slashes are stripped so we always emit "<base>/v1".
if [ -f "$CONFIG" ]; then
    base="${OLLAMA_BASE_URL%/}"
    sed -i -E "s#^api_base = \".*\"#api_base = \"${base}/v1\"#" "$CONFIG"
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
        "medmcp-entrypoint: WARNING: mounted docker config has no plaintext auths "
        "(host likely uses a credential helper) — private stack image pulls will "
        "fail; pre-pull them on the host or provide a token.\n"
    )
PY
fi

exec medmcp-workspace
