# syntax=docker/dockerfile:1
#
# MedMCP core (workspace server + vibe-acp + built frontend). CPU-only: it serves
# the UI and *launches* GPU stack containers via the mounted docker socket.
# Multi-arch: derives from medmcp-base (CUDA base, but nothing here runs on GPU)
# and a throwaway node stage; both are multi-arch.
ARG BASE_IMAGE=medmcp-base:dev

# ── Frontend build (throwaway node stage; never shipped) ─────────────────────
FROM node:24-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

# docker CLI only (multi-arch, daemonless) so the core can spawn stack containers
# (`docker run -i`) over the mounted rootless socket.
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app

# Python deps into /app/.venv from the frozen lock (build-time network; the
# runtime is offline). uv installs the project itself (editable) against ./src.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# App assets: built frontend, baked vibe config + system prompt + model def.
COPY --from=frontend /app/frontend/dist ./frontend/dist
COPY docker/config.toml ./.vibe/config.toml
COPY docker/hooks.toml ./.vibe/hooks.toml
COPY .vibe/prompts ./.vibe/prompts
COPY Modelfile.muse ./Modelfile.muse
COPY catalog.json ./catalog.json
COPY catalog.ghcr.json ./catalog.ghcr.json
COPY docker/entrypoint.sh /usr/local/bin/medmcp-entrypoint
RUN chmod +x /usr/local/bin/medmcp-entrypoint

# What this image was built from — a release tag for a released image, a commit
# sha for a rolling :main one. Reported by /healthz; empty for a local build.
ARG MEDMCP_BUILD=""
# The commit itself, which a release tag does not spell out.
ARG MEDMCP_REVISION=""

# OCI metadata. `image.source` is the one GitHub acts on: it links the published
# package to this repository, so the images are reachable from the repo instead
# of floating in the org with no visible relationship to the code. The rest is
# what `docker inspect` and SBOM tooling read to identify an image.
LABEL org.opencontainers.image.source="https://github.com/medmcp/medmcp" \
      org.opencontainers.image.title="MedMCP" \
      org.opencontainers.image.description="A local medical imaging agent" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="$MEDMCP_BUILD" \
      org.opencontainers.image.revision="$MEDMCP_REVISION"

ENV PATH=/app/.venv/bin:$PATH \
    UV_NO_SYNC=1 \
    MEDMCP_BUILD=$MEDMCP_BUILD \
    MEDMCP_WORKSPACE_HOST=0.0.0.0 \
    OLLAMA_BASE_URL=http://llm:11434

EXPOSE 8100
ENTRYPOINT ["tini", "--", "medmcp-entrypoint"]
