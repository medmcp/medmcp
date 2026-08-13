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

ENV PATH=/app/.venv/bin:$PATH \
    UV_NO_SYNC=1 \
    MEDMCP_WORKSPACE_HOST=0.0.0.0 \
    OLLAMA_BASE_URL=http://llm:11434

EXPOSE 8100
ENTRYPOINT ["tini", "--", "medmcp-entrypoint"]
