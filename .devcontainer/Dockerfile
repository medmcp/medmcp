# syntax=docker/dockerfile:1
#
# MedMCP dev container — the owned base + the full dev toolchain (node, just,
# docker CLI, build tools). Derives from medmcp-base so it shares the CUDA/python/
# uv layers with the deploy images. Python deps are synced at postCreate (against
# the mounted source), not baked, so the venv tracks live edits.
ARG BASE_IMAGE=medmcp-base:dev
FROM ${BASE_IMAGE}

ARG NODE_VERSION=24.16.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        gnupg \
        less \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Node (official tarball, arch-aware) for the frontend dev server / build.
RUN arch="$(dpkg --print-architecture)"; case "$arch" in \
        amd64) na=x64 ;; arm64) na=arm64 ;; *) echo "unsupported arch $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${na}.tar.gz" \
    | tar -xz -C /usr/local --strip-components=1

# just (task runner) and the docker CLI + compose plugin (to drive compose /
# launch stacks from inside the dev container via the mounted rootless socket).
RUN curl --proto '=https' --tlsv1.2 -fsSL https://just.systems/install.sh \
    | bash -s -- --to /usr/local/bin
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker
RUN mkdir -p /usr/libexec/docker/cli-plugins; \
    arch="$(dpkg --print-architecture)"; case "$arch" in \
        amd64) ca=x86_64 ;; arm64) ca=aarch64 ;; *) echo "unsupported arch $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-${ca}" \
        -o /usr/libexec/docker/cli-plugins/docker-compose; \
    chmod +x /usr/libexec/docker/cli-plugins/docker-compose

WORKDIR /workspaces/medmcp
