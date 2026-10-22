"""The isolated agent's only way out: an HTTP gateway in the core.

With agent isolation the agent container sits on an internal network: it can
reach the core and nothing else — not the model server, not the internet. Every
connection it legitimately needs goes through this gateway, which listens on the
core's address on that network only:

- ``/provider/<name>/…`` — a model provider from ``config.toml``: the local model
  server (chat and model-list paths only, so Ollama's admin API — pull, delete,
  create — stays out of reach) or the cloud model, while it is in force.
- ``/mcp/<name>/…`` — an external MCP server, while the feature and that server
  are switched on.

The gateway holds the credentials. The agent's copy of ``config.toml`` names the
gateway instead of the real endpoints (:func:`agent_config`), its environment
carries placeholders where keys used to be, and the gateway adds the real header
on the way out. So a command the agent runs can neither reach a host the operator
did not configure nor read a key to use elsewhere.

Each request re-reads what is in force, so switching the cloud model or an
external server off closes its route at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import tomllib
from typing import Any, cast
from urllib.parse import urlsplit

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from medmcp import settings

log: logging.Logger = logging.getLogger(__name__)
_audit: logging.Logger = logging.getLogger("medmcp.audit")

JsonDict = dict[str, Any]

GATEWAY_PORT = 11436
# The paths a model provider is reached on. Anything else on the local model
# server (Ollama's /api/pull, /api/delete, /api/create …) is not the agent's.
_PROVIDER_PATH_SUFFIXES: tuple[str, ...] = ("chat/completions", "models", "responses", "messages")
# What a forwarded path may contain. The path arrives percent-decoded, so a
# literal ?, # or % would be re-read as a query, fragment or escape when the
# upstream URL is built — a route around the endpoint check.
_SAFE_PATH = re.compile(r"[A-Za-z0-9._~\-/:@+,=]*")
# Request headers never forwarded: hop-by-hop, the agent's own (placeholder)
# credentials, and anything the client cannot be trusted to set.
_DROP_REQUEST_HEADERS: frozenset[str] = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "authorization",
        "x-api-key",
        "cookie",
    }
)
_DROP_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "content-length", "set-cookie"}
)
# Placed in the agent's environment where a key used to be: vibe refuses to open a
# session when a provider's key variable is empty, and the gateway replaces it.
PLACEHOLDER = "held-by-the-medmcp-gateway"


# ── the agent's copy of the config ───────────────────────────────────────────


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _path_and_query(url: str) -> str:
    parts = urlsplit(url)
    return parts.path.rstrip("/") + (f"?{parts.query}" if parts.query else "")


def agent_config(cfg: JsonDict, gateway: str) -> JsonDict:
    """*cfg* as the isolated agent must see it: every endpoint through *gateway*.

    A provider's ``api_base`` keeps its path under ``/provider/<name>``, an
    external server's ``url`` under ``/mcp/<name>``, and the server's ``auth``
    block is dropped — the gateway authenticates instead.
    """
    out = dict(cfg)
    providers: list[JsonDict] = []
    for raw in cast("list[JsonDict]", cfg.get("providers", [])):
        provider = dict(raw)
        if provider.get("api_base") and provider.get("name"):
            provider["api_base"] = (
                f"{gateway}/provider/{provider['name']}{_path_and_query(provider['api_base'])}"
            )
        providers.append(provider)
    out["providers"] = providers
    servers: list[JsonDict] = []
    for raw in cast("list[JsonDict]", cfg.get("mcp_servers", [])):
        server = dict(raw)
        if server.get("url"):
            server["url"] = f"{gateway}/mcp/{server['name']}{_path_and_query(server['url'])}"
            server.pop("auth", None)
        servers.append(server)
    out["mcp_servers"] = servers
    return out


def placeholder_env(secret_env: dict[str, str]) -> dict[str, str]:
    """The agent's stand-ins for *secret_env*: same names, no values."""
    return dict.fromkeys(secret_env, PLACEHOLDER)


# ── routing ──────────────────────────────────────────────────────────────────


def _core_config() -> JsonDict:
    try:
        with (settings.VIBE_HOME / "config.toml").open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _safe_path(path: str) -> bool:
    """Whether *path* is plain path characters with no ``.``/``..`` segment."""
    return bool(_SAFE_PATH.fullmatch(path)) and not any(
        seg in {".", ".."} for seg in path.split("/")
    )


def _model_endpoint(path: str, base: str) -> bool:
    """Whether *path* is a model endpoint under the provider's base path, exactly."""
    prefixes = {base, f"{base}/v1"} if base else {"", "v1"}
    path = path.strip("/")
    return any(
        path == f"{prefix}/{suffix}".lstrip("/")
        for prefix in prefixes
        for suffix in _PROVIDER_PATH_SUFFIXES
    )


def _provider_target(name: str) -> tuple[str, str, dict[str, str]] | None:
    """``(origin, base path, auth headers)`` for the provider *name* in force, or ``None``."""
    providers = cast("list[JsonDict]", _core_config().get("providers", []))
    provider = next((p for p in providers if p.get("name") == name), None)
    if provider is None or not provider.get("api_base"):
        return None
    if name != settings.CLOUD_PROVIDER_NAME:
        api_base = str(provider["api_base"])
        return _origin(api_base), urlsplit(api_base).path.strip("/"), {}
    cloud = settings.active_cloud_model()
    if cloud is None:
        return None
    env_name = str(cloud.get("api_key_env") or "")
    key = os.environ.get(env_name, "") if env_name else ""
    if env_name == settings.CLOUD_MODEL_KEY_ENV:
        key = settings.load_cloud_model_key() or key
    headers: dict[str, str] = {}
    if key:
        if cloud.get("api_style") == "anthropic":
            headers["x-api-key"] = key
        else:
            headers["authorization"] = f"Bearer {key}"
    api_base = str(cloud["api_base"])
    return _origin(api_base), urlsplit(api_base).path.strip("/"), headers


def _mcp_target(name: str) -> tuple[str, dict[str, str]] | None:
    """``(origin, auth headers)`` for the external server *name* now in force."""
    if not settings.external_mcp_enabled():
        return None
    srv = next(
        (
            s
            for s in cast("list[JsonDict]", settings.load_external_mcp()["servers"])
            if s.get("name") == name and s.get("active")
        ),
        None,
    )
    if srv is None:
        return None
    env_name = str(srv.get("api_key_env") or "")
    token = ""
    if env_name == settings.external_secret_env_var(name):
        token = settings.load_external_secrets().get(name, "")
    elif env_name:
        token = os.environ.get(env_name, "")
    headers: dict[str, str] = {}
    if token:
        header = str(srv.get("api_key_header") or "Authorization").lower()
        fmt = str(srv.get("api_key_format") or "Bearer {token}")
        headers[header] = fmt.replace("{token}", token)
    return _origin(str(srv["url"])), headers


class EgressGateway:
    """The gateway's HTTP server (see the module docstring)."""

    def __init__(
        self, host: str, port: int = GATEWAY_PORT, client: httpx.AsyncClient | None = None
    ) -> None:
        """Listen on *host*:*port* — the core's address on the agent network."""
        self.host = host
        self.port = port
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=None, write=120.0, pool=30.0),
            follow_redirects=False,
            trust_env=False,
        )
        self.app = Starlette(
            routes=[
                Route("/provider/{name}/{path:path}", self._provider, methods=["GET", "POST"]),
                Route("/mcp/{name}/{path:path}", self._mcp, methods=["GET", "POST", "DELETE"]),
            ]
        )
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        """Where the agent reaches the gateway."""
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        """Serve in the background on this event loop."""
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="warning", lifespan="off"
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(100):
            if self._server.started:
                break
            await asyncio.sleep(0.05)
        log.info("egress gateway for the agent at %s", self.base_url)

    async def aclose(self) -> None:
        """Stop serving and close the upstream client."""
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await self._task
        await self._client.aclose()

    async def _provider(self, request: Request) -> Response:
        name, path = request.path_params["name"], request.path_params["path"]
        target = _provider_target(name)
        if target is None:
            return self._refuse(request, f"provider {name!r} is not in force")
        origin, base, auth = target
        if not _safe_path(path) or not _model_endpoint(path, base):
            return self._refuse(request, f"path {path!r} is not a model endpoint")
        return await self._forward(request, origin, path, auth)

    async def _mcp(self, request: Request) -> Response:
        name, path = request.path_params["name"], request.path_params["path"]
        target = _mcp_target(name)
        if target is None:
            return self._refuse(request, f"external server {name!r} is not switched on")
        if not _safe_path(path):
            return self._refuse(request, f"path {path!r} is not allowed")
        return await self._forward(request, target[0], path, target[1])

    def _refuse(self, request: Request, reason: str) -> Response:
        _audit.warning("gateway refused %s %s: %s", request.method, request.url.path, reason)
        return JSONResponse({"error": reason}, status_code=403)

    async def _forward(
        self,
        request: Request,
        origin: str,
        path: str,
        auth: dict[str, str],
    ) -> Response:
        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS
        }
        headers.update(auth)
        # Built from parts, never by formatting a string the client influenced.
        upstream = self._client.build_request(
            request.method,
            httpx.URL(origin).copy_with(path="/" + path.lstrip("/")),
            params=request.query_params,
            headers=headers,
            content=await request.body(),
        )
        try:
            resp = await self._client.send(upstream, stream=True)
        except httpx.HTTPError as exc:
            log.warning("gateway upstream %s failed: %s", origin, exc)
            return JSONResponse({"error": f"upstream unreachable: {exc}"}, status_code=502)
        out_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS
        }
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=out_headers,
            background=BackgroundTask(resp.aclose),
        )
