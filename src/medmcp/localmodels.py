"""Local models: the catalog behind the Models window, and the Ollama side of a switch.

The workspace ships around one model (``settings.OLLAMA_MODEL``, created by the
compose stack from ``Modelfile.muse``) and can switch to another from a small
catalog. Everything here goes through Ollama's HTTP API, so it behaves the same
in the container install and host-native, and needs no docker access.

A switch has four parts, and only the last is visible to the agent:

1. **Download** the model's tag if Ollama does not hold it (streamed progress;
   Ollama resumes a partial download, so cancelling loses nothing).
2. **Inspect** it (``/api/show``): a model without tool calling is refused — the
   agent could do nothing with it — and its context length, default temperature
   and thinking support are read from the model rather than assumed.
3. **Prepare** a derived model that carries the context length. Ollama's
   OpenAI-compatible endpoint cannot set ``num_ctx`` per request, which is the
   reason ``muse-medmcp`` exists as a derived model too.
4. **Record** the choice (``settings.save_local_model``). The caller then
   re-syncs the vibe config and restarts the agent.

Only one model is kept in memory: Ollama is configured never to unload on its
own, so the previous model is unloaded explicitly (:func:`unload_others`).

UI-agnostic: no FastAPI import, so the server and a future CLI can both drive it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx

from medmcp import settings
from medmcp.acp import JsonDict

log: logging.Logger = logging.getLogger(__name__)

# Receives ``{"stage", "text", "completed"?, "total"?}`` as a switch advances.
ProgressFn = Callable[[JsonDict], None]

# The context length a selected model is prepared with: the default model's, or
# the model's own maximum when that is smaller.
TARGET_NUM_CTX: int = settings.DEFAULT_CONTEXT_WINDOW
# Used when Ollama does not report a model's maximum.
FALLBACK_NUM_CTX: int = 32_768
# The system prompt and the tool definitions take roughly 9k tokens before the
# first message; below this there is no room left for a conversation.
MIN_NUM_CTX: int = 16_384
# Share of the context at which vibe compacts — the ratio the shipped config
# uses for the default model (115000 of 131072).
COMPACT_FRACTION: float = 0.875
# For a model whose vendor sets no default. Low on purpose: the model is choosing
# tool arguments and file paths, where creativity is a defect.
DEFAULT_TEMPERATURE: float = 0.3


@dataclass(frozen=True)
class LocalModel:
    """A model the Models window offers."""

    id: str
    label: str
    vendor: str
    # The tag in Ollama's library.
    tag: str
    params: str
    size_gb: float
    license: str
    # Not Apache-2.0: the operator accepts the terms before the download starts.
    license_ack: bool = False


DEFAULT_MODEL_ID: str = "muse-glimmer"
CATALOG: tuple[LocalModel, ...] = (
    LocalModel(
        DEFAULT_MODEL_ID,
        "Muse Glimmer",
        "Meta",
        "muse-glimmer:30b",
        "30B, dense",
        18.2,
        "Apache 2.0",
    ),
    LocalModel(
        "devstral-small-2",
        "Devstral Small 2",
        "Mistral",
        "devstral-small-2:24b",
        "24B, dense",
        15.2,
        "Apache 2.0",
    ),
    LocalModel("gemma4", "Gemma 4 31B", "Google", "gemma4:31b", "31B, dense", 20.4, "Apache 2.0"),
    LocalModel(
        "granite4.2", "Granite 4.2 30B", "IBM", "granite4.2:30b", "30B, dense", 17.7, "Apache 2.0"
    ),
    LocalModel(
        "mistral-small3.2",
        "Mistral Small 3.2",
        "Mistral",
        "mistral-small3.2:24b",
        "24B, dense",
        15.2,
        "Apache 2.0",
    ),
    LocalModel(
        "nemotron-3.5-lightning",
        "Nemotron 3.5 Lightning",
        "NVIDIA",
        "nemotron-3.5-lightning:30b",
        "30B, MoE (3B active)",
        25.4,
        "NVIDIA Open Model License",
        license_ack=True,
    ),
    LocalModel(
        "qwen3.8", "Qwen3.8-27B", "Alibaba", "qwen3.8:27b", "27B, dense", 17.7, "Apache 2.0"
    ),
)


class ModelError(Exception):
    """A model change that cannot proceed. The message is shown to the operator."""


class LicenseConsentRequiredError(ModelError):
    """The model's licence has to be accepted before it is downloaded."""

    def __init__(self, model: LocalModel) -> None:
        """Remember which model asked, so the caller can put the question."""
        super().__init__(f"{model.label} is released under the {model.license}")
        self.model = model


# One change at a time: two switches racing would each unload the other's model.
_change_lock: asyncio.Lock = asyncio.Lock()


def _client(read_timeout: float | None = 30.0) -> httpx.AsyncClient:
    """An HTTP client for the Ollama server. Tests replace this with a mock transport."""
    return httpx.AsyncClient(
        base_url=settings.OLLAMA_BASE_URL, timeout=httpx.Timeout(10.0, read=read_timeout)
    )


def _lookup(model_id: str) -> LocalModel:
    for model in CATALOG:
        if model.id == model_id:
            return model
    raise FileNotFoundError(f"unknown model: {model_id!r}")


def _bare(name: str) -> str:
    """Ollama lists an untagged model as ``name:latest``; compare without it."""
    return name.removesuffix(":latest")


def derived_name(model: LocalModel) -> str:
    """The Ollama model the agent is pointed at for *model*."""
    return settings.OLLAMA_MODEL if model.id == DEFAULT_MODEL_ID else f"medmcp-{model.id}"


def _error_text(resp: httpx.Response) -> str:
    try:
        body = cast("JsonDict", resp.json())
    except ValueError:
        return resp.text.strip() or f"HTTP {resp.status_code}"
    return str(body.get("error") or f"HTTP {resp.status_code}")


async def _installed(client: httpx.AsyncClient) -> set[str]:
    resp = await client.get("/api/tags")
    resp.raise_for_status()
    models = cast("list[JsonDict]", cast("JsonDict", resp.json()).get("models") or [])
    return {_bare(str(m.get("name", ""))) for m in models}


def _is_downloaded(model: LocalModel, installed: set[str]) -> bool:
    name = settings.OLLAMA_MODEL if model.id == DEFAULT_MODEL_ID else model.tag
    return _bare(name) in installed


async def list_models() -> JsonDict:
    """The catalog with each model's state: downloaded, in use, removable.

    Ollama being unreachable is reported (``reachable: false``) rather than
    raised, so the window can say so and still show what is selected.
    """
    selected = await asyncio.to_thread(settings.load_local_model)
    active_id = str(selected["id"]) if selected else DEFAULT_MODEL_ID
    installed: set[str] = set()
    reachable = True
    try:
        async with _client() as client:
            installed = await _installed(client)
    except httpx.HTTPError as exc:
        log.warning("could not list Ollama models: %s", exc)
        reachable = False
    rows: list[JsonDict] = []
    for model in CATALOG:
        default = model.id == DEFAULT_MODEL_ID
        active = model.id == active_id
        downloaded = _is_downloaded(model, installed)
        rows.append(
            {
                "id": model.id,
                "label": model.label,
                "vendor": model.vendor,
                "tag": model.tag,
                "params": model.params,
                "size_gb": model.size_gb,
                "license": model.license,
                "license_ack": model.license_ack,
                "default": default,
                "active": active,
                "downloaded": downloaded,
                # The default is what everything falls back to, and the compose
                # stack re-creates it on every start, so removing it is not offered.
                "deletable": downloaded and not default and not active,
                "num_ctx": selected["num_ctx"] if selected and active else None,
            }
        )
    return {"models": rows, "active": active_id, "reachable": reachable}


async def _pull(client: httpx.AsyncClient, tag: str, on_progress: ProgressFn) -> None:
    """Download *tag*, reporting the bytes done across all of its layers."""
    layers: dict[str, tuple[int, int]] = {}
    loop = asyncio.get_running_loop()
    last_emit = 0.0
    last_text = ""
    async with client.stream("POST", "/api/pull", json={"model": tag}) as resp:
        if resp.status_code >= 400:
            await resp.aread()
            raise ModelError(f"download of {tag} failed: {_error_text(resp)}")
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            try:
                event = cast("JsonDict", json.loads(line))
            except ValueError:
                continue
            if event.get("error"):
                raise ModelError(f"download of {tag} failed: {event['error']}")
            digest, total = event.get("digest"), event.get("total")
            if isinstance(digest, str) and isinstance(total, int):
                done = event.get("completed")
                layers[digest] = (done if isinstance(done, int) else 0, total)
            text = str(event.get("status") or "")
            # Ollama reports several times a second per layer; a frame for each
            # would be most of the socket's traffic and none of its information.
            now = loop.time()
            if text == last_text and now - last_emit < 0.25:
                continue
            last_emit, last_text = now, text
            on_progress(
                {
                    "stage": "download",
                    "text": text,
                    "completed": sum(done for done, _ in layers.values()),
                    "total": sum(total for _, total in layers.values()),
                }
            )


async def _show(client: httpx.AsyncClient, name: str) -> JsonDict:
    resp = await client.post("/api/show", json={"model": name})
    if resp.status_code >= 400:
        raise ModelError(f"could not inspect {name}: {_error_text(resp)}")
    return cast("JsonDict", resp.json())


def _native_context(info: JsonDict) -> int | None:
    """The model's maximum context, from the ``<architecture>.context_length`` key."""
    model_info = info.get("model_info")
    if not isinstance(model_info, dict):
        return None
    for key, value in cast("dict[str, Any]", model_info).items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def _vendor_temperature(info: JsonDict) -> float | None:
    """The temperature the model's publisher set, from the ``parameters`` listing."""
    for line in str(info.get("parameters") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "temperature":
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


async def _create(client: httpx.AsyncClient, name: str, base: str, num_ctx: int) -> None:
    """Create (or refresh) the derived model that carries the context length."""
    resp = await client.post(
        "/api/create",
        json={"model": name, "from": base, "parameters": {"num_ctx": num_ctx}, "stream": False},
        timeout=httpx.Timeout(10.0, read=600.0),
    )
    if resp.status_code >= 400:
        raise ModelError(f"could not prepare {name}: {_error_text(resp)}")


async def unload_others(keep: str) -> None:
    """Unload every model Ollama holds in memory except *keep*. Best-effort.

    Needed after a switch, and again whenever something else may have loaded a
    model behind the selection's back — the compose stack's warm-up service
    preloads the default on every start.
    """
    try:
        async with _client() as client:
            resp = await client.get("/api/ps")
            resp.raise_for_status()
            loaded = cast("list[JsonDict]", cast("JsonDict", resp.json()).get("models") or [])
            for entry in loaded:
                name = str(entry.get("name", ""))
                if name and _bare(name) != _bare(keep):
                    await client.post("/api/generate", json={"model": name, "keep_alive": 0})
    except httpx.HTTPError as exc:
        log.warning("could not unload other models: %s", exc)


async def preload(name: str) -> None:
    """Load *name* into memory so the first prompt is not a cold start. Best-effort."""
    try:
        async with _client(read_timeout=600.0) as client:
            await client.post("/api/generate", json={"model": name})
    except httpx.HTTPError as exc:
        log.warning("could not preload %s: %s", name, exc)


async def select(
    model_id: str, *, accept_license: bool = False, on_progress: ProgressFn | None = None
) -> JsonDict:
    """Make *model_id* the local model, downloading and preparing it if needed.

    Returns the recorded entry (``id``, ``model`` — the Ollama name the agent
    uses — and the settings read from the model). The caller re-syncs the vibe
    config and restarts the agent; until then nothing has changed for a chat.

    Safe to cancel: the choice is recorded last, so an interrupted switch leaves
    the previous model selected and a partial download for Ollama to resume.
    """
    model = _lookup(model_id)
    if _change_lock.locked():
        raise ModelError("another model change is in progress")
    async with _change_lock:
        try:
            return await _select_locked(model, accept_license, on_progress or (lambda _f: None))
        except httpx.HTTPError as exc:
            raise ModelError(
                f"could not reach Ollama at {settings.OLLAMA_BASE_URL}: {exc}"
            ) from exc


async def _select_locked(
    model: LocalModel, accept_license: bool, on_progress: ProgressFn
) -> JsonDict:
    if model.id == DEFAULT_MODEL_ID:
        async with _client() as client:
            if not _is_downloaded(model, await _installed(client)):
                raise ModelError(
                    f"the default model {settings.OLLAMA_MODEL} is not present in Ollama; "
                    "it is created when the stack starts"
                )
        on_progress({"stage": "switch", "text": "Switching model"})
        await unload_others(settings.OLLAMA_MODEL)
        await asyncio.to_thread(settings.save_local_model, None)
        return {"id": model.id, "model": settings.OLLAMA_MODEL}

    if model.license_ack and not accept_license:
        raise LicenseConsentRequiredError(model)

    name = derived_name(model)
    # No read timeout: a download can sit on one layer for a long time.
    async with _client(read_timeout=None) as client:
        if not _is_downloaded(model, await _installed(client)):
            on_progress({"stage": "download", "text": "Starting download"})
            await _pull(client, model.tag, on_progress)
        on_progress({"stage": "prepare", "text": "Preparing model"})
        info = await _show(client, model.tag)
        capabilities = info.get("capabilities")
        if isinstance(capabilities, list) and "tools" not in capabilities:
            raise ModelError(
                f"{model.label} does not support tool calling in this Ollama version, "
                "so the agent cannot work with it"
            )
        num_ctx = min(_native_context(info) or FALLBACK_NUM_CTX, TARGET_NUM_CTX)
        if num_ctx < MIN_NUM_CTX:
            raise ModelError(
                f"{model.label} has a context of {num_ctx} tokens, too small for the agent"
            )
        await _create(client, name, model.tag, num_ctx)

    vendor_temperature = _vendor_temperature(info)
    thinks = isinstance(capabilities, list) and "thinking" in capabilities
    entry: JsonDict = {
        "id": model.id,
        "model": name,
        "base": model.tag,
        "temperature": DEFAULT_TEMPERATURE if vendor_temperature is None else vendor_temperature,
        # vibe sends a reasoning effort unless this is "off", and Ollama rejects
        # one for a model that cannot think.
        "thinking": "medium" if thinks else "off",
        "num_ctx": num_ctx,
        "compact_threshold": int(num_ctx * COMPACT_FRACTION),
    }
    on_progress({"stage": "switch", "text": "Switching model"})
    # Before the new one loads: both in memory at once is the case to avoid.
    await unload_others(name)
    await asyncio.to_thread(settings.save_local_model, entry)
    return entry


async def delete(model_id: str) -> None:
    """Remove a downloaded model (and its derived model) from Ollama.

    Raises ``FileNotFoundError`` for an unknown or not-downloaded model and
    :class:`ModelError` for the default model or the one in use.
    """
    model = _lookup(model_id)
    if model.id == DEFAULT_MODEL_ID:
        raise ModelError("the default model cannot be removed")
    selected = await asyncio.to_thread(settings.load_local_model)
    if selected is not None and selected["id"] == model.id:
        raise ModelError(f"{model.label} is in use; switch to another model first")
    if _change_lock.locked():
        raise ModelError("another model change is in progress")
    async with _change_lock:
        try:
            async with _client(read_timeout=120.0) as client:
                # The derived model first; it may not exist if a switch never finished.
                await client.request("DELETE", "/api/delete", json={"model": derived_name(model)})
                resp = await client.request("DELETE", "/api/delete", json={"model": model.tag})
        except httpx.HTTPError as exc:
            raise ModelError(
                f"could not reach Ollama at {settings.OLLAMA_BASE_URL}: {exc}"
            ) from exc
    if resp.status_code == 404:
        raise FileNotFoundError(f"{model.label} is not downloaded")
    if resp.status_code >= 400:
        raise ModelError(f"could not remove {model.label}: {_error_text(resp)}")
