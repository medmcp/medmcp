"""``medmcp-llm-shim`` — a reverse proxy that repairs Ollama's Glimmer tool calls.

Ollama's Muse Glimmer function-calling parser drops tool calls. Measured against
the real workspace system prompt and tool list, ~50% of streaming requests fail;
the failure mode differs by transport, and the streaming one is silent:

* **non-streaming** → ``HTTP 500 parse Glimmer call to <tool>: malformed ATEM parameter``
* **streaming** → the stream is truncated with **no** ``finish_reason``, no ``usage``
  chunk, and an empty ``model`` on the final chunk. HTTP status is already 200, so
  no client can tell the turn failed. vibe records an empty assistant message and
  the chat appears frozen.

This shim sits between vibe and Ollama and applies three repairs:

1. **Tool renaming.** The dominant trigger is the tool literally named ``skill``,
   which collides in Glimmer's ATEM recipient namespace (hence the sibling error
   ``Glimmer recipient "skill" does not match ...``). Renaming it on the wire lifts
   streaming success from ~50% to ~90% — the same as deleting the tool outright, but
   the capability is kept. Names are restored on the way back, so vibe still sees
   and executes its own builtin.
2. **Retry on silent truncation.** A missing ``finish_reason`` is a reliable
   abnormal-termination signal, so the shim retries instead of passing the empty
   turn through. vibe's own retry cannot help: a truncated stream is still HTTP 200,
   so vibe never sees an error.
3. **Argument sanitising.** Ollama rejects (400 ``invalid tool call arguments``) any
   assistant ``tool_calls[].function.arguments`` that is not a JSON *object* string.
   vibe replays whatever the model produced verbatim, so one malformed call poisons
   the history and every later request 400s — a permanently dead chat. Coercing to
   ``"{}"`` on the way out breaks that trap.

Reasoning deltas are relayed live so the UI keeps its "thinking" feed; only
``content`` and ``tool_calls`` are buffered until the turn is known to be healthy.
If every attempt fails the shim emits an explicit, clearly-labelled error message
rather than an empty turn — a visible failure beats a silent one.

Inert until ``api_base`` in ``.vibe/config.toml`` points at it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

log: logging.Logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

# Tool names rewritten on the way to Ollama and restored on the way back. Only the
# wire name changes; vibe keeps executing its own builtin under the original name.
DEFAULT_RENAMES: dict[str, str] = {"skill": "load_skill"}

_SSE_PREFIX = "data: "
_SSE_DONE = "[DONE]"

# Total upstream attempts per client request, including the first.
DEFAULT_ATTEMPTS: int = 3

_UPSTREAM_ENV = "MEDMCP_SHIM_UPSTREAM"
_DEFAULT_UPSTREAM = "http://llm:11434"


# ── request repair (pure) ──────────────────────────────────


def sanitize_arguments(raw: object) -> str:
    """Coerce a tool call's ``arguments`` into a JSON-object string.

    Ollama unmarshals ``arguments`` as a string and then parses it as a JSON object,
    rejecting the request outright if either step fails. Anything that would fail —
    ``None``, ``""``, whitespace, truncated JSON, or a non-object like ``[]`` — is
    replaced with ``"{}"`` so the request survives. A dict is re-serialised rather
    than dropped, since its content is still meaningful.
    """
    if isinstance(raw, dict):
        try:
            return json.dumps(cast("JsonDict", raw))
        except (TypeError, ValueError):
            return "{}"
    if not isinstance(raw, str):
        return "{}"
    try:
        parsed: object = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return "{}"
    return raw if isinstance(parsed, dict) else "{}"


def _rename_tool_calls(message: JsonDict, renames: Mapping[str, str]) -> None:
    """Rewrite tool-call names and sanitise arguments on one message, in place."""
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return
    for raw_call in cast("list[object]", raw_calls):
        if not isinstance(raw_call, dict):
            continue
        call = cast("JsonDict", raw_call)
        raw_fn = call.get("function")
        if not isinstance(raw_fn, dict):
            continue
        fn = cast("JsonDict", raw_fn)
        name = fn.get("name")
        if isinstance(name, str) and name in renames:
            fn["name"] = renames[name]
        fn["arguments"] = sanitize_arguments(fn.get("arguments"))


def repair_request(payload: JsonDict, renames: Mapping[str, str]) -> JsonDict:
    """Return ``payload`` with tools renamed and tool-call arguments sanitised.

    Renames are applied to the advertised ``tools`` list, to assistant ``tool_calls``
    already in the history, and to the ``name`` on ``tool`` result messages, so the
    model never sees the two spellings mixed.
    """
    out = cast("JsonDict", json.loads(json.dumps(payload)))

    raw_tools = out.get("tools")
    if isinstance(raw_tools, list):
        for raw_tool in cast("list[object]", raw_tools):
            if not isinstance(raw_tool, dict):
                continue
            raw_fn = cast("JsonDict", raw_tool).get("function")
            if not isinstance(raw_fn, dict):
                continue
            fn = cast("JsonDict", raw_fn)
            name = fn.get("name")
            if isinstance(name, str) and name in renames:
                fn["name"] = renames[name]

    raw_messages = out.get("messages")
    if isinstance(raw_messages, list):
        for raw_message in cast("list[object]", raw_messages):
            if not isinstance(raw_message, dict):
                continue
            message = cast("JsonDict", raw_message)
            _rename_tool_calls(message, renames)
            if message.get("role") == "tool":
                tool_name = message.get("name")
                if isinstance(tool_name, str) and tool_name in renames:
                    message["name"] = renames[tool_name]

    return out


def restore_response(payload: JsonDict, inverse: Mapping[str, str]) -> JsonDict:
    """Rewrite renamed tool names in a response back to what the caller expects."""
    for key in ("choices",):
        raw_choices = payload.get(key)
        if not isinstance(raw_choices, list):
            continue
        for raw_choice in cast("list[object]", raw_choices):
            if not isinstance(raw_choice, dict):
                continue
            choice = cast("JsonDict", raw_choice)
            for slot in ("message", "delta"):
                raw_msg = choice.get(slot)
                if isinstance(raw_msg, dict):
                    _restore_names(cast("JsonDict", raw_msg), inverse)
    return payload


def _restore_names(message: JsonDict, inverse: Mapping[str, str]) -> None:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return
    for raw_call in cast("list[object]", raw_calls):
        if not isinstance(raw_call, dict):
            continue
        raw_fn = cast("JsonDict", raw_call).get("function")
        if not isinstance(raw_fn, dict):
            continue
        fn = cast("JsonDict", raw_fn)
        name = fn.get("name")
        if isinstance(name, str) and name in inverse:
            fn["name"] = inverse[name]


# ── stream classification ──────────────────────────────────


class StreamOutcome:
    """Accumulates one upstream SSE stream and judges whether it completed.

    ``content`` and ``tool_calls`` chunks are withheld until the stream is known to
    be healthy, so a truncated attempt can be retried without the caller having seen
    a partial answer. Reasoning chunks are released immediately — they are display-only
    and replaying them across attempts is harmless.
    """

    def __init__(self) -> None:
        """Start an empty outcome for one upstream attempt."""
        self.saw_finish_reason: bool = False
        self.saw_usable: bool = False
        self.buffered: list[JsonDict] = []
        self.tail: list[JsonDict] = []

    @property
    def healthy(self) -> bool:
        """Whether the stream terminated normally.

        A missing ``finish_reason`` is the abnormal-termination signal: healthy turns
        always carry one (``"tool_calls"`` or ``"stop"``), truncated ones never do.
        """
        return self.saw_finish_reason

    def feed(self, event: JsonDict) -> JsonDict | None:
        """Absorb one SSE event; return an event to forward now, or ``None``."""
        raw_choices = event.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            # A usage-only or keepalive frame belongs with the tail, after content.
            self.tail.append(event)
            return None

        passthrough = False
        for raw_choice in cast("list[object]", raw_choices):
            if not isinstance(raw_choice, dict):
                continue
            choice = cast("JsonDict", raw_choice)
            if choice.get("finish_reason"):
                self.saw_finish_reason = True
            raw_delta = choice.get("delta")
            delta = cast("JsonDict", raw_delta) if isinstance(raw_delta, dict) else {}
            if delta.get("content") or delta.get("tool_calls"):
                self.saw_usable = True
            # Reasoning-only frames stream straight through.
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning and not delta.get("content") and not delta.get("tool_calls"):
                passthrough = True

        if passthrough:
            return event
        self.buffered.append(event)
        return None

    def release(self) -> list[JsonDict]:
        """Return the withheld events, in order, once the stream is trusted."""
        return [*self.buffered, *self.tail]


def parse_sse_line(line: str) -> JsonDict | None:
    """Decode one ``data:`` SSE line into an event, or ``None`` if not an event."""
    if not line.startswith(_SSE_PREFIX):
        return None
    body = line[len(_SSE_PREFIX) :].strip()
    if not body or body == _SSE_DONE:
        return None
    try:
        parsed: object = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    return cast("JsonDict", parsed) if isinstance(parsed, dict) else None


def _encode(event: JsonDict) -> bytes:
    return f"{_SSE_PREFIX}{json.dumps(event)}\n\n".encode()


def failure_event(model: str, detail: str) -> JsonDict:
    """Build a terminal chunk that reports an upstream failure in-band.

    Once SSE headers are sent the status code is fixed at 200, so the only way to
    make the failure visible is a final message. It is explicitly labelled as coming
    from the shim so it is never mistaken for the model's own words.
    """
    return {
        "id": "medmcp-shim-error",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": (
                        f"[medmcp: the local model server failed to return a usable "
                        f"response after retries — {detail}. This is an Ollama "
                        f"tool-call parsing failure, not a model refusal. Try again.]"
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    }


# ── proxy app ──────────────────────────────────────────────


def _inverse(renames: Mapping[str, str]) -> dict[str, str]:
    return {v: k for k, v in renames.items()}


def create_app(
    upstream: str,
    *,
    renames: Mapping[str, str] | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
) -> FastAPI:
    """Build the shim ASGI app forwarding to ``upstream``."""
    app = FastAPI(title="medmcp-llm-shim")
    mapping = dict(DEFAULT_RENAMES if renames is None else renames)
    inverse = _inverse(mapping)
    base = upstream.rstrip("/")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        try:
            raw: object = json.loads(await request.body())
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({"error": {"message": "invalid JSON"}}, status_code=400)
        if not isinstance(raw, dict):
            return JSONResponse({"error": {"message": "invalid payload"}}, status_code=400)

        payload = repair_request(cast("JsonDict", raw), mapping)
        model = str(payload.get("model") or "")
        url = f"{base}/v1/chat/completions"

        if payload.get("stream"):
            return StreamingResponse(
                _stream_with_retry(url, payload, inverse, attempts, model),
                media_type="text/event-stream",
            )
        return await _complete_with_retry(url, payload, inverse, attempts)

    @app.api_route(  # pyright: ignore[reportUntypedFunctionDecorator]
        "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"]
    )
    async def passthrough(request: Request, path: str) -> Response:  # pyright: ignore[reportUnusedFunction]
        """Forward every other Ollama route untouched, so the shim is a drop-in."""
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
            upstream_resp = await client.request(
                request.method,
                f"{base}/{path}",
                content=body,
                headers=headers,
                params=request.query_params,
            )
        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={k: v for k, v in upstream_resp.headers.items() if k.lower() not in excluded},
        )

    return app


async def _complete_with_retry(
    url: str, payload: JsonDict, inverse: Mapping[str, str], attempts: int
) -> Response:
    """Run a non-streaming completion, retrying parse failures and empty turns."""
    detail = "unknown error"
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        for attempt in range(1, attempts + 1):
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                detail = resp.text[:200]
                log.warning(
                    "shim: upstream %s on attempt %d/%d", resp.status_code, attempt, attempts
                )
                continue
            try:
                parsed: object = resp.json()
            except (json.JSONDecodeError, ValueError):
                detail = "upstream returned non-JSON"
                continue
            if not isinstance(parsed, dict):
                detail = "upstream returned a non-object"
                continue
            body = restore_response(cast("JsonDict", parsed), inverse)
            if _has_usable_choice(body):
                return JSONResponse(body)
            detail = "upstream returned an empty turn"
            log.warning("shim: empty turn on attempt %d/%d", attempt, attempts)
    return JSONResponse(
        {
            "error": {
                "message": f"medmcp-llm-shim: no usable response after {attempts} attempts "
                f"({detail})",
                "type": "api_error",
            }
        },
        status_code=502,
    )


def _has_usable_choice(body: JsonDict) -> bool:
    raw_choices = body.get("choices")
    if not isinstance(raw_choices, list):
        return False
    for raw_choice in cast("list[object]", raw_choices):
        if not isinstance(raw_choice, dict):
            continue
        raw_msg = cast("JsonDict", raw_choice).get("message")
        if not isinstance(raw_msg, dict):
            continue
        msg = cast("JsonDict", raw_msg)
        content = msg.get("content")
        if (isinstance(content, str) and content.strip()) or msg.get("tool_calls"):
            return True
    return False


async def _stream_with_retry(
    url: str,
    payload: JsonDict,
    inverse: Mapping[str, str],
    attempts: int,
    model: str,
) -> AsyncIterator[bytes]:
    """Relay a streaming completion, retrying attempts that truncate silently."""
    detail = "unknown error"
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        for attempt in range(1, attempts + 1):
            outcome = StreamOutcome()
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        detail = f"HTTP {resp.status_code}"
                        log.warning(
                            "shim: upstream %s on stream attempt %d/%d",
                            resp.status_code,
                            attempt,
                            attempts,
                        )
                        continue
                    async for line in resp.aiter_lines():
                        event = parse_sse_line(line)
                        if event is None:
                            continue
                        forward = outcome.feed(restore_response(event, inverse))
                        if forward is not None:
                            yield _encode(forward)
            except httpx.HTTPError as exc:
                detail = f"{type(exc).__name__}: {exc}"
                log.warning("shim: stream error on attempt %d/%d: %s", attempt, attempts, exc)
                continue

            if outcome.healthy:
                for event in outcome.release():
                    yield _encode(event)
                yield f"{_SSE_PREFIX}{_SSE_DONE}\n\n".encode()
                return

            detail = "stream truncated with no finish_reason"
            log.warning("shim: silent truncation on attempt %d/%d — retrying", attempt, attempts)

    yield _encode(failure_event(model, detail))
    yield f"{_SSE_PREFIX}{_SSE_DONE}\n\n".encode()


def _renames_from_env() -> dict[str, str]:
    """Parse ``MEDMCP_SHIM_RENAME`` (``old=new,old2=new2``); empty disables renaming."""
    spec = os.environ.get("MEDMCP_SHIM_RENAME")
    if spec is None:
        return dict(DEFAULT_RENAMES)
    out: dict[str, str] = {}
    for part in spec.split(","):
        if "=" not in part:
            continue
        old, new = part.split("=", 1)
        if old.strip() and new.strip():
            out[old.strip()] = new.strip()
    return out


def main() -> None:
    """Run the shim. Configured entirely by environment variables."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    upstream = (
        os.environ.get(_UPSTREAM_ENV) or os.environ.get("OLLAMA_BASE_URL") or _DEFAULT_UPSTREAM
    )
    host = os.environ.get("MEDMCP_SHIM_HOST", "127.0.0.1")
    port = int(os.environ.get("MEDMCP_SHIM_PORT", "11435"))
    attempts = int(os.environ.get("MEDMCP_SHIM_ATTEMPTS", str(DEFAULT_ATTEMPTS)))
    renames = _renames_from_env()
    log.info(
        "medmcp-llm-shim -> %s (renames=%s, attempts=%d)", upstream, renames or "none", attempts
    )
    uvicorn.run(
        create_app(upstream, renames=renames, attempts=attempts),
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
