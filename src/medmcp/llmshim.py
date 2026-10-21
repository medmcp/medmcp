"""``medmcp-llm-shim`` — a reverse proxy that repairs Ollama's Glimmer tool calls.

Ollama's Muse Glimmer function-calling parser drops tool calls. Measured against
the real workspace system prompt and tool list, ~50% of streaming requests fail;
the failure mode differs by transport, and the streaming one is silent:

* **non-streaming** → ``HTTP 500 parse Glimmer call to <tool>: malformed ATEM parameter``
* **streaming** → the stream is truncated with **no** ``finish_reason``, no ``usage``
  chunk, and an empty ``model`` on the final chunk. HTTP status is already 200, so
  no client can tell the turn failed. vibe records an empty assistant message and
  the chat appears frozen.

This shim sits between vibe and Ollama and applies these repairs:

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
4. **Own tool-call rendering and parsing** (``MEDMCP_SHIM_TOOLS=glimmer``, the
   default). With vibe 2.26's tool set Ollama's parser fails on nearly every call
   (measured: ~100% on 0.33.2 and 0.40.6 alike), while the model's own ATEM text is
   well-formed every time. So the shim renders the ``tools`` into the system prompt
   exactly as Ollama's Glimmer renderer would, forwards the request **without**
   ``tools`` — which leaves Ollama's parser inactive and the model's
   ``<atem:function_calls>`` block in the content stream — and parses that block
   itself into OpenAI ``tool_calls``. Values are typed from the tool's JSON schema.
   ``MEDMCP_SHIM_TOOLS=ollama`` restores Ollama's parser (repairs 1-3 still apply).
5. **Tool-name repair** (glimmer mode). The rendered tool section — Ollama's and
   the publisher's template alike — shows ``example_tool_name.example_function_name``
   as its example, and Muse copies the shape: ``bash.action``, ``todo.action``
   (32 of 100 replayed agent requests, measured 2026-10-21). A name that is not
   offered but whose part before the dot (``bash.action``) or after it
   (``file_system.bash``) is exactly one offered tool is mapped to that tool; any
   other unknown name is kept, and vibe answers it with its own error.

**Relay mode** (``MEDMCP_SHIM_RELAY``). ``live`` (default): reasoning and
``content`` deltas stream straight through, so the answer appears as it is
written; only ``tool_calls`` deltas are withheld until the turn is known to be
healthy — they are not user-visible while streaming, and the parser failure
above strikes exactly there, so the transparent retry still covers it. A
truncation *after* visible text has gone out cannot be retried without the
person seeing the answer twice, so the shim ends the stream without a
``finish_reason``: vibe ≥2.24 treats that as an incomplete stream, keeps the
partial text as an interrupted answer, and offers ``/retry`` to continue.
``buffered``: the original behaviour — ``content`` is withheld as well and every
truncation is retried invisibly, at the cost of the answer arriving all at once.
If every attempt fails the shim emits an explicit, clearly-labelled error message
rather than an empty turn — a visible failure beats a silent one.

Inert until ``api_base`` in ``.vibe/config.toml`` points at it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import time
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
# How much of a streamed turn is relayed before it is known healthy (see module doc).
RELAY_MODES: tuple[str, ...] = ("live", "buffered")
DEFAULT_RELAY: str = "live"
_RELAY_ENV = "MEDMCP_SHIM_RELAY"

_UPSTREAM_ENV = "MEDMCP_SHIM_UPSTREAM"
_DEFAULT_UPSTREAM = "http://llm:11434"

# Who parses the model's tool calls: the shim (Glimmer ATEM rendered and parsed here)
# or Ollama. See repair 4 in the module doc.
TOOL_MODES: tuple[str, ...] = ("glimmer", "ollama")
DEFAULT_TOOL_MODE: str = "glimmer"
_TOOL_MODE_ENV = "MEDMCP_SHIM_TOOLS"

# While the shim holds output back (a tool-call block being written, the whole
# answer in buffered relay) vibe sees silence, and vibe ≥2.26 fails a stream that
# is silent for ``api_stream_idle_timeout``. An SSE comment every so often keeps
# the connection visibly alive without adding to the answer. 0 disables it.
DEFAULT_KEEPALIVE_SEC: float = 15.0
_KEEPALIVE_ENV = "MEDMCP_SHIM_KEEPALIVE"
_SSE_KEEPALIVE = b": keepalive\n\n"

# ATEM markers, as in Ollama's ``model/parsers/glimmer.go``.
ATEM_CALLS_OPEN = "<atem:function_calls>"
ATEM_CALLS_CLOSE = "</atem:function_calls>"
_ATEM_INVOKE_RE = re.compile(r'<atem:invoke name="([^"]*)">(.*?)</atem:invoke>', re.DOTALL)
_ATEM_PARAM_RE = re.compile(r'<atem:parameter name="([^"]*)">(.*?)</atem:parameter>', re.DOTALL)

# The tool section Ollama's Glimmer renderer writes into the system message
# (``model/renderers/glimmer.go``); reproduced verbatim so the model sees what it
# was trained on.
_GLIMMER_TOOLS_PREFIX = (
    "In this environment you have access to a set of tools you can use to answer the "
    "user's question.\n"
    "\n"
    'You can invoke a function by writing a "<atem:function_calls>" block like the '
    "following:\n"
    "<atem:function_calls>\n"
    '<atem:invoke name="$FUNCTION_NAME">\n'
    '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n'
    "...\n"
    "</atem:invoke>\n"
    "</atem:function_calls>\n"
    "\n"
    "String and scalar parameters should be specified as is, while lists and objects "
    "should use JSON format. Note that spaces for string values are not stripped. The "
    "output is not expected to be valid XML and is parsed with regular expressions.\n"
    "Here are the functions available in JSONSchema format:\n"
    "// Tool metadata\n"
)
_GLIMMER_TOOLS_SUFFIX = (
    "\n"
    "\n"
    "Here's an example of how to call a function in the tool set:\n"
    "(If the tool namespace is not specified, invoke the function directly as "
    "`example_function_name` rather than `example_tool_name.example_function_name`)\n"
    "\n"
    "to=example_tool_name.example_function_name\n"
    "\n"
    "<atem:function_calls>\n"
    '<atem:invoke name="example_tool_name.example_function_name">\n'
    '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
    '<atem:parameter name="example_parameter_2">This is the value for the second '
    "parameter\n"
    "that can span\n"
    '"multiple" lines\n'
    "</atem:parameter>\n"
    "</atem:invoke>\n"
    "</atem:function_calls>"
)


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


# ── Glimmer tool calls: render and parse (pure) ────────────


def _tool_functions(payload: JsonDict) -> list[JsonDict]:
    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, list):
        return []
    out: list[JsonDict] = []
    for raw_tool in cast("list[object]", raw_tools):
        if not isinstance(raw_tool, dict):
            continue
        raw_fn = cast("JsonDict", raw_tool).get("function")
        if isinstance(raw_fn, dict) and isinstance(cast("JsonDict", raw_fn).get("name"), str):
            out.append(cast("JsonDict", raw_fn))
    return out


def _glimmer_json(value: object) -> str:
    # Go's encoder with SetEscapeHTML(false) plus the Transformers tojson spacing.
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def render_tool_section(functions: list[JsonDict]) -> str:
    """The tool definitions as Ollama's Glimmer renderer writes them into the prompt.

    A namespace is the part of a name before its first ``.`` (the whole name for
    ours); each gets a metadata line, then every function its JSON schema.
    """
    namespaces: list[str] = []
    for fn in functions:
        namespace = str(fn["name"]).split(".", 1)[0]
        if namespace not in namespaces:
            namespaces.append(namespace)
    parts = [_GLIMMER_TOOLS_PREFIX]
    for namespace in namespaces:
        parts.append(f'{{"name": {_glimmer_json(namespace)}, "description": ""}}\n')
    parts.append("// Function schemas")
    for fn in functions:
        parts.append(
            f'\n{{"name": {_glimmer_json(fn["name"])}, '
            f'"description": {_glimmer_json(fn.get("description") or "")}, '
            f'"parameters": {_glimmer_json(fn.get("parameters") or {})}}}'
        )
    parts.append(_GLIMMER_TOOLS_SUFFIX)
    return "".join(parts)


def glimmer_request(payload: JsonDict) -> tuple[JsonDict, dict[str, JsonDict]]:
    """Move ``tools`` into the system message so Ollama's parser stays out of the way.

    Returns the request to forward (no ``tools``/``tool_choice``, the tool section
    appended to the system message — one is created if the request has none) and
    the tools by name, for typing the parsed arguments. A request without tools is
    returned unchanged.
    """
    functions = _tool_functions(payload)
    if not functions:
        return payload, {}
    out = cast("JsonDict", json.loads(json.dumps(payload)))
    out.pop("tools", None)
    out.pop("tool_choice", None)
    section = render_tool_section(functions)
    raw_messages = out.get("messages")
    messages = cast("list[object]", raw_messages) if isinstance(raw_messages, list) else []
    for raw_message in messages:
        if isinstance(raw_message, dict) and cast("JsonDict", raw_message).get("role") == "system":
            message = cast("JsonDict", raw_message)
            content = message.get("content")
            text = content if isinstance(content, str) else ""
            message["content"] = f"{text}\n\n{section}" if text else section
            break
    else:
        messages.insert(0, {"role": "system", "content": section})
        out["messages"] = messages
    return out, {str(fn["name"]): fn for fn in functions}


def _schema_types(fn: JsonDict | None, param: str) -> set[str]:
    if fn is None:
        return set()
    params = fn.get("parameters")
    props = cast("JsonDict", params).get("properties") if isinstance(params, dict) else None
    prop = cast("JsonDict", props).get(param) if isinstance(props, dict) else None
    if not isinstance(prop, dict):
        return set()
    prop_dict = cast("JsonDict", prop)
    types: set[str] = set()
    raw_any = prop_dict.get("anyOf")
    options = cast("list[object]", raw_any) if isinstance(raw_any, list) else [prop_dict]
    for option in options:
        if not isinstance(option, dict):
            continue
        declared = cast("JsonDict", option).get("type")
        if isinstance(declared, str):
            types.add(declared)
        elif isinstance(declared, list):
            types.update(str(t) for t in cast("list[object]", declared))
    return types


def coerce_value(raw: str, types: set[str]) -> object:
    """Type a parameter's text the way Ollama's ``parseTypedToolValue`` does.

    Scalars are converted when the schema asks for them, lists and objects are
    parsed as JSON, and anything that does not convert stays a string — the model
    wrote it, so the tool gets to see it rather than a guess.
    """
    text = raw
    stripped = text.strip()
    if "null" in types and stripped == "null":
        return None
    if "string" in types and not (types - {"string", "null"}):
        return text
    if {"integer", "number"} & types:
        try:
            return int(stripped) if "integer" in types and "." not in stripped else float(stripped)
        except ValueError:
            pass
    if "boolean" in types and stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    if ({"array", "object"} & types or not types) and stripped[:1] in "[{":
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            return json.loads(stripped)
    return text


def resolve_tool_name(name: str, functions: Mapping[str, JsonDict]) -> str:
    """The offered tool a namespaced name means (repair 5), else *name* unchanged.

    ``bash.action`` -> ``bash`` and ``file_system.bash`` -> ``bash`` when ``bash``
    is offered. A name whose two halves are both offered tools is ambiguous and
    kept, like any name that matches nothing.
    """
    if name in functions or "." not in name:
        return name
    first, last = name.partition(".")[0], name.rpartition(".")[2]
    candidates = {part for part in (first, last) if part in functions}
    if len(candidates) != 1:
        return name
    (resolved,) = candidates
    log.info("tool name %r repaired to %r", name, resolved)
    return resolved


def parse_atem_block(
    block: str, functions: Mapping[str, JsonDict], *, first_index: int = 0
) -> list[JsonDict]:
    """Parse one ``<atem:function_calls>`` block into OpenAI tool-call objects.

    Lenient where Ollama's parser is not: whitespace between tags is ignored and
    an invoke with no parameters is a call with ``{}``. A namespaced name is
    mapped to the offered tool it means (:func:`resolve_tool_name`); other unknown
    names are kept — vibe answers the model with its own error, which is better
    than a dropped call. ``first_index`` numbers the calls after those of an
    earlier block in the same turn: clients merge streamed deltas by index.
    """
    calls: list[JsonDict] = []
    for index, invoke in enumerate(_ATEM_INVOKE_RE.finditer(block), start=first_index):
        name = resolve_tool_name(invoke.group(1).strip(), functions)
        fn = functions.get(name)
        arguments: JsonDict = {}
        for param in _ATEM_PARAM_RE.finditer(invoke.group(2)):
            arguments[param.group(1)] = coerce_value(
                param.group(2), _schema_types(fn, param.group(1))
            )
        calls.append(
            {
                "index": index,
                "id": f"call_{secrets.token_hex(6)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
    return calls


class AtemSplitter:
    """Separate the model's ATEM block from the text around it, chunk by chunk.

    Text before the block is released as it arrives, minus the shortest suffix
    that could be the start of the opening marker (the marker can straddle two
    deltas). From the marker on everything is held; when the closing marker
    arrives the block is complete and ``calls`` holds the parsed tool calls.
    """

    def __init__(self, functions: Mapping[str, JsonDict]) -> None:
        """Start a splitter for one stream, typing arguments from ``functions``."""
        self._functions = functions
        self._held = ""
        self._in_block = False
        self.calls: list[JsonDict] = []

    @property
    def open(self) -> bool:
        """Whether a block has started but not closed yet."""
        return self._in_block

    def feed(self, text: str) -> str:
        """Absorb one content delta; return the text safe to release now."""
        self._held += text
        out = ""
        while True:
            if not self._in_block:
                start = self._held.find(ATEM_CALLS_OPEN)
                if start < 0:
                    keep = _partial_suffix(self._held, ATEM_CALLS_OPEN)
                    cut = len(self._held) - keep
                    out, self._held = out + self._held[:cut], self._held[cut:]
                    return out
                out, self._held = out + self._held[:start], self._held[start:]
                self._in_block = True
            end = self._held.find(ATEM_CALLS_CLOSE)
            if end < 0:
                return out
            block = self._held[: end + len(ATEM_CALLS_CLOSE)]
            self.calls.extend(parse_atem_block(block, self._functions, first_index=len(self.calls)))
            self._held = self._held[end + len(ATEM_CALLS_CLOSE) :]
            self._in_block = False
            # Text after a block (rare) is treated like any other text: loop.

    def flush(self) -> str:
        """Release whatever is still held at the end of the stream."""
        out, self._held = self._held, ""
        return "" if self._in_block else out


def _partial_suffix(text: str, marker: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``marker``."""
    for n in range(min(len(text), len(marker) - 1), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0


# ── stream classification ──────────────────────────────────


class StreamOutcome:
    """Accumulates one upstream SSE stream and judges whether it completed.

    ``tool_calls`` chunks are withheld until the stream is known to be healthy, so
    a truncated attempt can be retried without the caller having seen a partial
    call. Reasoning chunks are released immediately — they are display-only and
    replaying them across attempts is harmless. ``content`` chunks stream through
    in ``live`` mode (``relayed_visible`` then records that a retry would show the
    answer twice) and are withheld in ``buffered`` mode.
    """

    def __init__(
        self, relay: str = DEFAULT_RELAY, *, functions: Mapping[str, JsonDict] | None = None
    ) -> None:
        """Start an empty outcome for one upstream attempt.

        With ``functions`` the shim parses the model's ATEM block out of the content
        stream itself (see :class:`AtemSplitter`); without, Ollama's tool calls are
        relayed as they come.
        """
        if relay not in RELAY_MODES:
            raise ValueError(f"unknown relay mode {relay!r}; expected one of {RELAY_MODES}")
        self.relay = relay
        self.saw_finish_reason: bool = False
        # Whether answer text the person can see has already been relayed.
        self.relayed_visible: bool = False
        self.buffered: list[JsonDict] = []
        self.tail: list[JsonDict] = []
        self._splitter = AtemSplitter(functions) if functions is not None else None
        self._last_chunk_shape: JsonDict | None = None

    @property
    def healthy(self) -> bool:
        """Whether the stream terminated normally.

        A missing ``finish_reason`` is the abnormal-termination signal: healthy turns
        always carry one (``"tool_calls"`` or ``"stop"``), truncated ones never do.
        A stream that ends inside an unterminated ATEM block is truncated too.
        """
        if self._splitter is not None and self._splitter.open:
            return False
        return self.saw_finish_reason

    def feed(self, event: JsonDict) -> JsonDict | None:
        """Absorb one SSE event; return an event to forward now, or ``None``."""
        if self._splitter is not None:
            split = self._split_atem(event)
            if split is None:
                return None
            event = split
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
            has_content = bool(delta.get("content"))
            has_calls = bool(delta.get("tool_calls"))
            # Reasoning-only frames stream straight through in every mode.
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning and not has_content and not has_calls:
                passthrough = True
            # Answer text streams through live; tool-call deltas never do.
            if self.relay == "live" and has_content and not has_calls:
                passthrough = True
                self.relayed_visible = True

        if passthrough:
            return event
        self.buffered.append(event)
        return None

    def release(self) -> list[JsonDict]:
        """Return the withheld events, in order, once the stream is trusted."""
        return [*self.buffered, *self.tail]

    def _split_atem(self, event: JsonDict) -> JsonDict | None:
        """Route content deltas through the splitter; turn a finished block into calls.

        Ollama's own ``finish_reason`` (``stop``, since it saw plain text) is rewritten
        to ``tool_calls`` when a block was parsed, and the calls are emitted as one
        delta just before it, the way the OpenAI stream shape expects.
        """
        assert self._splitter is not None
        raw_choices = event.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            return event
        choice0 = cast("list[object]", raw_choices)[0]
        if not isinstance(choice0, dict):
            return event
        choice = cast("JsonDict", choice0)
        self._last_chunk_shape = {k: v for k, v in event.items() if k != "choices"}
        raw_delta = choice.get("delta")
        delta = cast("JsonDict", raw_delta) if isinstance(raw_delta, dict) else {}
        content = delta.get("content")
        finish = choice.get("finish_reason")
        if isinstance(content, str) and content:
            delta["content"] = self._splitter.feed(content)
        if finish:
            tail_text = self._splitter.flush()
            if tail_text:
                delta["content"] = (delta.get("content") or "") + tail_text
            if self._splitter.calls:
                delta["tool_calls"] = self._splitter.calls
                delta.pop("content", None) if not delta.get("content") else None
                choice["finish_reason"] = "tool_calls"
            return event
        # A delta emptied by the splitter carries nothing worth relaying.
        if "content" in delta and not delta["content"]:
            delta.pop("content")
        if not delta or set(delta) <= {"role"}:
            return None
        return event


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
    relay: str = DEFAULT_RELAY,
    tool_mode: str = DEFAULT_TOOL_MODE,
    keepalive_sec: float = DEFAULT_KEEPALIVE_SEC,
) -> FastAPI:
    """Build the shim ASGI app forwarding to ``upstream``."""
    if relay not in RELAY_MODES:
        raise ValueError(f"unknown relay mode {relay!r}; expected one of {RELAY_MODES}")
    if tool_mode not in TOOL_MODES:
        raise ValueError(f"unknown tool mode {tool_mode!r}; expected one of {TOOL_MODES}")
    app = FastAPI(title="medmcp-llm-shim")
    # Renaming only matters while Ollama parses the calls; the shim's own parser
    # never sees a recipient namespace.
    mapping = (
        {} if tool_mode == "glimmer" else dict(DEFAULT_RENAMES if renames is None else renames)
    )
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
        functions: dict[str, JsonDict] | None = None
        if tool_mode == "glimmer":
            payload, functions = glimmer_request(payload)
            if not functions:
                functions = None

        if payload.get("stream"):
            return StreamingResponse(
                _stream_with_retry(
                    url,
                    payload,
                    inverse,
                    attempts,
                    model,
                    relay=relay,
                    functions=functions,
                    keepalive_sec=keepalive_sec,
                ),
                media_type="text/event-stream",
            )
        return await _complete_with_retry(url, payload, inverse, attempts, functions=functions)

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


def parse_completion(body: JsonDict, functions: Mapping[str, JsonDict]) -> JsonDict:
    """Turn an ATEM block in a non-streaming answer into ``tool_calls``."""
    raw_choices = body.get("choices")
    if not isinstance(raw_choices, list):
        return body
    for raw_choice in cast("list[object]", raw_choices):
        if not isinstance(raw_choice, dict):
            continue
        choice = cast("JsonDict", raw_choice)
        raw_msg = choice.get("message")
        if not isinstance(raw_msg, dict):
            continue
        msg = cast("JsonDict", raw_msg)
        content = msg.get("content")
        if not isinstance(content, str) or ATEM_CALLS_OPEN not in content:
            continue
        splitter = AtemSplitter(functions)
        text = splitter.feed(content) + splitter.flush()
        if splitter.calls:
            msg["tool_calls"] = splitter.calls
            msg["content"] = text or None
            choice["finish_reason"] = "tool_calls"
    return body


async def _complete_with_retry(
    url: str,
    payload: JsonDict,
    inverse: Mapping[str, str],
    attempts: int,
    *,
    client: httpx.AsyncClient | None = None,
    functions: Mapping[str, JsonDict] | None = None,
) -> Response:
    """Run a non-streaming completion, retrying parse failures and empty turns.

    A 4xx is the upstream rejecting the request itself; it is passed through at
    once, since re-sending the same request can only produce the same answer.
    ``client`` lets tests inject a transport.
    """
    detail = "unknown error"
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(900.0))
        if client is None
        else contextlib.nullcontext(client)
    ) as client:
        for attempt in range(1, attempts + 1):
            resp = await client.post(url, json=payload)
            if 400 <= resp.status_code < 500:
                log.warning("shim: upstream %s — a client error, not retried", resp.status_code)
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"),
                )
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
            if functions is not None:
                body = parse_completion(body, functions)
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
    *,
    relay: str = DEFAULT_RELAY,
    client: httpx.AsyncClient | None = None,
    functions: Mapping[str, JsonDict] | None = None,
    keepalive_sec: float = DEFAULT_KEEPALIVE_SEC,
) -> AsyncIterator[bytes]:
    """Relay a streaming completion, retrying attempts that truncate silently.

    In ``live`` relay a retry is only possible while no answer text has gone out;
    a truncation after that ends the stream without ``finish_reason`` (and without
    ``[DONE]``) so the caller sees an incomplete stream rather than a silent or a
    duplicated answer. ``client`` lets tests inject a transport. While upstream
    chunks arrive but nothing is forwarded for ``keepalive_sec``, an SSE comment
    goes out so the caller's idle timer does not fire on a held block.
    """
    detail = "unknown error"
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(900.0))
        if client is None
        else contextlib.nullcontext(client)
    ) as client:
        for attempt in range(1, attempts + 1):
            outcome = StreamOutcome(relay, functions=functions)
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        detail = f"HTTP {resp.status_code}"
                        if 400 <= resp.status_code < 500:
                            log.warning("shim: upstream %s — a client error, not retried", detail)
                            break
                        log.warning(
                            "shim: upstream %s on stream attempt %d/%d",
                            resp.status_code,
                            attempt,
                            attempts,
                        )
                        continue
                    last_out = time.monotonic()
                    async for line in resp.aiter_lines():
                        event = parse_sse_line(line)
                        if event is None:
                            continue
                        forward = outcome.feed(restore_response(event, inverse))
                        if forward is not None:
                            yield _encode(forward)
                            last_out = time.monotonic()
                        elif keepalive_sec > 0 and time.monotonic() - last_out >= keepalive_sec:
                            yield _SSE_KEEPALIVE
                            last_out = time.monotonic()
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
            if outcome.relayed_visible:
                # The person has seen part of the answer: a retry would show it
                # twice, so hand the truncation to the caller as what it is.
                log.warning(
                    "shim: truncation after visible text on attempt %d — "
                    "ending the stream as incomplete",
                    attempt,
                )
                return
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
    relay = os.environ.get(_RELAY_ENV, DEFAULT_RELAY).strip().lower() or DEFAULT_RELAY
    if relay not in RELAY_MODES:
        log.warning("shim: unknown %s=%r, using %r", _RELAY_ENV, relay, DEFAULT_RELAY)
        relay = DEFAULT_RELAY
    tool_mode = (
        os.environ.get(_TOOL_MODE_ENV, DEFAULT_TOOL_MODE).strip().lower() or DEFAULT_TOOL_MODE
    )
    if tool_mode not in TOOL_MODES:
        log.warning("shim: unknown %s=%r, using %r", _TOOL_MODE_ENV, tool_mode, DEFAULT_TOOL_MODE)
        tool_mode = DEFAULT_TOOL_MODE
    renames = _renames_from_env() if tool_mode == "ollama" else {}
    try:
        keepalive_sec = float(os.environ.get(_KEEPALIVE_ENV, str(DEFAULT_KEEPALIVE_SEC)))
    except ValueError:
        log.warning("shim: unreadable %s, using %s", _KEEPALIVE_ENV, DEFAULT_KEEPALIVE_SEC)
        keepalive_sec = DEFAULT_KEEPALIVE_SEC
    log.info(
        "medmcp-llm-shim -> %s (tools=%s, renames=%s, attempts=%d, relay=%s)",
        upstream,
        tool_mode,
        renames or "none",
        attempts,
        relay,
    )
    uvicorn.run(
        create_app(
            upstream,
            renames=renames,
            attempts=attempts,
            relay=relay,
            tool_mode=tool_mode,
            keepalive_sec=keepalive_sec,
        ),
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
