"""LLM-generated chat titles.

A chat's transcript is condensed into a short title — "Skull-strip and register
subject 12" rather than the first prompt's opening words — by the same local
model that powers the agent, through a lightweight direct call like the tool-call
explanations in :mod:`medmcp.explain`.

vibe grows session titles itself since 2.24, but only for its own terminal UI:
the generation is gated on the client entrypoint, so an ACP client such as the
workspace never receives one. The policy here mirrors vibe's — a bounded number
of refreshes, a transcript window of the opening intent plus the latest
exchange, a tiny deterministic completion — so the two surfaces behave alike if
vibe's gate ever lifts.

UI-agnostic: the server owns *when* a refresh runs (:class:`TitleCadence`) and
where the result lands (the UI session registry, which keeps a user's own name
for a chat ahead of anything generated); this module owns the transcript window,
the request, and the cleanup of what comes back.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import cast

import httpx

from medmcp import provenance
from medmcp.acp import JsonDict
from medmcp.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from medmcp.workspace_note import display_content_text, strip_workspace_note

_audit: logging.Logger = logging.getLogger("medmcp.audit")

# Kill switch for deployments that would rather not spend model time on names.
DISABLE_ENV_VAR = "MEDMCP_AUTO_TITLES"

_ELISION = "\n\n[…]\n\n"
_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
# Straight, backtick, and curly quotes a model may wrap its answer in.
_WRAPPING_QUOTES = "\"'`\u201c\u201d\u2018\u2019"

SYSTEM_PROMPT = (
    "You write short, descriptive titles for chat sessions between a clinician or "
    "researcher and MedMCP, an AI assistant that runs medical-imaging analysis tools "
    "on their behalf. Given a transcript, reply with a concise title naming what the "
    "session is about.\n\n"
    "Rules:\n"
    "- 3 to 8 words. No trailing period.\n"
    "- Name the task or the data, not the request: describe what is being worked on, "
    "not that the user asked for it.\n"
    "- Prefer specific nouns from the transcript — modality, pipeline step, subject, "
    'file — over generic phrases. "Skull-strip and register subject 12" beats '
    '"Process an image".\n'
    "- Plain text only, in sentence case. No quotes, backticks, markdown, code fences, "
    "or emoji.\n"
    "- Always answer in English; translate the intent of a transcript in another "
    "language rather than transliterating it.\n"
    "- If a `Current title:` is given, keep it unless the session's focus has clearly "
    "shifted, in which case refine it.\n"
    "- If the transcript is empty or describes no task, answer `New chat`.\n\n"
    "Respond with ONLY the title, on one line, with no quotes or explanation."
)


@dataclass(frozen=True, slots=True)
class TitlePolicy:
    """The tunable numbers behind generated titles, in one place.

    Cadence counts *completed turns* (the server sees turns, not model steps):
    the first title lands when the opening turn finishes, then a refresh every
    ``refresh_every_turns`` turns, ``max_generations`` calls per connection in
    total — the title runs on the same single-GPU model as the conversation, so
    it stays bounded rather than periodic-forever. A generation that yields
    nothing hands its slot back so the next turn retries; the attempt still
    counts, so a failing model cannot be retried without bound.
    """

    refresh_every_turns: int = 5
    max_generations: int = 3
    # Transcript window: the opening intent plus the latest exchange, each
    # message clamped so one large tool result can't crowd the rest out.
    max_transcript_chars: int = 6000
    head_transcript_chars: int = 1500
    max_message_chars: int = 2000
    max_tool_result_chars: int = 400
    # Request budget and response size for the background call.
    timeout_seconds: float = 20.0
    max_tokens: int = 96
    # Result shaping: hard length cap and answers treated as "nothing usable".
    max_title_chars: int = 72
    generic_titles: frozenset[str] = frozenset({"new chat", "new session", "untitled"})

    @property
    def tail_transcript_chars(self) -> int:
        """Characters kept from the end of an over-long transcript."""
        return self.max_transcript_chars - self.head_transcript_chars


DEFAULT_POLICY = TitlePolicy()


def enabled() -> bool:
    """Whether generated titles are on (default) — ``MEDMCP_AUTO_TITLES=0`` turns them off."""
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class TitleTicket:
    """A due title generation. Hand it back to :meth:`TitleCadence.restore` if it doesn't land."""

    prev_turn: int


class TitleCadence:
    """Decides when a title refresh is due, one instance per chat connection."""

    def __init__(self, policy: TitlePolicy = DEFAULT_POLICY) -> None:
        """Start with no turns seen and the initial title pending."""
        self._policy = policy
        self._turns = 0
        self._last_gen_turn = 0
        self._generations = 0

    def begin_if_due(self) -> TitleTicket | None:
        """Record one completed turn; return a ticket when a refresh is due now."""
        self._turns += 1
        if self._generations >= self._policy.max_generations:
            return None
        initial_pending = self._last_gen_turn == 0
        due = initial_pending or (
            self._turns - self._last_gen_turn >= self._policy.refresh_every_turns
        )
        if not due:
            return None
        ticket = TitleTicket(prev_turn=self._last_gen_turn)
        self._last_gen_turn = self._turns
        self._generations += 1
        return ticket

    def restore(self, ticket: TitleTicket) -> None:
        """Reschedule after a generation that produced nothing (the attempt stays counted)."""
        self._last_gen_turn = ticket.prev_turn


def load_session_messages(session_id: str, *, stop_ids: Collection[str] = ()) -> list[JsonDict]:
    """Read a chat's transcript from vibe's ``messages.jsonl`` (empty if not found).

    Concatenated across the session's compaction chain, like distillation, so a
    chat resumed across a pre-2.24 compaction still titles off its whole history;
    ``stop_ids`` keeps the walk out of forks (see :func:`provenance.find_vibe_session_dirs`).
    """
    from medmcp.distill import parse_messages_file  # local: distill pulls in yaml et al.

    messages: list[JsonDict] = []
    for session_dir in provenance.find_vibe_session_dirs(session_id, stop_ids=stop_ids):
        path = session_dir / "messages.jsonl"
        if path.exists():
            messages.extend(parse_messages_file(path))
    return messages


def _message_text(msg: JsonDict, policy: TitlePolicy) -> str | None:
    """Render one transcript message as ``role: text``, or ``None`` to skip it."""
    role = msg.get("role")
    if role == "system":
        return None
    if role == "user":
        # Skip what the harness injected (the compaction summary envelope, a
        # /retry continuation) — not something the person said.
        if msg.get("injected") or msg.get("context_boundary"):
            return None
        display = msg.get("user_display_content")
        text = display_content_text(cast("JsonDict", display)) if isinstance(display, dict) else ""
        if not text:
            content = msg.get("content")
            text = strip_workspace_note(content) if isinstance(content, str) else ""
        return f"user: {text.strip()}" if text.strip() else None
    if role == "assistant":
        parts: list[str] = []
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip()[: policy.max_message_chars])
        calls = msg.get("tool_calls")
        names: list[str] = []
        if isinstance(calls, list):
            for call in cast("list[object]", calls):
                if not isinstance(call, dict):
                    continue
                fn = cast("JsonDict", call).get("function")
                if not isinstance(fn, dict):
                    continue
                raw_name = str(cast("JsonDict", fn).get("name") or "")
                # [] leaves the medmcp-<stack>_<tool> convention (and "builtin"
                # for vibe's own tools) to strip the server prefix.
                tool = provenance.split_tool_name(raw_name, [])[1]
                if tool:
                    names.append(tool)
        if names:
            parts.append(f"(ran {', '.join(names)})")
        return f"assistant: {' '.join(parts)}" if parts else None
    if role == "tool":
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return f"tool: {content.strip()[: policy.max_tool_result_chars]}"
        return None
    return None


def build_transcript(messages: Sequence[JsonDict], *, policy: TitlePolicy = DEFAULT_POLICY) -> str:
    """Condense raw transcript messages into the window the model sees.

    Every message is clamped, and an over-long transcript keeps its head (the
    opening intent) and tail (the latest exchange) with the middle elided.
    """
    blocks: list[str] = []
    for msg in messages:
        rendered = _message_text(msg, policy)
        if rendered is None:
            continue
        blocks.append(rendered[: policy.max_message_chars])
    transcript = "\n\n".join(blocks).strip()
    if len(transcript) <= policy.max_transcript_chars:
        return transcript
    head = transcript[: policy.head_transcript_chars].rstrip()
    tail = transcript[-policy.tail_transcript_chars :].lstrip()
    return f"{head}{_ELISION}{tail}"


def _user_prompt(transcript: str, previous_title: str | None) -> str:
    if not previous_title:
        return transcript
    return f"Current title: {previous_title}\n\nTranscript:\n{transcript}"


def clean_title(content: str | None, *, policy: TitlePolicy = DEFAULT_POLICY) -> str | None:
    """Shape a raw model answer into a title, or ``None`` when nothing usable came back."""
    if not content:
        return None
    stripped = content.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    first_line = _CONTROL_CHARS_RE.sub("", first_line)
    collapsed = _WHITESPACE_RE.sub(" ", first_line).strip().strip(_WRAPPING_QUOTES).strip()
    if not collapsed or collapsed.lower().rstrip(".") in policy.generic_titles:
        return None
    if len(collapsed) > policy.max_title_chars:
        collapsed = collapsed[: policy.max_title_chars].rstrip() + "…"
    return collapsed


async def generate_title(
    messages: Sequence[JsonDict],
    *,
    previous_title: str | None = None,
    policy: TitlePolicy = DEFAULT_POLICY,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Ask the local model for a title describing *messages*.

    ``previous_title`` is fed back so the model refines an earlier title rather
    than starting over. Returns ``None`` when there is nothing to title yet, the
    model gave no usable answer, or the call failed — failures are logged, never
    raised: a title is a nicety and must not disturb the chat.
    """
    transcript = build_transcript(messages, policy=policy)
    if not transcript:
        return None
    payload: JsonDict = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(transcript, previous_title)},
        ],
        "stream": False,
        # Native /api/chat like explain.py: the OpenAI-compatible endpoint
        # ignores think:false and a thinking model then answers in "reasoning"
        # only. Deterministic — a title should not wander between refreshes.
        "think": False,
        "options": {"temperature": 0.0, "num_predict": policy.max_tokens},
    }
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=policy.timeout_seconds) as own:
                raw = await _post_chat(own, payload)
        else:
            raw = await _post_chat(client, payload)
    except Exception:
        _audit.warning("failed to generate a chat title", exc_info=True)
        return None
    return clean_title(raw, policy=policy)


async def _post_chat(client: httpx.AsyncClient, payload: JsonDict) -> str:
    resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    resp.raise_for_status()
    data = cast("JsonDict", resp.json())
    message = cast("JsonDict", data.get("message") or {})
    return str(message.get("content") or "")
