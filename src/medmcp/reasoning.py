"""Strip a local model's inline reasoning + stray control tokens from agent text.

Two things leak from locally-served models into the chat content stream and must
be removed before relaying to the UI:

- ``<thought>...</thought>`` spans — some models wrap chain-of-thought this way in
  the normal message content instead of a separate reasoning channel.
- **harmony-style control-token fragments** — e.g. ``<|channel|>``, ``<|message|>``,
  ``<|start|>``, ``<|end|>`` (and mangled forms like ``<channel|>`` when the server
  half-parses them). These are structural tokens, never meant to be shown.

:class:`ThoughtStripper` removes both from a token stream. It is streaming-aware:
a tag or token may be split across feeds, so it holds back only the shortest
possible partial-construct suffix and emits everything else.

NOTE: this does not hide reasoning that a harmony model emits as *analysis-channel
prose* (paragraphs, not tokens) — that needs the model/Ollama layer to route
reasoning into a ``reasoning_content`` field so vibe tags it ``agent_thought_chunk``.
This strips the structural tokens and ``<thought>`` spans only.
"""

from __future__ import annotations

import re

_THOUGHT_OPEN = "<thought>"
_THOUGHT_CLOSE = "</thought>"
# A complete special construct: a <thought> open/close tag, or a harmony control
# token. Control tokens always carry a pipe (<|channel|>, or a half-parsed
# <|channel> / <channel|>), so requiring one avoids eating real markup like <div>.
_SPECIAL = re.compile(r"</?thought>|<\|[a-zA-Z_]+\|?>|<[a-zA-Z_]+\|>")
# A trailing fragment that could still grow into one of the above on the next feed
# (a '<', optional '/'|'|', word chars, optional '|', with no closing '>' yet).
_PARTIAL = re.compile(r"<[/|]?[a-zA-Z_]*\|?$")


def _held_tail(buf: str, tag: str) -> int:
    """Length of the longest suffix of *buf* that is a proper prefix of *tag*."""
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


class ThoughtStripper:
    """Streaming remover of ``<thought>`` spans and harmony control tokens."""

    def __init__(self) -> None:
        """Start with an empty buffer, outside any thought span."""
        self._buf = ""
        self._in_thought = False

    def feed(self, text: str) -> str:
        """Consume streamed *text*; return it with thought spans + control tokens removed."""
        self._buf += text
        out: list[str] = []
        while True:
            if self._in_thought:
                idx = self._buf.find(_THOUGHT_CLOSE)
                if idx == -1:
                    k = _held_tail(self._buf, _THOUGHT_CLOSE)
                    self._buf = self._buf[len(self._buf) - k :]
                    break
                self._buf = self._buf[idx + len(_THOUGHT_CLOSE) :]
                self._in_thought = False
                continue
            m = _SPECIAL.search(self._buf)
            if m is not None:
                out.append(self._buf[: m.start()])
                token = m.group(0)
                self._buf = self._buf[m.end() :]
                if token == _THOUGHT_OPEN:
                    self._in_thought = True
                # else: a control token or a stray close — just drop it.
                continue
            # No complete construct left; emit the safe prefix and hold back a
            # trailing fragment that might still become one on the next feed.
            partial = _PARTIAL.search(self._buf)
            if partial is not None:
                out.append(self._buf[: partial.start()])
                self._buf = self._buf[partial.start() :]
            else:
                out.append(self._buf)
                self._buf = ""
            break
        return "".join(out)

    def flush(self) -> str:
        """End of turn: emit any safe remainder (only if outside a thought), then reset."""
        out = "" if self._in_thought else self._buf
        self._buf = ""
        self._in_thought = False
        return out

    def reset(self) -> None:
        """Discard any partial state at a turn boundary."""
        self._buf = ""
        self._in_thought = False
