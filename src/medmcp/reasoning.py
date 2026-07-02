"""Strip a local model's inline chain-of-thought from streamed agent text.

Some locally-served models (e.g. gemma via Ollama) emit their reasoning inside
``<thought>...</thought>`` spans in the *normal message content* rather than a
separate reasoning channel (vibe's ``agent_thought_chunk``, fed from a model's
``reasoning_content`` field). That content is relayed to the UI verbatim, so the
reasoning leaks into the chat.

:class:`ThoughtStripper` removes those spans from a token stream. It is
streaming-aware: an opening/closing tag may be split across feeds, so it holds
back only the shortest possible partial-tag suffix and emits everything else.
"""

from __future__ import annotations

_OPEN = "<thought>"
_CLOSE = "</thought>"


def _held_tail(buf: str, tag: str) -> int:
    """Length of the longest suffix of *buf* that is a proper prefix of *tag*.

    That suffix might be the start of a tag split across chunk boundaries, so it
    must be held back until the next feed rather than emitted.
    """
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


class ThoughtStripper:
    """Streaming remover of ``<thought>...</thought>`` spans (tags may split across feeds)."""

    def __init__(self) -> None:
        """Start with an empty buffer, outside any thought span."""
        self._buf = ""
        self._in_thought = False

    def feed(self, text: str) -> str:
        """Consume streamed *text*; return only the portion outside thought spans."""
        self._buf += text
        out: list[str] = []
        while self._buf:
            if not self._in_thought:
                i = self._buf.find(_OPEN)
                if i != -1:
                    out.append(self._buf[:i])
                    self._buf = self._buf[i + len(_OPEN) :]
                    self._in_thought = True
                    continue
                k = _held_tail(self._buf, _OPEN)
                cut = len(self._buf) - k
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            j = self._buf.find(_CLOSE)
            if j != -1:
                self._buf = self._buf[j + len(_CLOSE) :]
                self._in_thought = False
                continue
            k = _held_tail(self._buf, _CLOSE)
            self._buf = self._buf[len(self._buf) - k :]
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
