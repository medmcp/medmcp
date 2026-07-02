"""Tests for the streaming <thought> stripper."""

from __future__ import annotations

from medmcp.reasoning import ThoughtStripper


def _stream(chunks: list[str]) -> str:
    """Feed chunks through a fresh stripper and return the concatenated visible text."""
    s = ThoughtStripper()
    out = "".join(s.feed(c) for c in chunks)
    return out + s.flush()


def test_no_thought_passes_through() -> None:
    """Plain text with no thought span is emitted unchanged."""
    assert _stream(["Hello ", "world."]) == "Hello world."


def test_whole_thought_removed() -> None:
    """A complete thought span (and its tags) is dropped; surrounding text kept."""
    assert _stream(["before <thought>secret reasoning</thought> after"]) == "before  after"


def test_thought_only_yields_nothing() -> None:
    """A message that is entirely a thought produces no visible output."""
    assert _stream(["<thought>all of it</thought>"]) == ""


def test_tag_split_across_chunks() -> None:
    """Open/close tags split across feeds are still recognized and stripped."""
    assert _stream(["ans<thou", "ght>hidden</thou", "ght>=42"]) == "ans=42"


def test_open_tag_char_by_char() -> None:
    """Feeding one char at a time never leaks a partial tag or thought content."""
    text = "A<thought>B</thought>C"
    assert _stream(list(text)) == "AC"


def test_multiple_thoughts() -> None:
    """Several thought spans in one stream are all removed."""
    assert _stream(["a<thought>x</thought>b<thought>y</thought>c"]) == "abc"


def test_unterminated_thought_dropped() -> None:
    """An unclosed thought at end of turn is dropped, not leaked."""
    assert _stream(["visible <thought>dangling reasoning that never closes"]) == "visible "


def test_trailing_partial_open_is_flushed_as_text() -> None:
    """A literal '<' tail that never becomes a tag is emitted on flush, not lost."""
    assert _stream(["done<"]) == "done<"


def test_control_token_dropped() -> None:
    """A harmony control token (and its mangled form) is removed, text kept."""
    assert _stream(["done<|channel|>"]) == "done"
    assert _stream(["planning batch...<channel|>"]) == "planning batch..."


def test_control_tokens_various() -> None:
    """The common harmony tokens are all stripped, surrounding text preserved."""
    assert _stream(["a<|start|>b<|message|>c<|end|>d"]) == "abcd"


def test_control_token_split_across_chunks() -> None:
    """A control token split across feeds is still recognized and dropped."""
    assert _stream(["answer<|chan", "nel|>more"]) == "answermore"


def test_plain_less_than_is_kept() -> None:
    """A literal '<' in prose (e.g. a comparison) is not mistaken for a token."""
    assert _stream(["EDSS < 3 and p < 0.05"]) == "EDSS < 3 and p < 0.05"


def test_reset_clears_in_thought_state() -> None:
    """reset() drops a half-open thought so the next turn starts clean."""
    s = ThoughtStripper()
    assert s.feed("keep <thought>partial") == "keep "
    s.reset()
    assert s.feed("fresh answer") == "fresh answer"
