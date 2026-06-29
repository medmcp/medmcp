"""The viewer "workspace context" note and its inverse.

When a file is open in the viewer, the workspace server appends a
``[workspace context: …]`` note to the prompt text it sends the agent. The note
lets the agent resolve references like "this image" *and* hands it the absolute
on-disk path the stack tools expect (the workspace is bind-mounted at the
identical absolute path — path parity). The note is live-turn metadata, not part
of the user's request, so it is stripped wherever the prompt text resurfaces:
session titles, replayed turns, and workflow distillation.

The note's format (:func:`build_workspace_note`) and the pattern that removes it
(:func:`strip_workspace_note`) live here together so they stay in sync — they
were previously duplicated across ``server.py`` and ``distill.py``.
"""

from __future__ import annotations

import re

# The note is always appended last, separated by a blank line, so the strip
# pattern cuts from its marker to the end of the text — this also handles a title
# truncated mid-note (no closing "]"). The "\n\n" anchor avoids touching a
# bracketed phrase a user happened to type inline.
_NOTE_RE = re.compile(r"\n\n\[workspace context:.*$", re.DOTALL)


def build_workspace_note(absolute_path: str) -> str:
    """Build the ``[workspace context: …]`` note for the file open in the viewer.

    *absolute_path* must already be resolved to its on-disk absolute path (path
    parity) — handing the agent the absolute path lets its first tool call hit
    instead of recovering after a filesystem search.
    """
    return (
        f'\n\n[workspace context: the file "{absolute_path}" is currently open in the '
        'viewer; references like "this image" or "the current image" mean that file]'
    )


def strip_workspace_note(text: str) -> str:
    """Remove the appended ``[workspace context: …]`` note from *text*."""
    return _NOTE_RE.sub("", text).strip()
