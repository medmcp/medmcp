"""The viewer "workspace context" note and its inverse.

When a file is open in the viewer, the workspace server sends a
``[workspace context: …]`` note with the prompt (as a content block flagged
``automatic``, which vibe ≥2.23 keeps out of auto-title derivation). The note
lets the agent resolve references like "this image" *and* hands it the absolute
on-disk path the stack tools expect (the workspace is bind-mounted at the
identical absolute path — path parity). The note is live-turn metadata, not part
of the user's request: replayed turns carry the note-free text natively (the
prompt's ``user_display_content`` meta), while the persisted prompt text still
contains the note, so :func:`strip_workspace_note` remains the fallback for
pre-2.23 transcripts and for distillation's seed request.

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
