"""Render a chat as a standalone, printable HTML page ("Export" in the Chats menu).

The page is built from what is already on disk — vibe's transcript
(``messages.jsonl`` across the compaction chain) and, when provenance was
recorded, the run log (approval decisions) and the environment manifest — so it
works for any chat still in the list, including ones from before this existed.

It is meant to be saved as a PDF from the browser: light, paged, no app chrome,
no external resources. The agent's reasoning and the system prompt are left out;
what the person said, what the agent answered, and every tool call with its
arguments, decision and result are in.

Markdown in messages is rendered with ``markdown-it-py`` with raw HTML disabled,
so transcript text can never inject markup into the page.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Collection
from datetime import UTC, datetime
from typing import cast

from markdown_it import MarkdownIt

from medmcp import provenance
from medmcp.acp import JsonDict
from medmcp.workspace_note import display_content_text, strip_workspace_note

# Tool output longer than this is cut in the export (the full text stays in the
# transcript); a single file listing can otherwise run to hundreds of pages.
MAX_OUTPUT_CHARS = 4000

_DECISION_LABELS = {
    "allow_once": "Approved",
    "allow_always": "Approved for this chat",
    "reject_once": "Denied",
}

# vibe's own session-id shape; anything else (``..``) never reaches a path join.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

_md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")


def _utc_label(stamp: object) -> str:
    """``2026-10-15T20:14:28+00:00`` as ``2026-10-15 20:14 UTC`` ("" if unknown)."""
    if not isinstance(stamp, str):
        return ""
    try:
        return datetime.fromisoformat(stamp).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return ""


def is_session_id(value: str) -> bool:
    """Whether *value* has the shape of a vibe session id."""
    return _SESSION_ID_RE.fullmatch(value) is not None


def _markdown(text: str) -> str:
    return _md.render(text)


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n… [{len(text) - MAX_OUTPUT_CHARS} more characters]"


def _user_text(msg: JsonDict) -> str:
    """What the person typed: the note-free display text, else the stripped content."""
    display = msg.get("user_display_content")
    if isinstance(display, dict):
        text = display_content_text(cast("JsonDict", display))
        if text:
            return text
    content = msg.get("content")
    return strip_workspace_note(content) if isinstance(content, str) else ""


def _pretty_arguments(raw: object) -> str:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return json.dumps(raw, indent=2, ensure_ascii=False)


def _report_body(report: str) -> str:
    """The provenance report minus its own title and session line, headings demoted."""
    lines = [
        ("#" + line) if line.startswith("## ") else line
        for line in report.splitlines()
        if not line.startswith("# ") and not line.startswith("**Session:**")
    ]
    return "\n".join(lines)


def _tool_block(call: JsonDict, result: JsonDict | None, event: JsonDict | None) -> str:
    fn = cast("JsonDict", call.get("function") or {})
    name = str(fn.get("name") or "tool")
    decision = str((event or {}).get("permission_decision") or "")
    status = str((event or {}).get("status") or "")
    badges: list[str] = []
    if decision:
        badges.append(_DECISION_LABELS.get(decision, decision.replace("_", " ")))
    if status and status != "completed":
        badges.append(status)
    badge_html = "".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges)
    parts = [
        '<section class="tool">',
        '<div class="tool-head">'
        f'<span class="tool-name">{html.escape(name)}</span>{badge_html}</div>',
    ]
    explanation = (event or {}).get("explanation")
    if isinstance(explanation, str) and explanation:
        parts.append(f'<p class="tool-why">{html.escape(explanation)}</p>')
    args = _pretty_arguments(fn.get("arguments"))
    if args and args not in ("{}", "null"):
        parts.append(f"<pre>{html.escape(args)}</pre>")
    if result is not None:
        output = result.get("content")
        if isinstance(output, str) and output.strip():
            parts.append('<div class="tool-label">Result</div>')
            parts.append(f"<pre>{html.escape(_clip(output.strip()))}</pre>")
    parts.append("</section>")
    return "\n".join(parts)


def render_chat_html(
    session_id: str, *, title: str | None = None, stop_ids: Collection[str] = ()
) -> str | None:
    """Return the export page for *session_id*, or ``None`` if it has no transcript."""
    from medmcp.titles import load_session_messages  # local: titles pulls in httpx

    messages = load_session_messages(session_id, stop_ids=stop_ids)
    if not messages:
        return None
    results = {str(m.get("tool_call_id")): m for m in messages if m.get("role") == "tool"}
    events = {str(e.get("tool_call_id")): e for e in provenance.read_run_events(session_id)}
    manifest = provenance.read_manifest(session_id) or {}
    model = cast("JsonDict", manifest.get("model") or {})

    body: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            if msg.get("injected") or msg.get("context_boundary"):
                continue
            text = _user_text(msg).strip()
            if text:
                body.append(
                    '<article class="msg user"><div class="who">You</div>'
                    f"{_markdown(text)}</article>"
                )
        elif role == "assistant":
            content = msg.get("content")
            calls = cast("list[JsonDict]", msg.get("tool_calls") or [])
            if not (isinstance(content, str) and content.strip()) and not calls:
                continue
            inner = [_markdown(content)] if isinstance(content, str) and content.strip() else []
            for call in calls:
                cid = str(call.get("id") or "")
                inner.append(_tool_block(call, results.get(cid), events.get(cid)))
            body.append(
                '<article class="msg agent"><div class="who">MedMCP</div>'
                + "\n".join(inner)
                + "</article>"
            )

    heading = title or "MedMCP chat"
    meta_rows = [
        ("Exported", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")),
        ("Chat started", _utc_label(manifest.get("created_at"))),
        ("Model", str(model.get("name") or "")),
        ("Chat ID", session_id),
    ]
    meta_html = "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>" for k, v in meta_rows if v
    )
    report = ""
    if manifest or events:
        report = (
            '<section class="appendix"><h2>Provenance record</h2>'
            + _markdown(_report_body(provenance.render_report(session_id)))
            + "</section>"
        )
    return _PAGE.format(
        title=html.escape(heading),
        meta=meta_html,
        body="\n".join(body),
        report=report,
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --ink: #1b2330; --muted: #5d6676; --rule: #dde1e8; --soft: #f4f6f9; --accent: #2a5db0; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: #fff;
  color: var(--ink);
  font: 14px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
main {{ max-width: 46rem; margin: 0 auto; padding: 2rem 16px 4rem; }}
header {{ border-bottom: 2px solid var(--ink); padding-bottom: 1rem; margin-bottom: 1.5rem; }}
h1 {{ font-size: 1.6rem; margin: 0 0 .75rem; line-height: 1.2; }}
dl {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
  gap: .4rem 1.5rem;
  margin: 0;
}}
dt {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
dd {{ margin: 0; font-size: .85rem; overflow-wrap: anywhere; }}
.actions {{ margin-top: 1rem; }}
button {{
  font: inherit;
  padding: .45rem .9rem;
  border: 1px solid var(--accent);
  background: var(--accent);
  color: #fff;
  border-radius: 4px;
  cursor: pointer;
}}
.msg {{ padding: .75rem 0 .5rem; border-bottom: 1px solid var(--rule); break-inside: auto; }}
.who {{
  font-size: .7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
}}
.user .who {{ color: var(--accent); }}
.msg p {{ margin: .4rem 0; }}
table {{ border-collapse: collapse; margin: .5rem 0; font-size: .85rem; }}
th, td {{ border: 1px solid var(--rule); padding: .25rem .5rem; text-align: left; }}
code, pre {{
  font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: .78rem;
}}
pre {{
  background: var(--soft);
  padding: .5rem .65rem;
  border-radius: 4px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: .35rem 0;
}}
.tool {{
  border-left: 3px solid var(--rule);
  padding: .2rem 0 .2rem .75rem;
  margin: .6rem 0;
  break-inside: avoid-page;
}}
.tool-head {{ display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; }}
.tool-name {{
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-weight: 600;
  font-size: .82rem;
}}
.badge {{
  font-size: .7rem;
  padding: 0 .4rem;
  border: 1px solid var(--muted);
  border-radius: 999px;
  color: var(--muted);
}}
.tool-why {{ color: var(--muted); font-size: .85rem; }}
.tool-label {{
  font-size: .7rem;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--muted);
  margin-top: .3rem;
}}
.appendix {{
  margin-top: 2.5rem;
  padding-top: 1rem;
  border-top: 2px solid var(--ink);
  font-size: .85rem;
}}
.appendix h2 {{ font-size: 1.15rem; }}
.appendix h3 {{ font-size: .95rem; margin: 1rem 0 .25rem; }}
@media print {{
  .actions {{ display: none; }}
  main {{ padding: 0; max-width: none; }}
  @page {{ margin: 16mm 14mm; }}
}}
</style>
</head>
<body>
<main>
<header>
<h1>{title}</h1>
<dl>{meta}</dl>
<div class="actions"><button type="button" onclick="window.print()">Save as PDF</button></div>
</header>
{body}
{report}
</main>
</body>
</html>
"""
