"""CLI for inspecting provenance and distilling reusable workflows.

Exposed as the ``medmcp`` console script::

    medmcp list                 # list sessions with a provenance record
    medmcp report  <session>    # (re)render report.md and print its path
    medmcp distill <session>    # distill a draft workflow from the raw log
    medmcp distill <session> --no-llm   # skip the LLM narrative pass
    medmcp promote <name>       # move a reviewed draft into active/ (reusable)
    medmcp workflows            # list personal workflows (draft + promoted)
    medmcp delete   <name>      # delete a personal workflow (draft or active)
"""

from __future__ import annotations

import argparse
import sys

from medmcp import distill, provenance


def _list_sessions() -> int:
    """Print the session ids that have a provenance directory."""
    root = provenance.VIBE_HOME / "provenance"
    if not root.is_dir():
        print("No provenance records found.", file=sys.stderr)
        return 0
    sessions = sorted(p.name for p in root.iterdir() if p.is_dir())
    for session_id in sessions:
        manifest = provenance.read_manifest(session_id)
        created = manifest.get("created_at", "?") if manifest else "?"
        n_events = len(provenance.read_run_events(session_id))
        print(f"{session_id}\t{created}\t{n_events} tool calls")
    return 0


def _report(session_id: str, *, to_stdout: bool) -> int:
    """Render and write report.md for *session_id*."""
    if to_stdout:
        print(provenance.render_report(session_id))
        return 0
    path = provenance.write_report(session_id)
    if path is None:
        print(f"No provenance to report for session {session_id!r}.", file=sys.stderr)
        return 1
    print(str(path))
    return 0


def _distill(session_id: str, *, use_llm: bool) -> int:
    """Distill a draft workflow for *session_id*."""
    try:
        draft_dir = distill.distill_session(session_id, use_llm=use_llm)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Draft workflow written to: {draft_dir}")
    print("Review and edit, then promote it into a skill_paths directory to reuse it.")
    return 0


def _promote(name: str) -> int:
    """Move a reviewed draft workflow into ``active/`` so it loads as a skill."""
    try:
        dst = distill.promote_draft(name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Promoted workflow to: {dst}")
    print("Restart the UI to pick it up; it will be available via the skill system.")
    return 0


def _list_workflows() -> int:
    """Print personal workflows discovered under ``.vibe/workflows/``."""
    root = provenance.VIBE_HOME / "workflows"
    found = False
    for kind in ("active", "draft"):
        base = root / kind
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if (d / "SKILL.md").exists():
                found = True
                print(f"{d.name}\t{kind}")
    if not found:
        print("No personal workflows found.", file=sys.stderr)
    return 0


def _delete(name: str) -> int:
    """Delete a personal workflow (from active/ or draft/) by name."""
    try:
        removed = distill.delete_workflow(name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Deleted workflow: {removed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``medmcp`` console script."""
    parser = argparse.ArgumentParser(prog="medmcp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list sessions with a provenance record")

    report_p = sub.add_parser("report", help="render report.md for a session")
    report_p.add_argument("session_id")
    report_p.add_argument(
        "--print", dest="to_stdout", action="store_true", help="print to stdout instead of writing"
    )

    distill_p = sub.add_parser("distill", help="distill a reusable workflow from a session")
    distill_p.add_argument("session_id")
    distill_p.add_argument(
        "--no-llm", dest="use_llm", action="store_false", help="skip the LLM narrative pass"
    )

    promote_p = sub.add_parser("promote", help="move a reviewed draft workflow into active/")
    promote_p.add_argument("name")

    sub.add_parser("workflows", help="list personal workflows (draft + promoted)")

    delete_p = sub.add_parser("delete", help="delete a personal workflow by name")
    delete_p.add_argument("name")

    args = parser.parse_args(argv)
    if args.command == "list":
        return _list_sessions()
    if args.command == "report":
        return _report(args.session_id, to_stdout=args.to_stdout)
    if args.command == "distill":
        return _distill(args.session_id, use_llm=args.use_llm)
    if args.command == "promote":
        return _promote(args.name)
    if args.command == "workflows":
        return _list_workflows()
    if args.command == "delete":
        return _delete(args.name)
    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
