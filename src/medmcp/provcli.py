"""CLI for inspecting provenance and distilling reusable workflows.

Exposed as the ``medmcp`` console script::

    medmcp list                 # list sessions with a provenance record
    medmcp report  <session>    # (re)render report.md and print its path
    medmcp distill <session>    # distill a draft workflow from the raw log
    medmcp distill <session> --no-llm   # skip the LLM narrative pass
    medmcp promote <name>       # move a reviewed draft into active/ (reusable)
    medmcp replay  <name> -i in_1=path ...       # replay a workflow on new inputs
    medmcp replay  <name> --batch batch_plan.csv # roll it out over a cohort manifest
    medmcp workflows            # list personal workflows (draft + promoted)
    medmcp delete   <name>      # delete a personal workflow (draft or active)
    medmcp export   <name>      # write a shareable <name>.workflow.yaml
    medmcp import   <file>      # import a shared workflow file as a draft
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from medmcp import batchplan, distill, provenance, share


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
    print("Review and edit it, then promote it to keep it.")
    return 0


def _promote(name: str) -> int:
    """Move a reviewed draft workflow into ``active/``."""
    try:
        dst = distill.promote_draft(name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Promoted workflow to: {dst}")
    print("Run it from the workspace UI's workflow panel.")
    return 0


def _parse_bindings(pairs: list[str]) -> dict[str, str]:
    """Parse repeatable ``key=value`` CLI pairs into a dict."""
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"expected key=value, got {pair!r}")
        out[key.strip()] = value.strip()
    return out


def _workflow_dir(name: str) -> Path | None:
    """Resolve a workflow name to its recipe directory (active first, then draft)."""
    root = provenance.VIBE_HOME / "workflows"
    for kind in ("active", "draft"):
        candidate = root / kind / name
        if (candidate / "recipe.yaml").exists():
            return candidate
    return None


def _replay(
    name: str, inputs: dict[str, str], batch: str | None, column_map: dict[str, str]
) -> int:
    """Replay a workflow deterministically — a single input set or a batch manifest.

    Replay calls the stack tools directly (no LLM, no chat permission flow), so
    this is the headless twin of the UI's Run panel. A ``--batch`` manifest rolls
    the same recipe over every ``ok`` cohort row, sharing one set of stacks.
    """
    # Imported lazily so lightweight commands (list/workflows) don't pay to import
    # the MCP transport and stack-discovery machinery.
    from medmcp import replay, settings

    workflow_dir = _workflow_dir(name)
    if workflow_dir is None:
        print(f"error: no workflow named {name!r}", file=sys.stderr)
        return 1
    recipe = distill.load_recipe(workflow_dir)
    servers = settings.active_servers()
    cwd = os.environ.get("MEDMCP_WORKSPACE") or None

    if batch is not None:
        rows = batchplan.read_manifest(batch)
        try:
            binding = batchplan.runs_from_manifest(recipe, rows, column_map=column_map or None)
        except batchplan.BatchPlanError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if binding.skipped:
            print(f"Skipping {len(binding.skipped)} flagged item(s) — resolve them first:")
            for item in binding.skipped:
                label = item.get("subject", "?")
                if item.get("session"):
                    label = f"{label}/{item['session']}"
                print(f"  - {label}: {item.get('status')} ({item.get('reason')})")
        if not binding.runs:
            print("No 'ok' items to roll out.", file=sys.stderr)
            return 1
        pairs = ", ".join(f"{k}<-{v}" for k, v in binding.column_map.items())
        print(f"Rolling out {recipe.name!r} over {len(binding.runs)} item(s) [{pairs}] ...")
        results = asyncio.run(
            replay.run_batch(recipe, list(binding.runs), servers=servers, cwd=cwd)
        )
        failed = 0
        for index, result in enumerate(results):
            label = binding.labels[index] if index < len(binding.labels) else str(index)
            if result.ok:
                print(f"  [{index + 1}/{len(results)}] {label}: ok")
            else:
                failed += 1
                print(f"  [{index + 1}/{len(results)}] {label}: FAILED — {result.error}")
        print(f"Done: {len(results) - failed} ok, {failed} failed.")
        return 1 if failed else 0

    result = asyncio.run(replay.run(recipe, inputs, servers=servers, cwd=cwd))
    if result.error is not None and not result.steps:
        print(f"error: {result.error}", file=sys.stderr)
        return 1
    for step in result.steps:
        mark = "ok" if step.ok else f"FAILED — {step.error}"
        print(f"  step {step.index} {step.server}:{step.tool}: {mark}")
    print("Replay ok." if result.ok else f"Replay failed: {result.error}")
    return 0 if result.ok else 1


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


def _export(name: str, out: str | None) -> int:
    """Serialize a workflow into a single shareable ``<name>.workflow.yaml`` file."""
    try:
        text = share.export_workflow(name)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out_path = Path(out) if out else Path(f"{name}{share.EXPORT_SUFFIX}")
    out_path.write_text(text, encoding="utf-8")
    print(f"Exported workflow to: {out_path}")
    return 0


def _import(path: str) -> int:
    """Import a shared workflow file as a reviewable draft."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    try:
        draft = share.import_workflow(text)
    except share.WorkflowShareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Imported workflow as draft: {draft}")
    print("Review it, then promote it to reuse.")
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

    replay_p = sub.add_parser("replay", help="replay a workflow on new inputs (single or batch)")
    replay_p.add_argument("name")
    replay_p.add_argument(
        "-i",
        "--input",
        action="append",
        default=[],
        metavar="in_N=VALUE",
        help="bind one recipe input (repeatable); for a single replay",
    )
    replay_p.add_argument(
        "--batch",
        metavar="PLAN_CSV",
        help="roll out over a plan_batch manifest CSV instead of one input set",
    )
    replay_p.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="in_N=COLUMN",
        help="explicit recipe-input -> manifest-column mapping (repeatable)",
    )

    sub.add_parser("workflows", help="list personal workflows (draft + promoted)")

    delete_p = sub.add_parser("delete", help="delete a personal workflow by name")
    delete_p.add_argument("name")

    export_p = sub.add_parser("export", help="write a shareable <name>.workflow.yaml")
    export_p.add_argument("name")
    export_p.add_argument("--out", help="output path (default <name>.workflow.yaml)")

    import_p = sub.add_parser("import", help="import a shared workflow file as a draft")
    import_p.add_argument("file")

    args = parser.parse_args(argv)
    if args.command == "list":
        return _list_sessions()
    if args.command == "report":
        return _report(args.session_id, to_stdout=args.to_stdout)
    if args.command == "distill":
        return _distill(args.session_id, use_llm=args.use_llm)
    if args.command == "promote":
        return _promote(args.name)
    if args.command == "replay":
        try:
            inputs = _parse_bindings(args.input)
            column_map = _parse_bindings(args.map)
        except ValueError as exc:
            parser.error(str(exc))
        return _replay(args.name, inputs, args.batch, column_map)
    if args.command == "workflows":
        return _list_workflows()
    if args.command == "delete":
        return _delete(args.name)
    if args.command == "export":
        return _export(args.name, args.out)
    if args.command == "import":
        return _import(args.file)
    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
