#!/usr/bin/env python3
"""Print one version's section of CHANGELOG.md, for use as release notes.

The release workflow feeds the output to ``gh release create --notes-file``, so
a missing section is a hard error: a release whose notes silently came out empty
is worse than one that fails to publish.

Usage:
    python scripts/changelog_section.py 0.2.0 [--changelog CHANGELOG.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract(text: str, version: str) -> str:
    """Return the body of ``version``'s section, without its own heading.

    Accepts the heading spellings the file has used: ``## 0.1.0``,
    ``## [0.2.0]``, and either with a trailing date.

    Args:
        text: Full contents of the changelog.
        version: Version to look for, without a leading ``v``.

    Returns:
        The section body, stripped of surrounding blank lines.

    Raises:
        LookupError: If no heading for that version is present.
    """
    start = re.compile(rf"^## +\[?{re.escape(version)}\]?\b.*$", re.MULTILINE)
    match = start.search(text)
    if match is None:
        raise LookupError(f"no '## {version}' heading in the changelog")

    rest = text[match.end() :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    body = rest[: following.start()] if following else rest
    return body.strip("\n")


def main() -> int:
    """Parse arguments, print the section, and report failure as an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="version to extract, e.g. 0.2.0")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="path to the changelog (default: ./CHANGELOG.md)",
    )
    args = parser.parse_args()

    try:
        section = extract(args.changelog.read_text(encoding="utf-8"), args.version)
    except (OSError, LookupError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not section:
        print(f"error: section for {args.version} is empty", file=sys.stderr)
        return 1

    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
