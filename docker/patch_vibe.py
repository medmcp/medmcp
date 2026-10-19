"""Apply MedMCP's patches to the installed mistral-vibe package.

Run once at image build, after ``uv sync``::

    /app/.venv/bin/python docker/patch_vibe.py

Each patch replaces one exact snippet of vibe's source. If a snippet is missing
(vibe was upgraded and the code moved), the build fails rather than shipping an
unpatched agent; ``tests/test_patch_vibe.py`` checks the same against the locked
vibe version in CI. A file that already carries the patch is left alone.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Patch:
    """One exact replacement in one vibe source file."""

    name: str
    path: str  # relative to the vibe package directory
    old: str
    new: str


PATCHES: tuple[Patch, ...] = (
    # Every chat gets the same scratchpad path, so every chat starts with the same
    # system prompt (the path is part of it) and Ollama can reuse its cached
    # prompt prefix: the first reply of a new chat skips re-reading ~10k tokens.
    # Trade-off: chats share one scratch folder, and it is not deleted per chat.
    Patch(
        name="stable scratchpad",
        path="core/scratchpad.py",
        old=(
            "        dir_path = Path(\n"
            "            tempfile.mkdtemp(\n"
            '                prefix=f"{SCRATCHPAD_PREFIX}{shorten_session_id(session_id)}-"\n'
            "            )\n"
            "        )\n"
        ),
        new=(
            '        dir_path = Path(tempfile.gettempdir()) / f"{SCRATCHPAD_PREFIX}shared"'
            "  # MEDMCP_STABLE_SCRATCHPAD\n"
            "        dir_path.mkdir(parents=True, exist_ok=True)\n"
        ),
    ),
    Patch(
        name="keep the shared scratchpad",
        path="core/scratchpad.py",
        old="    if scratchpad_dir is None:\n        return\n    try:\n        shutil.rmtree",
        new=(
            '    if scratchpad_dir is None or scratchpad_dir.name.endswith("shared"):\n'
            "        return\n    try:\n        shutil.rmtree"
        ),
    ),
    # A command with an output redirect can write anywhere, so a chat-long
    # approval never covers it, whatever rule was approved earlier. The core
    # already stops offering such approvals; this enforces it inside the agent.
    Patch(
        name="no chat-long approval for redirects",
        path="core/tools/permissions.py",
        old=(
            "    def covers(self, tool_name: str, rp: RequiredPermission) -> bool:\n"
            "        return any(\n"
        ),
        new=(
            "    def covers(self, tool_name: str, rp: RequiredPermission) -> bool:\n"
            '        if "<redirect>" in rp.invocation_pattern:  # MEDMCP_NO_REDIRECT_RULES\n'
            "            return False\n"
            "        return any(\n"
        ),
    ),
)


def apply(package_dir: Path) -> list[str]:
    """Apply every patch under *package_dir*; return what was done per patch.

    Raises:
        RuntimeError: If a patch's snippet is neither present nor already applied.
    """
    done: list[str] = []
    for patch in PATCHES:
        file = package_dir / patch.path
        text = file.read_text()
        if patch.new in text:
            done.append(f"{patch.name}: already applied")
            continue
        if text.count(patch.old) != 1:
            raise RuntimeError(
                f"vibe patch {patch.name!r} does not apply to {file}: the code it "
                "replaces has changed. Update docker/patch_vibe.py for this vibe version."
            )
        file.write_text(text.replace(patch.old, patch.new))
        done.append(f"{patch.name}: applied")
    return done


def main() -> None:
    """Patch the vibe package importable from this interpreter."""
    import vibe

    package_dir = Path(vibe.__file__).parent
    for line in apply(package_dir):
        print(f"patch_vibe: {line}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
