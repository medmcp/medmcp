"""CLI entrypoint for the MedMCP Chainlit UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Chainlit UI."""
    app = str(Path(__file__).resolve().parent / "app.py")
    subprocess.run([sys.executable, "-m", "chainlit", "run", app, "-w"], check=True)


if __name__ == "__main__":
    main()
