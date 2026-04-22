"""CLI entrypoint for the MedMCP Chainlit UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Chainlit UI on localhost.

    The ``--host 127.0.0.1`` flag ensures the server never accidentally binds
    to all interfaces. See the security model in ``app.py`` for why this matters.
    """
    app = str(Path(__file__).resolve().parent / "app.py")
    result = subprocess.run(
        [sys.executable, "-m", "chainlit", "run", app, "-w", "--host", "127.0.0.1"],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
