"""Small state files under ``.vibe/``: tolerant readers and atomic writers.

Every module that keeps a JSON or TOML state file (settings, sessions, runs,
the update check, the path guard's ledger) goes through these, so the two
properties they all need are implemented once:

- **Reading never raises.** A missing, unreadable or malformed file reads as
  "absent" (``None`` / ``[]``); a malformed one is logged, since it is worth a
  look, while a missing one is the normal first-run state.
- **Writing is atomic across processes.** The payload goes to a uniquely named
  temp file in the target directory (never a fixed ``.tmp`` path, which a second
  writer — another server instance, the CLI — could consume or interleave into)
  and ``os.replace`` makes the last complete writer win. ``mkstemp`` creates the
  file 0600, which is what the secret-bearing files rely on.

Stdlib only (``tomli_w`` is imported on first TOML write), so import-light hooks
such as ``pathguard`` can use it too.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, cast

JsonDict = dict[str, Any]

log = logging.getLogger(__name__)


def read_json(path: Path) -> JsonDict | None:
    """The JSON object in *path*, or ``None`` when absent, unreadable or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        log.warning("could not read %s: expected a JSON object", path)
        return None
    return cast("JsonDict", data)


def read_jsonl(path: Path) -> list[JsonDict]:
    """The JSON objects in a ``.jsonl`` file, one per line; blank and bad lines skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[JsonDict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(cast("JsonDict", item))
    return out


def _atomic_write(path: Path, write: Callable[[BinaryIO], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_json(path: Path, payload: JsonDict) -> None:
    """Write *payload* as JSON to *path* atomically (see the module docstring)."""
    _atomic_write(path, lambda fh: fh.write(json.dumps(payload).encode("utf-8")))


def atomic_write_toml(path: Path, payload: JsonDict) -> None:
    """Write *payload* as TOML to *path* atomically (see the module docstring)."""
    import tomli_w

    _atomic_write(path, lambda fh: tomli_w.dump(payload, fh))
