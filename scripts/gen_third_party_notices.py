#!/usr/bin/env python3
"""Generate THIRD_PARTY_NOTICES.md from the resolved dependency state.

Reads only already-resolved local state — the uv lock (via ``uv export``) and the
installed ``node_modules`` tree — so it runs fully offline (no PyPI/npm network
access, which the institutional network throttles). Re-run via ``just notices``
whenever a runtime dependency is added, removed, or bumped.

Scope is the *distributed* closure: the Python runtime deps that ship in the
image (``uv sync --no-dev``) and the production frontend deps that get bundled
into ``frontend/dist`` (``npm ls --omit=dev``). Dev-only tools (pytest, ruff,
pyright, …) are excluded — they are never redistributed.

License strings are normalized to SPDX identifiers: a synonym table handles the
common spellings ("MIT License" -> MIT, "Apache 2.0" -> Apache-2.0, …) and the
actual license file is sniffed to resolve vague declarations ("UNKNOWN", a bare
"BSD", "Dual License") and PEP 639 packages that ship no license header at all.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "frontend"
OUTPUT = REPO_ROOT / "THIRD_PARTY_NOTICES.md"

HEADER = """\
# Third-party notices

MedMCP redistributes the third-party components listed below (they are bundled
into the published container image). Each remains under its own license; the full
license texts travel with each package's distribution (the frontend bundle's are
collected in `frontend/dist/LICENSES.txt` at build time).

This file is generated — do not edit it by hand. Regenerate with `just notices`
after changing a runtime dependency. The model and base OS/CUDA layers are not
package dependencies and are attributed in [`NOTICE`](NOTICE) instead.
"""

# Synonym table: lower-cased declared string -> canonical SPDX id.
_CANON = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache v2": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "isc": "ISC",
    "isc license": "ISC",
    "isc license (iscl)": "ISC",
    "mpl-2.0": "MPL-2.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "psf-2.0": "PSF-2.0",
    "python software foundation license": "PSF-2.0",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "modified bsd license": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "ofl-1.1": "OFL-1.1",
}

# Declarations too vague to trust — resolve these by sniffing the license file.
_AMBIGUOUS = {"", "bsd", "bsd license", "dual license", "unknown", "see package"}

FALLBACK = "see package"


def _norm(name: str) -> str:
    """Normalize a distribution name for matching (PEP 503-ish)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _canon_license(raw: str) -> str:
    """Map a declared license string to a canonical SPDX id, or "" if too vague."""
    key = re.sub(r"\s+", " ", raw).strip().rstrip(".").lower()
    if key in _CANON:
        return _CANON[key]
    if key in _AMBIGUOUS:
        return ""  # caller should fall back to a file sniff
    return raw.strip()  # already an SPDX id or expression (e.g. "Apache-2.0 OR MIT")


def _sniff_license_file(*dirs: Path) -> str:
    """Infer a canonical SPDX id from a bundled LICENSE file's text."""
    candidates: list[Path] = []
    for d in dirs:
        licenses_dir = d / "licenses"
        if licenses_dir.is_dir():
            candidates.extend(sorted(licenses_dir.rglob("*")))
        candidates.extend(sorted(d.glob("LICENSE*")) + sorted(d.glob("COPYING*")))
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[:4000].upper()
        if "GNU AFFERO" in text:
            return "AGPL"
        if "GNU LESSER" in text:
            return "LGPL"
        if "GNU GENERAL PUBLIC" in text:
            return "GPL"
        if "MOZILLA PUBLIC LICENSE" in text:
            return "MPL-2.0"
        if "APACHE LICENSE" in text:
            return "Apache-2.0"
        if "ISC LICENSE" in text or "THE ISC LICENSE" in text:
            return "ISC"
        if "PERMISSION IS HEREBY GRANTED, FREE OF CHARGE" in text or "MIT LICENSE" in text:
            return "MIT"
        if "REDISTRIBUTION AND USE IN SOURCE AND BINARY" in text:
            return "BSD-3-Clause" if "NEITHER THE NAME" in text else "BSD-2-Clause"
    return ""


def _resolve(raw: str, *dirs: Path) -> str:
    """Canonicalize a declared license, sniffing the file when it is too vague."""
    return _canon_license(raw) or _sniff_license_file(*dirs) or FALLBACK


# ── Python ───────────────────────────────────────────────────────────────────


def _index_installed_python() -> dict[str, dict[str, str]]:
    """Map normalized distribution name -> {version, license} from the venv."""
    index: dict[str, dict[str, str]] = {}
    for site in REPO_ROOT.glob(".venv/lib/python*/site-packages"):
        for dist in site.glob("*.dist-info"):
            meta = dist / "METADATA"
            if not meta.is_file():
                continue
            name = version = expr = field = classifier = ""
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Name: "):
                    name = line[6:].strip()
                elif line.startswith("Version: "):
                    version = line[9:].strip()
                elif line.startswith("License-Expression: "):
                    expr = line[20:].strip()
                elif line.startswith("License: "):
                    field = line[9:].strip()
                elif line.startswith("Classifier: License ::") and not classifier:
                    classifier = line.split("::")[-1].strip()
                elif line == "":
                    break  # end of headers; body (long license text) follows
            raw = expr or (field if len(field) <= 40 else "") or classifier
            if name:
                index[_norm(name)] = {"version": version, "license": _resolve(raw, dist)}
    return index


def _python_rows() -> list[tuple[str, str, str]]:
    """Return (name, version, license) for the runtime Python closure."""
    out = subprocess.run(
        ["uv", "export", "--no-dev", "--no-hashes", "--no-emit-project"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    index = _index_installed_python()
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", line)
        if not m:
            continue
        name, version = m.group(1), m.group(2)
        info = index.get(_norm(name))
        if info is None:
            # In the lock for cross-platform resolution but not installed here
            # (e.g. Windows-only deps) — so it never ships in the Linux image.
            continue
        rows.append((name, info.get("version") or version, info.get("license", FALLBACK)))
    return sorted(rows, key=lambda r: r[0].lower())


# ── JavaScript ─────────────────────────────────────────────────────────────────


def _collect_js(node: dict[str, object], acc: dict[str, str]) -> None:
    """Recursively collect name -> version from an ``npm ls --json`` tree."""
    deps = node.get("dependencies")
    if not isinstance(deps, dict):
        return
    for name, meta in deps.items():
        if isinstance(meta, dict):
            version = meta.get("version")
            if isinstance(version, str):
                acc[name] = version
            _collect_js(meta, acc)


def _js_license(name: str) -> str:
    """Resolve a frontend package's license from its package.json (+ LICENSE file)."""
    pkg_dir = FRONTEND / "node_modules" / Path(name)
    pkg = pkg_dir / "package.json"
    raw = ""
    if pkg.is_file():
        data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
        lic = data.get("license")
        if isinstance(lic, dict):
            lic = lic.get("type", "")
        raw = lic if isinstance(lic, str) else ""
    return _resolve(raw, pkg_dir)


def _js_rows() -> list[tuple[str, str, str]]:
    """Return (name, version, license) for the production frontend closure."""
    proc = subprocess.run(
        ["npm", "ls", "--omit=dev", "--all", "--json"],
        cwd=FRONTEND,
        capture_output=True,
        text=True,
    )
    tree = json.loads(proc.stdout or "{}")
    collected: dict[str, str] = {}
    _collect_js(tree, collected)
    rows = [(name, version, _js_license(name)) for name, version in collected.items()]
    return sorted(rows, key=lambda r: r[0].lower())


# ── Render ─────────────────────────────────────────────────────────────────────


def _table(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| Component | Version | License |", "| --- | --- | --- |"]
    lines += [f"| {name} | {version} | {lic} |" for name, version, lic in rows]
    return "\n".join(lines)


def main() -> None:
    """Render THIRD_PARTY_NOTICES.md from the resolved Python + JS closures."""
    py = _python_rows()
    js = _js_rows()
    if not py or not js:
        # A missing ecosystem means the env isn't ready — writing now would drop
        # every package from that side. Abort instead of truncating the file.
        need = []
        if not py:
            need.append("Python deps (run `uv sync`)")
        if not js:
            need.append("frontend deps (run `npm ci` in frontend/)")
        raise SystemExit(f"refusing to write {OUTPUT.name}: missing " + " and ".join(need))
    body = (
        f"{HEADER}\n"
        f"## Python ({len(py)} packages, runtime closure)\n\n{_table(py)}\n\n"
        f"## JavaScript / frontend ({len(js)} packages, production closure)\n\n{_table(js)}\n"
    )
    OUTPUT.write_text(body, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(py)} Python, {len(js)} JS packages)")


if __name__ == "__main__":
    main()
