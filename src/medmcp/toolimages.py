"""Images in stack tool results, carried past vibe to a local model that can see them.

vibe 2.26.1 keeps only the text of an MCP tool result: ``_parse_call_result`` in
``vibe/core/tools/mcp/tools.py`` drops every image block. A stack that answers with
a rendered slice, a montage or a plot reaches the model as its caption alone, and a
workflow that judges its result by looking (interactive segmentation does) runs
blind. The core sits on both sides of vibe, so it carries the images around it:

- **Out of the stack** (:func:`extract`, the broker hook for every approved call):
  each image block is stored here, content-addressed, and replaced by a marker,
  ``[medmcp-image:<id> returned by the tool]``. vibe passes the marker on as text,
  in the tool result or in the output of the ``run_typescript`` program that
  printed it, and the chat shows it that way.
- **Into the request** (:func:`attach_images`, in the LLM shim): the markers in the
  tool results of a chat request are looked up and, when the model can see images,
  the most recent ones go after those results in a user-role message labelled as
  coming from MedMCP; each marker is rewritten, in the request only, to say what
  became of its image.

Local models only, like the shim's other repairs: a cloud model does not go
through the shim, so it reads the markers as the broker wrote them and no image
leaves the machine this way. A result that also carries ``structuredContent`` reaches the
model as that JSON alone, so its markers, and its images, do not.

Stdlib and ``mcp.types`` only, so the shim can import it without the server.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mcp.types as mcp_types

log: logging.Logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

# The image types a vision model behind an OpenAI-compatible endpoint accepts as a
# data URL; anything else is described, not stored.
_EXTENSIONS: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}
_MIME_BY_EXTENSION: dict[str, str] = {ext: mime for mime, ext in _EXTENSIONS.items()}

#: Largest image kept, decoded. A render for the model is a few hundred kilobytes.
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024
#: Size of the store before the least recently used images are removed.
DEFAULT_STORE_BYTES: int = 256 * 1024 * 1024
#: Images attached to one request, the most recent first. Each costs context on
#: every later turn, so older ones are named but not sent again.
DEFAULT_MAX_ATTACHED: int = 8

_ID_LEN = 32
_ID_RE = re.compile(rf"[0-9a-f]{{{_ID_LEN}}}")
# Any marker the broker wrote or the shim rewrote: the id, then free text.
MARKER_RE: re.Pattern[str] = re.compile(rf"\[medmcp-image:([0-9a-f]{{{_ID_LEN}}})[^\]\n]*\]")

RETURNED = "returned by the tool"
NOT_SHOWN = "not shown to the model"
ATTACHED = "attached as image {n}"
EARLIER = "not attached: only the most recent images are"
GONE = "no longer available"
NO_VISION = "not shown: this model cannot see images"

ATTACHMENT_NOTE = (
    "[MedMCP] The images returned by the tool calls above, in order. This message "
    "comes from MedMCP, not from the user."
)


def marker(image_id: str, status: str = RETURNED) -> str:
    """The text that stands in for stored image *image_id*."""
    return f"[medmcp-image:{image_id} {status}]"


class ImageStore:
    """Content-addressed image files in one directory, bounded in total size.

    The broker writes and the shim reads, in two processes of the core; a file
    appears under its final name only once complete. Reading an image refreshes
    it, so the images a chat still shows are the last to go.
    """

    def __init__(self, root: Path, max_bytes: int = DEFAULT_STORE_BYTES) -> None:
        """Use *root* (created on first write), holding at most *max_bytes*."""
        self.root = root
        self.max_bytes = max_bytes

    def put(self, data: bytes, mime: str) -> str:
        """Store *data* of type *mime* (a key of the supported types); return its id."""
        image_id = hashlib.sha256(data).hexdigest()[:_ID_LEN]
        path = self.root / f"{image_id}.{_EXTENSIONS[mime]}"
        if path.exists():
            _touch(path)
            return image_id
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".incoming-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
            _touch(path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._prune(keep=path)
        return image_id

    def has(self, image_id: str) -> bool:
        """Whether image *image_id* is here (without reading or refreshing it)."""
        return _ID_RE.fullmatch(image_id) is not None and any(
            (self.root / f"{image_id}.{ext}").is_file() for ext in _MIME_BY_EXTENSION
        )

    def get(self, image_id: str) -> tuple[bytes, str] | None:
        """The bytes and type of image *image_id*, or ``None`` if it is not here."""
        if not _ID_RE.fullmatch(image_id):
            return None
        for ext, mime in _MIME_BY_EXTENSION.items():
            path = self.root / f"{image_id}.{ext}"
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                continue
            except OSError:
                return None
            with contextlib.suppress(OSError):
                _touch(path)
            return data, mime
        return None

    def _prune(self, keep: Path) -> None:
        """Remove the least recently used images until the store fits its bound."""
        files: list[tuple[float, int, Path]] = []
        for path in self.root.iterdir():
            if path.name.startswith("."):
                continue
            with contextlib.suppress(OSError):
                st = path.stat()
                files.append((st.st_mtime, st.st_size, path))
        total = sum(size for _, size, _ in files)
        for _, size, path in sorted(files, key=lambda f: f[0]):
            if total <= self.max_bytes:
                break
            if path == keep:
                continue
            with contextlib.suppress(OSError):
                path.unlink()
                total -= size


def _touch(path: Path) -> None:
    # File times come from a coarse kernel clock: writes a few milliseconds apart
    # share one, and the least recently used image would be a guess.
    now = time.time_ns()
    os.utime(path, ns=(now, now))


# ── out of the stack ─────────────────────────────────────────────────────────


def _replace(block: mcp_types.ImageContent, store: ImageStore) -> mcp_types.TextContent:
    mime = block.mimeType.lower()
    if mime not in _EXTENSIONS:
        text = f"[medmcp-image of type {block.mimeType}: {NOT_SHOWN}]"
    else:
        try:
            data = base64.b64decode(block.data, validate=True)
        except (binascii.Error, ValueError):
            data = b""
        if not data:
            text = f"[medmcp-image unreadable: {NOT_SHOWN}]"
        elif len(data) > MAX_IMAGE_BYTES:
            text = f"[medmcp-image of {len(data):,} bytes, too large: {NOT_SHOWN}]"
        else:
            text = marker(store.put(data, mime))
    return mcp_types.TextContent(type="text", text=text)


def extract(result: mcp_types.CallToolResult, store: ImageStore) -> mcp_types.CallToolResult:
    """*result* with each image block stored in *store* and replaced by its marker.

    A result without images is returned as it is.
    """
    if not any(isinstance(block, mcp_types.ImageContent) for block in result.content):
        return result
    content: list[mcp_types.ContentBlock] = [
        _replace(block, store) if isinstance(block, mcp_types.ImageContent) else block
        for block in result.content
    ]
    return result.model_copy(update={"content": content})


# ── into the request ─────────────────────────────────────────────────────────


def _rewrite_text(text: str, status_of: Callable[[str], str]) -> str:
    return MARKER_RE.sub(lambda m: marker(m.group(1), status_of(m.group(1))), text)


def _rewrite(message: JsonDict, status_of: Callable[[str], str]) -> None:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = _rewrite_text(content, status_of)
    elif isinstance(content, list):
        for part in cast("list[object]", content):
            if isinstance(part, dict):
                p = cast("JsonDict", part)
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    p["text"] = _rewrite_text(cast("str", p["text"]), status_of)


def _text_of(message: JsonDict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(cast("JsonDict", part).get("text") or "")
            for part in cast("list[object]", content)
            if isinstance(part, dict)
        )
    return ""


@dataclass
class _Found:
    message: int  # index in messages
    image_id: str


def attach_images(payload: JsonDict, store: ImageStore, *, vision: bool, max_images: int) -> int:
    """Attach the most recent tool-result images in *payload* for the model; in place.

    The last *max_images* distinct images named by markers in ``tool`` messages
    are sent, each in a user-role message right after the run of tool results it
    first came in, and their markers say ``attached as image <n>``. Older images
    are named as not attached; images the store no longer has, as gone. Without
    *vision*, or with *max_images* 0, nothing is attached and every marker says
    so. Markers in other messages are left alone: only a tool result can name an
    image.

    Returns the number of images attached.
    """
    raw = payload.get("messages")
    if not isinstance(raw, list):
        return 0
    messages = cast("list[object]", raw)
    found: list[_Found] = []
    for i, message in enumerate(messages):
        if isinstance(message, dict) and cast("JsonDict", message).get("role") == "tool":
            text = _text_of(cast("JsonDict", message))
            found.extend(_Found(i, m.group(1)) for m in MARKER_RE.finditer(text))
    if not found:
        return 0

    images: dict[str, tuple[bytes, str]] = {}
    chosen: list[str] = []  # newest first while choosing
    if vision and max_images > 0:
        for f in reversed(found):
            if f.image_id in images or f.image_id in chosen:
                continue
            stored = store.get(f.image_id)
            if stored is None:
                continue
            images[f.image_id] = stored
            chosen.append(f.image_id)
            if len(chosen) == max_images:
                break
    # Numbered in reading order, as they appear in the attachments.
    order = {image_id: n for n, image_id in enumerate(sorted(chosen, key=_first_seen(found)), 1)}

    def status_of(image_id: str) -> str:
        if max_images <= 0:
            return NOT_SHOWN
        if not vision:
            return NO_VISION
        if image_id in order:
            return ATTACHED.format(n=order[image_id])
        return EARLIER if store.has(image_id) else GONE

    for index in sorted({f.message for f in found}):
        _rewrite(cast("JsonDict", messages[index]), status_of)
    if not order:
        return 0

    # One attachment message after each run of tool results that has chosen images.
    by_run: dict[int, list[str]] = {}
    for f in found:
        if f.image_id not in order:
            continue
        end = f.message
        while end + 1 < len(messages) and _is_tool(messages[end + 1]):
            end += 1
        ids = by_run.setdefault(end, [])
        if f.image_id not in ids and all(f.image_id not in v for v in by_run.values()):
            ids.append(f.image_id)
    for end in sorted(by_run, reverse=True):
        parts: list[JsonDict] = [{"type": "text", "text": ATTACHMENT_NOTE}]
        for image_id in by_run[end]:
            data, mime = images[image_id]
            parts.append({"type": "text", "text": f"Image {order[image_id]}: {image_id}"})
            url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        messages.insert(end + 1, {"role": "user", "content": parts})
    return len(order)


def _is_tool(message: object) -> bool:
    return isinstance(message, dict) and cast("JsonDict", message).get("role") == "tool"


def _first_seen(found: list[_Found]) -> Callable[[str], int]:
    first: dict[str, int] = {}
    for n, f in enumerate(found):
        first.setdefault(f.image_id, n)
    return lambda image_id: first[image_id]
