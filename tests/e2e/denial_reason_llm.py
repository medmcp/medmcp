"""A scripted OpenAI-compatible model for ``tests/e2e/denial_reason.py``.

Its first answer calls the stack tool ``inspect`` from ``run_typescript`` with an
input file that does not exist, which the path guard turns back; every
later answer ends the turn. Each request's newest tool results are printed as
``TOOL-RESULT <text>`` so the check can read what the model was told. Stdlib
only — it runs inside the core image.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

JsonDict = dict[str, Any]

CALL = (
    "async function main() {"
    " return await tools.mcp_medmcp_e2e.inspect({input_path: 'missing_scan.nii.gz'}); }"
)


def _text(message: JsonDict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts = cast("list[JsonDict]", content)
        return "\n".join(str(p.get("text") or "") for p in parts)
    return str(content or "")


class Handler(BaseHTTPRequestHandler):
    """Answers chat completions with the script; anything else is a 404."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: ANN401
        """Quiet."""

    def _json(self, code: int, obj: JsonDict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Model list."""
        self._json(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self) -> None:
        """One scripted turn, streamed."""
        length = int(self.headers.get("Content-Length", "0"))
        req: JsonDict = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": "not here"})
            return
        messages: list[JsonDict] = req.get("messages", [])
        last = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
        for m in messages[last + 1 :]:
            if m.get("role") == "tool":
                print("TOOL-RESULT " + json.dumps(_text(m)), flush=True)
        tools = [str(t.get("function", {}).get("name")) for t in req.get("tools", [])]
        first = last < 0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base: JsonDict = {
            "id": "e2e",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
        }

        def send(choice: JsonDict) -> None:
            chunk = {**base, "model": "fake", "choices": [{"index": 0, **choice}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()

        if first:
            name = next((t for t in tools if t.endswith("run_typescript")), "run_typescript")
            call: JsonDict = {"index": 0, "id": "call_0", "type": "function"}
            call["function"] = {"name": name, "arguments": json.dumps({"code": CALL})}
            send({"delta": {"role": "assistant", "tool_calls": [call]}, "finish_reason": None})
            send({"delta": {}, "finish_reason": "tool_calls"})
        else:
            send({"delta": {"role": "assistant", "content": "Done."}, "finish_reason": None})
            send({"delta": {}, "finish_reason": "stop"})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 11434), Handler).serve_forever()
