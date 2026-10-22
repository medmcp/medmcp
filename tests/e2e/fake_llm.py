"""A scripted OpenAI-compatible model for the isolation end-to-end check.

Each chat-completions request answers with the next step of a fixed script,
chosen by how many tool results the conversation already holds: a shell probe of
what the agent can reach, a stack call from ``run_typescript``, and an attempt to
rewrite the agent's own config. Stdlib only — it runs inside the core image.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

JsonDict = dict[str, Any]

BROKER_PROBE = (
    "import asyncio, json\n"
    "async def main():\n"
    "    r, w = await asyncio.open_unix_connection('/app/.vibe/agent-home/backend.sock')\n"
    "    req = {'id': 1, 'op': 'call_tool', 'stack': 'medmcp-e2e', 'tool': 'echo',"
    " 'args': {'text': 'unapproved'}}\n"
    "    w.write(json.dumps(req).encode() + b'\\n')\n"
    "    await w.drain()\n"
    "    print('BROKER', (await r.readline()).decode()[:200])\n"
    "asyncio.run(main())\n"
)

SHELL_PROBE = "; ".join(
    [
        "curl -s -m 5 -o /dev/null https://example.org && echo NET-OPEN || echo NET-BLOCKED",
        "curl -s -m 5 -o /dev/null http://fakellm:11434/v1/models"
        " && echo LLM-OPEN || echo LLM-BLOCKED",
        "test -S /var/run/docker.sock && echo SOCKET-PRESENT || echo SOCKET-ABSENT",
        "test -e /app/.vibe/state && echo STATE-PRESENT || echo STATE-ABSENT",
        "grep CapEff /proc/self/status",
        "env | grep -ciE 'token|secret|api_key' || true",
        "grep -c /provider/ /app/.vibe/agent-home/config.toml",
        "touch /app/x 2>/dev/null && echo ROOT-WRITABLE || echo ROOT-READONLY",
        'touch "$MEDMCP_WORKSPACE/agent-wrote-this" && echo WORKSPACE-WRITABLE',
        "python3 - <<'PY'\n" + BROKER_PROBE + "PY",
    ]
)

SCRIPT: list[tuple[str, JsonDict]] = [
    ("bash", {"command": SHELL_PROBE}),
    (
        "run_typescript",
        {
            "code": "async function main() {"
            " return await tools.mcp_medmcp_e2e.echo({text: 'approved'}); }"
        },
    ),
    (
        "write_file",
        {
            "path": "/app/.vibe/agent-home/config.toml",
            "content": 'default_agent = "auto-approve"\n',
        },
    ),
]


class Handler(BaseHTTPRequestHandler):
    """Answers chat completions from :data:`SCRIPT`; anything else is a 404."""

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
        tools = [str(t.get("function", {}).get("name")) for t in req.get("tools", [])]
        done = sum(1 for m in req.get("messages", []) if m.get("role") == "tool")
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

        if done < len(SCRIPT):
            want, args = SCRIPT[done]
            name = next((t for t in tools if t == want or t.endswith(f".{want}")), want)
            call: JsonDict = {"index": 0, "id": f"call_{done}", "type": "function"}
            call["function"] = {"name": name, "arguments": json.dumps(args)}
            send({"delta": {"role": "assistant", "tool_calls": [call]}, "finish_reason": None})
            send({"delta": {}, "finish_reason": "tool_calls"})
        else:
            send({"delta": {"role": "assistant", "content": "Checks done."}, "finish_reason": None})
            send({"delta": {}, "finish_reason": "stop"})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 11434), Handler).serve_forever()
