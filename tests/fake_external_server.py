r"""A stand-in *external* MCP server that demands a bearer token.

`fake_stack_server.py` covers the local stdio case; this covers the other side of
external MCP, which nothing else can: a remote HTTP server that rejects an
unauthenticated request. It exists to answer one question end to end — does the
token an operator typed into the workspace actually arrive at the service? — and
to make the three interesting outcomes reproducible: right token, wrong token,
no token at all.

Run it beside a workspace container, on the same docker network, from the image
that already has the `mcp` library::

    docker run --rm -d --name mcp-authtest --network medmcp_default \\
        --entrypoint python \\
        -e EXPECTED_TOKEN=hunter2 -e MCP_TEST_PORT=9000 \\
        -v "$PWD/tests/fake_external_server.py:/srv.py:ro" \\
        medmcp-core:dev /srv.py

``--entrypoint python`` matters: the image's own entrypoint ignores the command
and starts the workspace server, which listens on 8100 and looks convincingly
like this one having come up on the wrong port.

Then add ``http://mcp-authtest:9000/mcp`` in Settings → Advanced with the same
token, ask the agent to call ``whoami``, and read ``docker logs mcp-authtest``.

The log prints the credential it received **in full** — that is the evidence the
whole exercise is after, and this server is a throwaway holding a throwaway
token. The tool's *reply* masks it instead: replies land in the chat transcript
and the provenance record, which are kept.
"""

from __future__ import annotations

import os
import sys

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

EXPECTED_TOKEN: str = os.environ.get("EXPECTED_TOKEN", "hunter2")
HEADER: str = os.environ.get("EXPECTED_HEADER", "authorization")
SCHEME: str = os.environ.get("EXPECTED_SCHEME", "Bearer ")
# Deliberately not `PORT`: the image this runs in already sets that, and
# inheriting it silently moves the server to another port.
PORT: int = int(os.environ.get("MCP_TEST_PORT", "9000"))

# FastMCP's DNS-rebinding guard allows only localhost by default, and this
# server is reached by container name — without this every authenticated
# request comes back 421 "Invalid Host header", which reads like an auth
# failure and is not one.
mcp: FastMCP = FastMCP(
    "fake-external",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_last_credential: str = ""


@mcp.tool()
def whoami() -> str:
    """Report how this call was authenticated, without echoing the credential."""
    if not _last_credential:
        return "called with no credential"
    return (
        f"authenticated with a {len(_last_credential)}-character token "
        f"ending {_last_credential[-4:]}"
    )


@mcp.tool()
def echo(text: str) -> str:
    """Return *text* unchanged, so the agent has something trivial to call."""
    return text


class RequireToken:
    """Reject any MCP request that does not carry the expected credential.

    A real service answers 401 and the agent surfaces it; that is the failure we
    want reproducible, because it is what a wrong or missing token looks like
    from the workspace and is otherwise indistinguishable from plumbing that
    dropped the token silently.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap *app*, checking every HTTP request before it reaches the MCP handler."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass the request through, or answer 401 when the credential is wrong."""
        # One process serving one probe at a time, so a module global is enough
        # to let `whoami` describe the call that reached it.
        global _last_credential
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        supplied = request.headers.get(HEADER, "")
        # Full value to the log on purpose: this is the proof the test is for.
        print(
            f"[fake-external] {request.method} {request.url.path} {HEADER}={supplied!r}", flush=True
        )
        expected = f"{SCHEME}{EXPECTED_TOKEN}"
        if supplied != expected:
            print(f"[fake-external] REJECTED (expected {expected!r})", flush=True)
            response = JSONResponse(
                {"error": "unauthorized", "detail": f"expected {HEADER}: {SCHEME}<token>"},
                status_code=401,
            )
            await response(scope, receive, send)
            return
        _last_credential = EXPECTED_TOKEN
        await self.app(scope, receive, send)


def main() -> None:
    """Serve the MCP endpoint at ``/mcp`` with the credential check in front."""
    app = mcp.streamable_http_app()
    app.add_middleware(RequireToken)
    print(
        f"[fake-external] listening on :{PORT}/mcp; expecting {HEADER}: {SCHEME}{EXPECTED_TOKEN}",
        file=sys.stderr,
        flush=True,
    )
    # Bound wide on purpose: reached from a sibling container, never published.
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
