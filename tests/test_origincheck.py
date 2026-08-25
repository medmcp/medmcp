"""Tests for the browser-origin guard on the workspace server.

The server binds loopback with no authentication, which is safe against the
network and not against the browser: any page the operator visits can reach
``127.0.0.1``, and a WebSocket upgrade is exempt from the same-origin policy
entirely. Since ``read_file`` needs no approval, an unrefused socket is enough
to have the agent read the workspace and stream it back to that page — so what
is asserted here is mostly what does *not* get through.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# pyright: reportPrivateUsage=false
from medmcp import origincheck, server


class TestRejectReason:
    """The header rules, independent of any server."""

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1:8100", "localhost:8100", "[::1]:8100", "127.0.0.1", "localhost", "127.1.2.3"],
    )
    def test_loopback_hosts_pass(self, host: str) -> None:
        """Every spelling of "this machine" is accepted, on any port."""
        assert origincheck.reject_reason(host, "") is None

    @pytest.mark.parametrize("host", ["evil.example", "evil.example:8100", "10.0.0.5:8100"])
    def test_foreign_host_is_refused(self, host: str) -> None:
        """A non-loopback Host means DNS rebinding: the name resolved here."""
        assert origincheck.reject_reason(host, "") is not None

    @pytest.mark.parametrize(
        "origin",
        ["http://localhost:5173", "http://127.0.0.1:8100", "https://localhost", "http://[::1]:80"],
    )
    def test_loopback_origins_pass(self, origin: str) -> None:
        """The page's own origin, and the dev server that proxies to it."""
        assert origincheck.reject_reason("127.0.0.1:8100", origin) is None

    @pytest.mark.parametrize(
        "origin",
        [
            "https://evil.example",
            "http://evil.example:8100",
            "null",
            "https://127.0.0.1.evil.example",
            "https://evil.example/?x=http://localhost",
        ],
    )
    def test_foreign_origins_are_refused(self, origin: str) -> None:
        """Including the near-misses: a suffix, a sandboxed frame, a lookalike path."""
        assert origincheck.reject_reason("127.0.0.1:8100", origin) is not None

    def test_absent_headers_pass(self) -> None:
        """No Origin means a non-browser client; HTTP/1.1 guarantees a Host."""
        assert origincheck.reject_reason("", "") is None

    def test_operator_can_allow_a_deployment_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deployment behind a proxy says so explicitly; the default is closed."""
        assert origincheck.reject_reason("medmcp.hospital.example", "") is not None
        monkeypatch.setenv("MEDMCP_ALLOWED_HOSTS", "medmcp.hospital.example")
        monkeypatch.setenv("MEDMCP_ALLOWED_ORIGINS", "https://medmcp.hospital.example")
        assert (
            origincheck.reject_reason("medmcp.hospital.example", "https://medmcp.hospital.example")
            is None
        )


class TestGuardOnTheApp:
    """The middleware, against the real routes."""

    def test_same_origin_request_is_served(self) -> None:
        """The workspace's own page keeps working."""
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        assert (
            client.get("/healthz", headers={"Origin": "http://127.0.0.1:8100"}).status_code == 200
        )

    def test_healthcheck_without_an_origin_is_served(self) -> None:
        """The container healthcheck sends no Origin and must not be refused."""
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        assert client.get("/healthz").status_code == 200

    def test_cross_origin_http_is_refused(self) -> None:
        """A page on another site cannot reach the API at all."""
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        resp = client.get("/api/external-mcp", headers={"Origin": "https://evil.example"})
        assert resp.status_code == 403

    def test_rebound_host_is_refused(self) -> None:
        """A name resolved to 127.0.0.1 is caught by the Host check."""
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        resp = client.get("/healthz", headers={"Host": "evil.example"})
        assert resp.status_code == 403

    def test_cross_origin_websocket_never_reaches_the_agent(self) -> None:
        """The upgrade fails before `accept`, so no session is created.

        This is the one that mattered: `/ws/chat` handed a live agent session to
        any origin, and `read_file` needs no approval, so the page could have
        the workspace read out to it with no dialog shown.
        """
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws/chat", headers={"Origin": "https://evil.example"}),
        ):
            pass  # pragma: no cover — the connect itself raises

    def test_cross_origin_replay_socket_is_refused(self) -> None:
        """`/ws/replay` runs workflows with no permission flow; same door, same lock."""
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws/replay", headers={"Origin": "https://evil.example"}),
        ):
            pass  # pragma: no cover — the connect itself raises
