"""Who may use the API: the optional password, and never the agent's network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from medmcp import access, server

from .conftest import AUTH_HEADERS, TEST_PASSWORD


@pytest.fixture
def anonymous() -> TestClient:
    """A local client with the right Host and no password — another account on the machine."""
    return TestClient(server.app, base_url="http://127.0.0.1:8100")


@pytest.fixture
def no_password(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default install: no MEDMCP_PASSWORD."""
    monkeypatch.delenv(access.PASSWORD_ENV)
    access.password.cache_clear()


class TestPassword:
    """With MEDMCP_PASSWORD set, every /api and /ws request carries it."""

    def test_api_without_it_is_refused(self, anonymous: TestClient) -> None:
        """No Origin passes the origin check; the password is what stops it."""
        assert anonymous.get("/api/workspace").status_code == 401
        assert anonymous.get("/api/raw/anything").status_code == 401

    def test_bearer_and_login_cookie_are_accepted(self, anonymous: TestClient) -> None:
        """A script sends the header; a logged-in browser sends the cookie."""
        assert anonymous.get("/api/access", headers=AUTH_HEADERS).status_code == 200
        anonymous.cookies.set(access.COOKIE_NAME, access.cookie_value())
        assert anonymous.get("/api/access").status_code == 200

    def test_the_password_itself_is_not_a_cookie(self, anonymous: TestClient) -> None:
        """The cookie holds a derived value, so a leaked cookie is not the password."""
        anonymous.cookies.set(access.COOKIE_NAME, TEST_PASSWORD)
        assert anonymous.get("/api/access").status_code == 401

    def test_a_wrong_password_is_refused(self, anonymous: TestClient) -> None:
        """Presence is not enough."""
        assert (
            anonymous.get("/api/access", headers={"Authorization": "Bearer x"}).status_code == 401
        )

    def test_healthcheck_and_page_stay_open(self, anonymous: TestClient) -> None:
        """The compose healthcheck and the login screen itself need no password."""
        assert anonymous.get("/healthz").status_code == 200
        assert anonymous.get("/").status_code in {200, 503}

    @pytest.mark.parametrize("path", ["/ws/chat", "/ws/replay", "/ws/update"])
    def test_websocket_without_it_never_opens(self, anonymous: TestClient, path: str) -> None:
        """Refused before accept: no agent session, no replay."""
        with pytest.raises(WebSocketDisconnect), anonymous.websocket_connect(path):
            pass  # pragma: no cover — the connect itself raises


class TestLogin:
    """Logging in trades the password for an HttpOnly, SameSite=Strict cookie."""

    def test_wrong_password_does_not_log_in(self, anonymous: TestClient) -> None:
        """A guess gets a 403 and no cookie."""
        resp = anonymous.post("/api/login", json={"password": "guess"})
        assert resp.status_code == 403
        assert access.COOKIE_NAME not in resp.cookies

    def test_right_password_logs_the_browser_in(self, anonymous: TestClient) -> None:
        """After logging in, the cookie alone opens the API."""
        resp = anonymous.post("/api/login", json={"password": TEST_PASSWORD})
        assert resp.status_code == 200
        header = resp.headers["set-cookie"].lower()
        assert "httponly" in header
        assert "samesite=strict" in header
        assert anonymous.get("/api/access").status_code == 200


@pytest.mark.usefixtures("no_password")
def test_without_a_password_the_api_is_open(anonymous: TestClient) -> None:
    """The default install: the port is loopback-only and the machine single-user."""
    assert anonymous.get("/api/access").status_code == 200


class TestAgentNetwork:
    """The isolated agent shares a network with the core; nothing on it may use the UI."""

    @pytest.mark.usefixtures("no_password")
    def test_refused_even_without_a_password(self) -> None:
        """The password is optional; this refusal is not."""
        access.refuse_network("172.19.0.0/16")
        agent = TestClient(server.app, base_url="http://127.0.0.1:8100", client=("172.19.0.3", 1))
        assert agent.get("/api/access").status_code == 403
        assert agent.get("/").status_code == 403
        with pytest.raises(WebSocketDisconnect), agent.websocket_connect("/ws/replay"):
            pass  # pragma: no cover — the connect itself raises

    def test_the_password_does_not_let_it_in(self) -> None:
        """Knowing the password (it may read medmcp.env) changes nothing."""
        access.refuse_network("172.19.0.0/16")
        agent = TestClient(server.app, base_url="http://127.0.0.1:8100", client=("172.19.0.3", 1))
        assert agent.get("/api/access", headers=AUTH_HEADERS).status_code == 403

    def test_the_published_port_still_answers(self) -> None:
        """Browsers arrive from the default network's gateway, outside the agent's."""
        access.refuse_network("172.19.0.0/16")
        assert access.peer_refused("172.19.0.3")
        assert not access.peer_refused("172.18.0.1")
        assert not access.peer_refused("testclient")
