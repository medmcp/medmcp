"""The access token: loopback is shared, so the API needs more than an address."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from medmcp import access, server

from .conftest import TEST_TOKEN


@pytest.fixture
def anonymous() -> TestClient:
    """A local client with the right Host and no token — another account on the machine."""
    return TestClient(server.app, base_url="http://127.0.0.1:8100")


class TestGuard:
    """Every /api and /ws request carries the token; the page and /healthz do not need it."""

    def test_api_without_a_token_is_refused(self, anonymous: TestClient) -> None:
        """No Origin passes the origin check; the token is what stops it."""
        assert anonymous.get("/api/workspace").status_code == 401
        assert anonymous.get("/api/raw/anything").status_code == 401

    def test_bearer_and_cookie_are_accepted(self, anonymous: TestClient) -> None:
        """A script sends the header; a paired browser sends the cookie."""
        bearer = {"Authorization": f"Bearer {TEST_TOKEN}"}
        assert anonymous.get("/api/access", headers=bearer).status_code == 200
        anonymous.cookies.set(access.COOKIE_NAME, TEST_TOKEN)
        assert anonymous.get("/api/access").status_code == 200

    def test_a_wrong_token_is_refused(self, anonymous: TestClient) -> None:
        """Presence is not enough."""
        assert (
            anonymous.get("/api/access", headers={"Authorization": "Bearer x"}).status_code == 401
        )

    def test_healthcheck_and_page_stay_open(self, anonymous: TestClient) -> None:
        """The compose healthcheck and the pairing screen itself need no token."""
        assert anonymous.get("/healthz").status_code == 200
        assert anonymous.get("/").status_code in {200, 503}

    @pytest.mark.parametrize("path", ["/ws/chat", "/ws/replay", "/ws/update"])
    def test_websocket_without_a_token_never_opens(self, anonymous: TestClient, path: str) -> None:
        """Refused before accept: no agent session, no replay."""
        with pytest.raises(WebSocketDisconnect), anonymous.websocket_connect(path):
            pass  # pragma: no cover — the connect itself raises


class TestPairing:
    """The pairing link trades the token for an HttpOnly, SameSite=Strict cookie."""

    def test_wrong_token_does_not_pair(self, anonymous: TestClient) -> None:
        """A guess gets a 403 and no cookie."""
        resp = anonymous.post("/api/pair", json={"token": "guess"})
        assert resp.status_code == 403
        assert access.COOKIE_NAME not in resp.cookies

    def test_right_token_pairs_the_browser(self, anonymous: TestClient) -> None:
        """After pairing, the cookie alone opens the API."""
        resp = anonymous.post("/api/pair", json={"token": TEST_TOKEN})
        assert resp.status_code == 200
        header = resp.headers["set-cookie"].lower()
        assert "httponly" in header
        assert "samesite=strict" in header
        assert anonymous.get("/api/access").status_code == 200


class TestToken:
    """Generated once, kept private, overridable."""

    def test_generated_token_is_stable_and_private(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A restart must not unpair every browser; other users must not read it."""
        monkeypatch.delenv(access.TOKEN_ENV)
        monkeypatch.setattr(access, "TOKEN_PATH", tmp_path / "state" / "ui_token.json")
        access.token.cache_clear()
        first = access.token()
        access.token.cache_clear()
        assert access.token() == first
        assert len(first) >= 32
        assert access.TOKEN_PATH.stat().st_mode & 0o077 == 0

    def test_a_short_configured_token_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A typo'd MEDMCP_UI_TOKEN must not become a guessable password."""
        monkeypatch.setenv(access.TOKEN_ENV, "abc")
        monkeypatch.setattr(access, "TOKEN_PATH", tmp_path / "ui_token.json")
        access.token.cache_clear()
        assert access.token() != "abc"

    def test_pairing_url(self) -> None:
        """The link people open once per browser."""
        assert access.pairing_url("http://localhost:8100/") == (
            f"http://localhost:8100/?token={TEST_TOKEN}"
        )


def test_a_racing_second_creator_adopts_the_first_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`medmcp url` and the starting server must agree on the token they print."""
    monkeypatch.delenv(access.TOKEN_ENV)
    monkeypatch.setattr(access, "TOKEN_PATH", tmp_path / "ui_token.json")
    first = access._create_token()  # pyright: ignore[reportPrivateUsage]
    second = access._create_token()  # pyright: ignore[reportPrivateUsage]
    assert first == second
    assert access.TOKEN_PATH.stat().st_mode & 0o077 == 0
    assert list(tmp_path.iterdir()) == [access.TOKEN_PATH]
