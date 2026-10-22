"""The isolated agent's egress gateway: configured endpoints only, credentials added here."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from medmcp import gateway, settings

LOCAL = "http://127.0.0.1:11435/v1"
CONFIG = f"""\
[[providers]]
name = "ollama"
api_base = "{LOCAL}"
api_key_env_var = ""
api_style = "openai"
backend = "generic"
"""


def test_agent_config_points_every_endpoint_at_the_gateway() -> None:
    """Providers and external servers go through the gateway; no auth block survives."""
    cfg = {
        "providers": [
            {"name": "ollama", "api_base": LOCAL},
            {"name": "cloud", "api_base": "https://api.anthropic.com", "api_key_env_var": "K"},
        ],
        "mcp_servers": [
            {
                "name": "pubmed",
                "url": "https://mcp.example.org/sse?x=1",
                "auth": {"type": "static"},
            },
            {"name": "medmcp-neuro", "command": "medmcp-mcp-proxy", "args": ["medmcp-neuro"]},
        ],
    }
    out = gateway.agent_config(cfg, "http://172.19.0.2:11436")
    assert [p["api_base"] for p in out["providers"]] == [
        "http://172.19.0.2:11436/provider/ollama/v1",
        "http://172.19.0.2:11436/provider/cloud",
    ]
    assert out["mcp_servers"][0] == {
        "name": "pubmed",
        "url": "http://172.19.0.2:11436/mcp/pubmed/sse?x=1",
    }
    assert out["mcp_servers"][1] == cfg["mcp_servers"][1]


def test_placeholders_carry_names_not_values() -> None:
    """The agent learns which variables exist, never what they hold."""
    assert gateway.placeholder_env({"MEDMCP_CLOUD_API_KEY": "sk-real"}) == {
        "MEDMCP_CLOUD_API_KEY": gateway.PLACEHOLDER
    }


class _Body(httpx.AsyncByteStream):
    """A response body that streams, as a real upstream's does (MockTransport's is read)."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._data


class TestRoutes:
    """Against a recording upstream."""

    @pytest.fixture
    def seen(self) -> list[httpx.Request]:
        """Requests that reached the upstream."""
        return []

    @pytest.fixture
    def client(self, vibe_home: Path, seen: list[httpx.Request]) -> TestClient:
        """The gateway app over a mock upstream, with the shipped local provider."""
        (vibe_home / "config.toml").write_text(CONFIG)

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200, stream=_Body(b'{"ok": true}'), headers={"content-type": "application/json"}
            )

        upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return TestClient(gateway.EgressGateway("127.0.0.1", client=upstream).app)

    def test_chat_reaches_the_local_model_without_the_agents_headers(
        self, client: TestClient, seen: list[httpx.Request]
    ) -> None:
        """Forwarded to the configured origin; a credential the agent sent is dropped."""
        resp = client.post(
            "/provider/ollama/v1/chat/completions",
            json={"model": "m"},
            headers={"Authorization": "Bearer from-agent"},
        )
        assert resp.status_code == 200
        assert str(seen[0].url) == "http://127.0.0.1:11435/v1/chat/completions"
        assert "authorization" not in seen[0].headers

    @pytest.mark.parametrize("path", ["api/pull", "api/delete", "api/create", "v1/../api/pull"])
    def test_the_model_servers_admin_api_is_out_of_reach(
        self, client: TestClient, seen: list[httpx.Request], path: str
    ) -> None:
        """Only model endpoints are forwarded."""
        assert client.post(f"/provider/ollama/{path}", json={}).status_code == 403
        assert seen == []

    def test_a_provider_not_in_force_is_refused(
        self, client: TestClient, seen: list[httpx.Request]
    ) -> None:
        """The cloud route is closed while no cloud model is in force."""
        assert client.post("/provider/cloud/v1/messages", json={}).status_code == 403
        assert seen == []

    def test_the_cloud_key_is_added_here(
        self, client: TestClient, seen: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The agent never holds it: the gateway sets the provider's header."""
        cfg = CONFIG + '\n[[providers]]\nname = "cloud"\napi_base = "https://api.anthropic.com"\n'
        (settings.VIBE_HOME / "config.toml").write_text(cfg)
        monkeypatch.setattr(
            settings,
            "active_cloud_model",
            lambda: {
                "api_base": "https://api.anthropic.com",
                "api_style": "anthropic",
                "api_key_env": settings.CLOUD_MODEL_KEY_ENV,
            },
        )
        monkeypatch.setattr(settings, "load_cloud_model_key", lambda: "sk-real")
        resp = client.post(
            "/provider/cloud/v1/messages", json={}, headers={"x-api-key": gateway.PLACEHOLDER}
        )
        assert resp.status_code == 200
        assert str(seen[0].url) == "https://api.anthropic.com/v1/messages"
        assert seen[0].headers["x-api-key"] == "sk-real"

    def test_an_external_server_is_reached_with_its_token_only_while_on(
        self, client: TestClient, seen: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closed until switched on; then forwarded with the token the gateway holds."""
        assert client.post("/mcp/pubmed/mcp", json={}).status_code == 403
        monkeypatch.setenv("PUBMED_TOKEN", "t-real")
        settings.acknowledge_external_mcp()
        settings.set_external_mcp_enabled(True)
        settings.add_external_server(
            "pubmed", "streamable-http", "https://mcp.example.org/mcp", "PUBMED_TOKEN"
        )
        resp = client.post("/mcp/pubmed/mcp", json={"jsonrpc": "2.0"})
        assert json.loads(resp.text) == {"ok": True}
        assert str(seen[0].url) == "https://mcp.example.org/mcp"
        assert seen[0].headers["authorization"] == "Bearer t-real"
        settings.set_external_server_active("pubmed", False)
        assert client.post("/mcp/pubmed/mcp", json={}).status_code == 403
