"""Tests for the release check and the update plan (``medmcp.update``).

Most of what matters here is what the plan refuses: an install the helper is
not designed to rewrite must get the host command, never a half-applied
update. The inspect records are shaped like ``docker inspect`` output from a
real ``oci://`` install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

# pyright: reportPrivateUsage=false
from medmcp import server, update

JsonDict = dict[str, Any]

_WS_HEADERS = {"host": "127.0.0.1:8100", "origin": "http://127.0.0.1:8100"}
CACHE_DIR = "/home/op/.cache/docker-compose/cd9c97c4"
COMPOSE_TEXT = """
services:
  llm:
    image: ${OLLAMA_IMAGE:-ollama/ollama:0.33.2}
    environment:
      OLLAMA_KEEP_ALIVE: "${OLLAMA_KEEP_ALIVE:--1}"
    volumes:
      - "${OLLAMA_MODELS_DIR:-ollama-models}:/root/.ollama"
  medmcp:
    image: ghcr.io/medmcp/core:${MEDMCP_TAG:-main}
    environment:
      MEDMCP_GPU: "${MEDMCP_GPU:-all}"
      MEDMCP_CATALOG_URL: "${MEDMCP_CATALOG_URL:-/app/catalog.ghcr.json}"
      MEDMCP_SHIM_RELAY: "${MEDMCP_SHIM_RELAY:-live}"
      MEDMCP_LLM_SHIM: "${MEDMCP_LLM_SHIM:-1}"
      OLLAMA_MODEL: "${OLLAMA_MODEL:-muse-medmcp}"
"""


@pytest.fixture(autouse=True)
def _state_dir(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(update, "UPDATE_STATE_PATH", tmp_path / "update.json")
    monkeypatch.setattr(update, "UPDATE_RESULT_PATH", tmp_path / "update_result.json")
    monkeypatch.delenv("MEDMCP_UPDATE_CHECK", raising=False)
    monkeypatch.delenv("MEDMCP_UPDATE_URL", raising=False)


def _core_info() -> JsonDict:
    """A published-image, ``oci://``-installed core, as ``docker inspect`` reports it."""
    info: JsonDict = {
        "Id": "abc123def456",
        "Config": {
            "Image": "ghcr.io/medmcp/core:v0.2.3",
            "Labels": {
                "com.docker.compose.project": "medmcp",
                "com.docker.compose.project.config_files": f"{CACHE_DIR}/compose.yaml",
                "com.docker.compose.project.working_dir": CACHE_DIR,
                "com.docker.compose.service": "medmcp",
            },
            "Env": [
                "MEDMCP_WORKSPACE=/data/imaging",
                "MEDMCP_GPU=0",
                "OLLAMA_MODEL=muse-medmcp",
                "MEDMCP_CATALOG_URL=/app/catalog.ghcr.json",
                "MEDMCP_SHIM_RELAY=live",
                "MEDMCP_LLM_SHIM=1",
                "GHCR_TOKEN=",
                "PATH=/app/.venv/bin:/usr/bin",
                "MEDMCP_BUILD=v0.2.3",
            ],
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/run/user/1000/docker.sock",
                "Destination": "/var/run/docker.sock",
            },
            {"Type": "bind", "Source": "/data/imaging", "Destination": "/data/imaging"},
            {
                "Type": "bind",
                "Source": "/home/op/.docker/config.json",
                "Destination": "/run/host-docker-config.json",
            },
            {"Type": "volume", "Name": "medmcp_vibe-state", "Destination": "/app/.vibe/state"},
        ],
    }
    return info


def _llm_info() -> JsonDict:
    return {
        "Config": {"Image": "ollama/ollama:0.33.2", "Env": ["OLLAMA_KEEP_ALIVE=-1"]},
        "Mounts": [
            {"Type": "volume", "Name": "medmcp_ollama-models", "Destination": "/root/.ollama"}
        ],
    }


# ── Versions ──────────────────────────────────────────────────────────────────


class TestVersions:
    """Release-tag parsing and ordering."""

    @pytest.mark.parametrize(
        ("candidate", "current", "expected"),
        [
            ("v0.3.0", "0.2.3", True),
            ("0.2.4", "0.2.3", True),
            ("v1.0.0", "0.9.9", True),
            ("v0.2.3", "0.2.3", False),
            ("v0.2.2", "0.2.3", False),
            ("v0.3.0-rc1", "0.3.0", False),  # a pre-release never replaces its base
            ("v0.3.0", "0.3.0-rc1", True),  # but the release replaces the candidate
            ("v0.3.0", "0+unknown", False),  # a checkout without metadata is never "behind"
            ("latest", "0.2.3", False),
        ],
    )
    def test_is_newer(self, candidate: str, current: str, expected: bool) -> None:
        """Is newer."""
        assert update.is_newer(candidate, current) is expected

    def test_parse_release_reads_the_github_shape(self) -> None:
        """Parse release reads the github shape."""
        info = update.parse_release(
            {
                "tag_name": "v0.3.0",
                "body": "### Added\n- things",
                "html_url": "https://github.com/medmcp/medmcp/releases/tag/v0.3.0",
                "published_at": "2026-09-10T08:00:00Z",
            }
        )
        assert info is not None
        assert (info.version, info.tag) == ("0.3.0", "v0.3.0")
        assert info.notes.startswith("### Added")

    def test_parse_release_rejects_garbage(self) -> None:
        """Parse release rejects garbage."""
        assert update.parse_release({"tag_name": "nightly"}) is None
        assert update.parse_release(["v0.3.0"]) is None
        assert update.parse_release(None) is None

    def test_tag_is_kept_to_image_tag_characters(self) -> None:
        """The tag becomes an image reference, so anything odd in it is refused."""
        assert update.parse_version("v0.3.0-rc.1") == ((0, 3, 0), "rc.1")
        assert update.parse_version("v0.3.0-$(rm)") is None
        assert update.parse_version("v0.3.0-a b") is None
        assert update.parse_release({"tag_name": "v0.3.0-x/y"}) is None

    def test_release_link_is_https_or_nothing(self) -> None:
        """The page link is the one thing the browser is pointed at from the document."""
        assert update.parse_release({"tag_name": "v1.0.0", "html_url": "javascript:x"})
        info = update.parse_release({"tag_name": "v1.0.0", "html_url": "javascript:x"})
        assert info is not None and info.url == ""
        info = update.parse_release({"tag_name": "v1.0.0", "html_url": "http://mirror/r"})
        assert info is not None and info.url == ""
        info = update.parse_release({"tag_name": "v1.0.0", "html_url": "https://mirror/r"})
        assert info is not None and info.url == "https://mirror/r"


# ── The check ─────────────────────────────────────────────────────────────────


class TestCheck:
    """Fetching, persisting and gating the release check."""

    @pytest.mark.asyncio
    async def test_fetch_uses_the_release_endpoint(self) -> None:
        """Fetch uses the release endpoint."""
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["ua"] = request.headers["user-agent"]
            return httpx.Response(200, json={"tag_name": "v0.3.0", "body": "notes"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            info = await update.fetch_latest_release(client=client)
        assert seen["url"] == update.DEFAULT_UPDATE_URL
        # The request names the product and nothing else — not even the version.
        assert seen["ua"] == "medmcp"
        assert info.version == "0.3.0"

    @pytest.mark.asyncio
    async def test_fetch_reads_a_local_mirror_file(self, tmp_path: Path) -> None:
        """An air-gapped mirror (or a test) can supply the release document as a file."""
        doc = tmp_path / "latest.json"
        doc.write_text(json.dumps({"tag_name": "v9.9.9"}))
        info = await update.fetch_latest_release(str(doc))
        assert info.tag == "v9.9.9"

    @pytest.mark.asyncio
    async def test_run_check_records_the_release(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Run check records the release."""
        doc = tmp_path / "latest.json"
        doc.write_text(json.dumps({"tag_name": "v0.3.0", "body": "n"}))
        monkeypatch.setenv("MEDMCP_UPDATE_URL", str(doc))
        state = await update.run_check()
        assert cast("JsonDict", state["latest"])["version"] == "0.3.0"
        assert state["error"] is None
        assert state["checked_at"]
        assert update.check_due(state) is False

    @pytest.mark.asyncio
    async def test_failed_check_keeps_the_last_release(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An outage must not turn a known update into "up to date"."""
        update.save_state({**update.load_state(), "latest": {"version": "0.3.0", "tag": "v0.3.0"}})
        monkeypatch.setenv("MEDMCP_UPDATE_URL", str(tmp_path / "missing.json"))
        state = await update.run_check(force=True)
        assert cast("JsonDict", state["latest"])["version"] == "0.3.0"
        assert state["error"]

    @pytest.mark.asyncio
    async def test_disabled_check_touches_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Disabled check touches nothing."""
        monkeypatch.setenv("MEDMCP_UPDATE_CHECK", "0")
        monkeypatch.setenv("MEDMCP_UPDATE_URL", str(tmp_path / "unused.json"))
        state = await update.run_check(force=True)
        assert state["checked_at"] is None
        assert not update.UPDATE_STATE_PATH.exists()

    @pytest.mark.asyncio
    async def test_manual_check_is_rate_limited(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Manual check is rate limited."""
        doc = tmp_path / "latest.json"
        doc.write_text(json.dumps({"tag_name": "v0.3.0"}))
        monkeypatch.setenv("MEDMCP_UPDATE_URL", str(doc))
        first = await update.run_check(force=True)
        doc.write_text(json.dumps({"tag_name": "v0.4.0"}))
        second = await update.run_check(force=True)
        assert second["checked_at"] == first["checked_at"]
        assert cast("JsonDict", second["latest"])["version"] == "0.3.0"

    @pytest.mark.asyncio
    async def test_auto_check_off_stops_the_daily_check_but_not_a_manual_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The operator's switch governs the unattended request only."""
        doc = tmp_path / "latest.json"
        doc.write_text(json.dumps({"tag_name": "v0.3.0"}))
        monkeypatch.setenv("MEDMCP_UPDATE_URL", str(doc))
        update.set_auto_check(False)
        assert update.auto_check_enabled() is False
        assert update.check_enabled() is True
        state = await update.run_check()
        assert state["checked_at"] is None
        state = await update.run_check(force=True)
        assert cast("JsonDict", state["latest"])["version"] == "0.3.0"
        assert state["auto_check"] is False

    @pytest.mark.asyncio
    async def test_env_off_forbids_manual_checks_too(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """MEDMCP_UPDATE_CHECK=0 is the deployment's decision; the UI cannot override it."""
        monkeypatch.setenv("MEDMCP_UPDATE_CHECK", "0")
        monkeypatch.setenv("MEDMCP_UPDATE_URL", str(tmp_path / "latest.json"))
        state = await update.run_check(force=True)
        assert state["checked_at"] is None
        assert update.auto_check_enabled() is False

    def test_dismiss_and_ack_round_trip(self) -> None:
        """Dismiss and ack round trip."""
        update.save_state({**update.load_state(), "last_result": {"status": "ok"}})
        assert update.dismiss("0.3.0")["dismissed"] == "0.3.0"
        assert update.ack_result()["last_result"] is None

    def test_collect_updater_result_folds_and_removes_the_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Collect updater result folds and removes the file."""
        monkeypatch.setattr(update, "remove_stale_updater", lambda: None)
        update.UPDATE_RESULT_PATH.write_text(json.dumps({"status": "ok", "to": "v0.3.0"}))
        result = update.collect_updater_result()
        assert result == {"status": "ok", "to": "v0.3.0"}
        assert not update.UPDATE_RESULT_PATH.exists()
        assert update.load_state()["last_result"] == result


# ── The plan ──────────────────────────────────────────────────────────────────


class TestPlan:
    """What the plan derives from an inspect record, and what it refuses."""

    def test_oci_install_is_planned(self) -> None:
        """Oci install is planned."""
        defaults = update.compose_defaults(COMPOSE_TEXT)
        plan = update.plan_apply(_core_info(), _llm_info(), defaults=defaults)
        assert plan.image("v0.3.0") == "ghcr.io/medmcp/core:v0.3.0"
        assert plan.current_tag == "v0.2.3"
        assert plan.project == "medmcp"
        assert plan.project_dir == CACHE_DIR
        assert plan.state_volume == "medmcp_vibe-state"
        assert plan.socket_source == "/run/user/1000/docker.sock"

    def test_only_operator_set_values_are_carried(self) -> None:
        """A value equal to the compose default, or empty, is left to the new release's default."""
        defaults = update.compose_defaults(COMPOSE_TEXT)
        info = _core_info()
        info["Config"]["Env"].append("MEDMCP_STACK_POOL=")
        plan = update.plan_apply(info, _llm_info(), defaults=defaults)
        assert plan.env == {
            "MEDMCP_WORKSPACE": "/data/imaging",
            "MEDMCP_GPU": "0",
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "HOME": "/home/op",
        }

    def test_llm_overrides_are_carried(self) -> None:
        """Llm overrides are carried."""
        llm = _llm_info()
        llm["Config"] = {"Image": "ollama/ollama:0.34.0", "Env": ["OLLAMA_KEEP_ALIVE=30m"]}
        llm["Mounts"] = [{"Type": "bind", "Source": "/srv/models", "Destination": "/root/.ollama"}]
        plan = update.plan_apply(_core_info(), llm, defaults=update.compose_defaults(COMPOSE_TEXT))
        assert plan.env["OLLAMA_IMAGE"] == "ollama/ollama:0.34.0"
        assert plan.env["OLLAMA_KEEP_ALIVE"] == "30m"
        assert plan.env["OLLAMA_MODELS_DIR"] == "/srv/models"

    def test_source_build_is_refused(self) -> None:
        """Source build is refused."""
        info = _core_info()
        info["Config"]["Image"] = "medmcp-core:dev"
        with pytest.raises(update.UpdateNotApplicableError, match="locally built"):
            update.plan_apply(info, None)

    def test_hand_managed_compose_files_are_refused_with_the_command(self) -> None:
        """Hand managed compose files are refused with the command."""
        info = _core_info()
        info["Config"]["Labels"]["com.docker.compose.project.config_files"] = (
            "/srv/medmcp/docker-compose.yml,/srv/medmcp/override.yml"
        )
        with pytest.raises(update.UpdateNotApplicableError) as exc:
            update.plan_apply(info, None)
        assert "local compose files" in str(exc.value)
        assert exc.value.command == (
            "docker compose -f /srv/medmcp/docker-compose.yml -f /srv/medmcp/override.yml "
            "up -d --pull always"
        )

    def test_a_previous_ui_update_is_still_updatable(self) -> None:
        """After an update the labels name the baked compose file, not the oci cache."""
        info = _core_info()
        info["Config"]["Labels"]["com.docker.compose.project.config_files"] = str(
            update.BAKED_COMPOSE_PATH
        )
        plan = update.plan_apply(info, None)
        assert plan.project_dir == CACHE_DIR

    def test_not_compose_managed_is_refused(self) -> None:
        """Not compose managed is refused."""
        info = _core_info()
        info["Config"]["Labels"] = {}
        with pytest.raises(update.UpdateNotApplicableError, match="not managed by docker compose"):
            update.plan_apply(info, None)

    def test_missing_state_volume_is_refused(self) -> None:
        """Missing state volume is refused."""
        info = _core_info()
        info["Mounts"] = [m for m in info["Mounts"] if m["Destination"] != "/app/.vibe/state"]
        with pytest.raises(update.UpdateNotApplicableError, match="named volume"):
            update.plan_apply(info, None)

    def test_updater_command_is_the_whole_contract(self) -> None:
        """Updater command is the whole contract."""
        plan = update.plan_apply(
            _core_info(), _llm_info(), defaults=update.compose_defaults(COMPOSE_TEXT)
        )
        args = update.updater_command(plan, "v0.3.0")
        joined = " ".join(args)
        assert args[:2] == ["run", "-d"]
        assert args[-3:] == ["medmcp-update", "ghcr.io/medmcp/core:v0.3.0", "apply"]
        assert "--entrypoint" in args
        assert "/run/user/1000/docker.sock:/var/run/docker.sock" in args
        assert "medmcp_vibe-state:/app/.vibe/state" in args
        assert "/home/op/.docker/config.json:/run/host-docker-config.json:ro" in args
        assert "MEDMCP_UPDATE_REPLACE=abc123def456" in args
        assert "MEDMCP_UPDATE_PREVIOUS_TAG=v0.2.3" in args
        assert f"MEDMCP_UPDATE_PROJECT_DIR={CACHE_DIR}" in args
        env_arg = next(a for a in args if a.startswith("MEDMCP_UPDATE_ENV="))
        assert json.loads(env_arg.partition("=")[2]) == plan.env
        assert "medmcp-workspace" not in joined

    def test_pulled_image_must_carry_the_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A registry answering the tag with a differently labelled image is not run."""
        labelled = {"v": "v0.2.9"}

        def _run(args: list[str], *, timeout: float = 60.0) -> str:
            assert args[:2] == ["image", "inspect"]
            return json.dumps(
                [{"Config": {"Labels": {"org.opencontainers.image.version": labelled["v"]}}}]
            )

        monkeypatch.setattr(update, "_run", _run)
        with pytest.raises(RuntimeError, match=r"labelled 'v0\.2\.9'"):
            update.verify_pulled_image("ghcr.io/medmcp/core:v0.3.0", "v0.3.0")
        labelled["v"] = "v0.3.0"
        update.verify_pulled_image("ghcr.io/medmcp/core:v0.3.0", "v0.3.0")

    def test_updater_log_line_redacts_the_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The token can be in the env the helper gets; it must not be in the log."""
        plan = update.plan_apply(_core_info(), None)
        plan.env["GHCR_TOKEN"] = "ghp_secret"
        seen: list[str] = []

        def _run(args: list[str], *, timeout: float = 60.0) -> str:
            return "cid\n"

        def _info(msg: str, *a: object) -> None:
            seen.append(msg % a)

        monkeypatch.setattr(update, "remove_stale_updater", lambda: None)
        monkeypatch.setattr(update, "_run", _run)
        monkeypatch.setattr(update.log, "info", _info)
        assert update.start_updater(plan, "v0.3.0") == "cid"
        assert "ghp_secret" not in seen[0]
        assert "MEDMCP_UPDATE_ENV=<redacted>" in seen[0]

    def test_compose_defaults(self) -> None:
        """Compose defaults."""
        assert update.compose_defaults(COMPOSE_TEXT)["MEDMCP_GPU"] == "all"
        assert update.compose_defaults(COMPOSE_TEXT)["OLLAMA_KEEP_ALIVE"] == "-1"


# ── Status / API ──────────────────────────────────────────────────────────────


class TestStatus:
    """The API payload and the endpoints over it."""

    def test_status_reports_availability_and_the_host_commands(self) -> None:
        """Status reports availability and the host commands."""
        state = {**update.load_state(), "latest": {"version": "0.3.0", "tag": "v0.3.0"}}
        out = update.status(current_version="0.2.3", build="v0.2.3", plan_error=None, state=state)
        assert out["available"] is True
        assert out["can_apply"] is True
        assert out["host_commands"] == {
            "update": "docker compose -f oci://ghcr.io/medmcp/compose:v0.3.0 up -d --pull always",
            "rollback": "docker compose -f oci://ghcr.io/medmcp/compose:v0.2.3 up -d --pull always",
        }

    def test_status_with_a_reason_cannot_apply(self) -> None:
        """Status with a reason cannot apply."""
        state = {**update.load_state(), "latest": {"version": "0.3.0", "tag": "v0.3.0"}}
        err = update.PlanError("nope", "docker compose -f mine.yml up -d --pull always")
        out = update.status(current_version="0.2.3", build="", plan_error=err, state=state)
        assert out["available"] is True
        assert out["can_apply"] is False
        assert out["apply_reason"] == "nope"
        # The install-specific command replaces the generic one.
        assert out["host_commands"]["update"] == "docker compose -f mine.yml up -d --pull always"
        assert out["host_commands"]["rollback"].endswith("compose:latest up -d --pull always")

    def test_status_up_to_date(self) -> None:
        """Status up to date."""
        state = {**update.load_state(), "latest": {"version": "0.2.3", "tag": "v0.2.3"}}
        out = update.status(current_version="0.2.3", build="v0.2.3", plan_error=None, state=state)
        assert out["available"] is False
        assert out["host_commands"] is None

    def test_endpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Endpoints."""
        monkeypatch.setattr(server, "__version__", "0.2.3")
        monkeypatch.setattr(update, "plan_error", lambda: update.PlanError("not here"))
        update.save_state(
            {**update.load_state(), "latest": {"version": "0.3.0", "tag": "v0.3.0", "notes": "x"}}
        )
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        body = client.get("/api/update").json()
        assert body["available"] is True
        assert body["dismissed"] is False
        assert body["apply_reason"] == "not here"
        body = client.post("/api/update/dismiss", json={"version": "0.3.0"}).json()
        assert body["dismissed"] is True
        body = client.post("/api/update/auto-check", json={"enabled": False}).json()
        assert body["auto_check"] is False
        assert body["enabled"] is True

    def test_auto_check_endpoint_refuses_when_env_forbids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the deployment's switch off there is nothing for the UI to turn on."""
        monkeypatch.setenv("MEDMCP_UPDATE_CHECK", "0")
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        assert client.post("/api/update/auto-check", json={"enabled": True}).status_code == 400

    def test_ws_update_refuses_a_version_not_on_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The image pulled is always the one the check found, never a client-chosen one."""
        monkeypatch.setattr(server, "__version__", "0.2.3")
        update.save_state({**update.load_state(), "latest": {"version": "0.3.0", "tag": "v0.3.0"}})
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        with client.websocket_connect("/ws/update", headers=_WS_HEADERS) as ws:
            ws.send_json({"version": "0.4.0"})
            frame = ws.receive_json()
        assert frame["type"] == "error"

    def test_ws_update_relays_the_plan_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ws update relays the plan reason."""
        monkeypatch.setattr(server, "__version__", "0.2.3")

        def _refuse() -> update.ApplyPlan:
            raise update.UpdateNotApplicableError("run it on the host")

        monkeypatch.setattr(update, "current_plan", _refuse)
        update.save_state({**update.load_state(), "latest": {"version": "0.3.0", "tag": "v0.3.0"}})
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        with client.websocket_connect("/ws/update", headers=_WS_HEADERS) as ws:
            ws.send_json({"version": "0.3.0"})
            frame = ws.receive_json()
        assert frame == {"type": "error", "message": "run it on the host"}


# ── Rehearsal ─────────────────────────────────────────────────────────────────


class TestRehearsal:
    """A ``simulate`` key plays the flow through without docker."""

    @pytest.fixture(autouse=True)
    def _real_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(update, "_simulated_version", None)
        monkeypatch.setattr(server, "__version__", "0.2.3")

    def test_status_presents_the_stand_in_release(self) -> None:
        """The rehearsed version is offered and can be applied, whatever the plan says."""
        state = {**update.load_state(), "simulate": "0.9.0"}
        out = update.status(
            current_version="0.2.3", build="v0.2.3", plan_error=update.PlanError("no"), state=state
        )
        assert out["rehearsal"] is True
        assert out["available"] is True
        assert out["can_apply"] is True
        assert cast("JsonDict", out["latest"])["tag"] == "v0.9.0"
        assert "Rehearsal" in cast("JsonDict", out["latest"])["notes"]

    def test_a_rehearsal_older_than_the_install_is_ignored(self) -> None:
        """Nothing to walk through if the stand-in would not be an update."""
        state = {**update.load_state(), "simulate": "0.1.0"}
        out = update.status(current_version="0.2.3", build="v0.2.3", plan_error=None, state=state)
        assert out["rehearsal"] is False
        assert out["available"] is False

    @pytest.mark.asyncio
    async def test_run_rehearsal_records_and_flips(self) -> None:
        """Progress is played, the outcome recorded, the key cleared, the version flipped."""
        update.save_state({**update.load_state(), "simulate": "0.9.0"})
        lines: list[str] = []
        await update.run_rehearsal(
            "0.9.0", lines.append, previous="v0.2.3", step_delay=0, restart_delay=0
        )
        assert lines[0].startswith("v0.9.0: Pulling")
        assert lines[-1].endswith("core:v0.9.0")
        state = update.load_state()
        assert state["simulate"] is None
        assert cast("JsonDict", state["last_result"])["status"] == "ok"
        assert update.effective_version("0.2.3") == "0.9.0"
        # Afterwards the install reads as up to date and no rehearsal is pending.
        out = update.status(current_version="0.2.3", build="v0.2.3", plan_error=None)
        assert out["current"]["version"] == "0.9.0"
        assert out["available"] is False

    def test_ws_update_plays_the_rehearsal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The socket streams the scripted download and ends with ``restarting``."""
        monkeypatch.setattr(update, "_REHEARSAL_LINES", ("{tag}: one", "{tag}: two"))
        update.save_state({**update.load_state(), "simulate": "0.9.0"})
        client = TestClient(server.app, base_url="http://127.0.0.1:8100")
        with client.websocket_connect("/ws/update", headers=_WS_HEADERS) as ws:
            ws.send_json({"version": "0.9.0"})
            frames = [ws.receive_json() for _ in range(3)]
        assert [f["type"] for f in frames] == ["progress", "progress", "restarting"]
        assert frames[0]["line"] == "v0.9.0: one"
        assert frames[2]["version"] == "0.9.0"
        assert update.load_state()["simulate"] is None


class TestCurrentDigest:
    """The running image's registry digest rides along, for a digest-pinned rollback."""

    def test_plan_records_the_running_digest(self) -> None:
        """RepoDigests of the container's image, matched on the repository."""
        info = _core_info()
        info["Image_RepoDigests"] = [
            "mirror/core@sha256:" + "11" * 32,
            "ghcr.io/medmcp/core@sha256:" + "22" * 32,
        ]
        plan = update.plan_apply(info, None)
        assert plan.current_digest == "sha256:" + "22" * 32
        assert "MEDMCP_UPDATE_PREVIOUS_DIGEST=sha256:" + "22" * 32 in update.updater_command(
            plan, "v0.3.0"
        )

    def test_plan_without_digest_is_still_a_plan(self) -> None:
        """A local build has none; the rollback then goes by tag."""
        plan = update.plan_apply(_core_info(), None)
        assert plan.current_digest is None
        assert "MEDMCP_UPDATE_PREVIOUS_DIGEST=" in update.updater_command(plan, "v0.3.0")
