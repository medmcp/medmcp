"""Tool effects: what a stack tool declares it does to the data, and where it shows.

The contract (``_meta["org.medmcp/effect"]``, else the MCP annotations), the
pool's memory of it, the approval card's labels, the destructive-tool approval
policy, the line the proxy adds for the model, and ``GET /api/stacks``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

import mcp.types as mcp_types
import pytest
from fastapi.testclient import TestClient

from medmcp import explain, riskcheck, server, settings, tooleffect
from medmcp.backend_pool import BackendPool
from medmcp.proxy import with_effect_line

from ._pool import FAKE_STACK_SERVER, broker, fake_spec, resolver
from .conftest import AUTH_HEADERS
from .test_proxy import _proxy_client  # pyright: ignore[reportPrivateUsage]
from .test_server import _StubWs  # pyright: ignore[reportPrivateUsage]

JsonDict = dict[str, Any]

# What the fake stack's tools declare (tests/fake_stack_server.py).
FAKE_EFFECTS = {
    "inspect_image": "read_only",
    "make_mask": "writes_new",
    "erase_file": "destructive",
}
FAKE_UNTAGGED = {"echo", "sleep", "crash", "warmup_count", "odd_effect"}


# ── the contract ─────────────────────────────────────────────────────────────


class TestDeclaredEffect:
    """``_meta`` first; the annotations only when it is absent; unset is untagged."""

    @pytest.mark.parametrize("value", ["read_only", "writes_new", "destructive"])
    def test_meta_names_the_effect(self, value: str) -> None:
        """Each contract value is read as it is."""
        assert tooleffect.declared_effect({"org.medmcp/effect": value}, None) == value

    def test_meta_wins_over_the_annotations(self) -> None:
        """A stack that says both is taken at its ``_meta`` word."""
        meta = {"org.medmcp/effect": "destructive"}
        assert tooleffect.declared_effect(meta, {"readOnlyHint": True}) == "destructive"

    @pytest.mark.parametrize("junk", ["harmless", "READ_ONLY", "", 1, None, ["read_only"]])
    def test_a_junk_meta_value_is_untagged_and_not_a_fallback(self, junk: object) -> None:
        """Present but outside the contract: ignored, the annotations not consulted."""
        meta = {"org.medmcp/effect": junk}
        assert tooleffect.declared_effect(meta, {"readOnlyHint": True}) is None

    @pytest.mark.parametrize(
        ("annotations", "expected"),
        [
            ({"readOnlyHint": True}, "read_only"),
            ({"readOnlyHint": True, "destructiveHint": True}, "read_only"),
            ({"destructiveHint": True}, "destructive"),
            ({"readOnlyHint": False, "destructiveHint": True}, "destructive"),
            ({"readOnlyHint": False, "destructiveHint": False}, "writes_new"),
            # Unset stays unset: the spec's destructiveHint=true default is not applied.
            ({"readOnlyHint": False}, None),
            ({"destructiveHint": False}, None),
            ({}, None),
            ({"readOnlyHint": None, "destructiveHint": None}, None),
            ({"readOnlyHint": "true"}, None),
            ({"idempotentHint": True, "openWorldHint": False}, None),
        ],
    )
    def test_annotations_stand_in_when_meta_is_absent(
        self, annotations: JsonDict, expected: str | None
    ) -> None:
        """Each combination of the two hints, with other ``_meta`` keys around."""
        assert tooleffect.declared_effect({"other": 1}, annotations) == expected
        assert tooleffect.declared_effect(None, annotations) == expected

    def test_nothing_declared_is_untagged(self) -> None:
        """No ``_meta``, no annotations."""
        assert tooleffect.declared_effect(None, None) is None

    def test_read_from_an_mcp_tool(self) -> None:
        """The MCP model's ``_meta`` alias and its annotations object are both read."""
        tool = mcp_types.Tool.model_validate(
            {
                "name": "t",
                "inputSchema": {"type": "object"},
                "annotations": {"readOnlyHint": False, "destructiveHint": False},
            }
        )
        assert tooleffect.tool_effect(tool) == "writes_new"
        tool = mcp_types.Tool.model_validate(
            {
                "name": "t",
                "inputSchema": {"type": "object"},
                "_meta": {"org.medmcp/effect": "read_only"},
                "annotations": {"destructiveHint": True},
            }
        )
        assert tooleffect.tool_effect(tool) == "read_only"
        unset = mcp_types.Tool(name="t", inputSchema={"type": "object"})
        assert tooleffect.tool_effect(unset) is None


class TestModelLine:
    """The one line the model gets, appended to an otherwise untouched description."""

    def test_each_effect_line(self) -> None:
        """The exact wording."""
        assert tooleffect.describe("Do it.", "read_only") == "Do it.\n\nEffect: read-only."
        assert tooleffect.describe("Do it.", "writes_new") == "Do it.\n\nEffect: writes new files."
        assert tooleffect.describe("Do it.", "destructive") == (
            "Do it.\n\nEffect: destructive (deletes, moves or changes the user's files)."
        )

    def test_the_description_is_kept_byte_for_byte(self) -> None:
        """Trailing whitespace and newlines included."""
        text = "Line one.  \n\n  Args:\n    x: thing\n"
        out = tooleffect.describe(text, "read_only")
        assert out is not None
        assert out.startswith(text)
        assert out == text + "\nEffect: read-only."

    def test_untagged_or_empty(self) -> None:
        """Nothing for an untagged tool; just the line for one without a description."""
        assert tooleffect.describe("Do it.", None) == "Do it."
        assert tooleffect.describe(None, None) is None
        assert tooleffect.describe(None, "destructive") == (
            "Effect: destructive (deletes, moves or changes the user's files)."
        )


# ── the pool's memory of it ──────────────────────────────────────────────────


class TestPoolKeepsEffects:
    """Listed with the schemas, written to the state file, dropped on evict."""

    async def test_round_trip_through_the_state_file_and_evict(self, tmp_path: Path) -> None:
        """Listing records each tagged tool's effect; a new pool reads them; evict forgets."""
        cache = tmp_path / "tool_schemas.json"
        pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}), schema_cache=cache)
        try:
            assert pool.known_effects("fake") is None
            await pool.list_tools("fake")
            assert pool.known_effects("fake") == FAKE_EFFECTS
            for tool in FAKE_UNTAGGED:
                assert pool.tool_effect("fake", tool) is None
            assert pool.tool_effect("fake", "erase_file") == "destructive"
            assert pool.tool_effect("elsewhere", "erase_file") is None
        finally:
            await pool.aclose()

        saved = json.loads(cache.read_text())
        assert saved["version"] == 2
        entry = saved["stacks"]["fake"]["make_mask"]
        assert entry["effect"] == "writes_new"
        assert entry["schema"]["properties"]["output_path"]["type"] == "string"
        assert saved["stacks"]["fake"]["echo"]["effect"] is None

        again = BackendPool(resolve_spec=resolver({}), schema_cache=cache)
        assert again.known_effects("fake") == FAKE_EFFECTS
        assert again.tool_effect("fake", "inspect_image") == "read_only"
        assert "echo" in (again.known_tools("fake") or {})

        await again.evict("fake")
        assert again.known_effects("fake") is None
        assert again.tool_effect("fake", "erase_file") is None
        third = BackendPool(resolve_spec=resolver({}), schema_cache=cache)
        assert third.known_effects("fake") is None
        assert third.tool_effect("fake", "erase_file") is None

    async def test_prewarm_records_them(self) -> None:
        """Activation pre-warm lists the tools, effects included."""
        pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}))
        try:
            await pool.prewarm(["fake"])
            assert pool.known_effects("fake") == FAKE_EFFECTS
        finally:
            await pool.aclose()

    def test_a_relisting_replaces_the_effects(self) -> None:
        """A stack that changes a tool's declaration is believed as of its latest list."""
        pool = BackendPool(resolve_spec=resolver({}))
        schema = {"type": "object"}

        def listed(meta: JsonDict | None) -> list[mcp_types.Tool]:
            return [mcp_types.Tool(name="t", inputSchema=schema, _meta=meta)]

        pool._remember("s", listed({"org.medmcp/effect": "writes_new"}))  # pyright: ignore[reportPrivateUsage]
        assert pool.tool_effect("s", "t") == "writes_new"
        pool._remember("s", listed({"org.medmcp/effect": "destructive"}))  # pyright: ignore[reportPrivateUsage]
        assert pool.tool_effect("s", "t") == "destructive"
        pool._remember("s", listed(None))  # pyright: ignore[reportPrivateUsage]
        assert pool.tool_effect("s", "t") is None
        assert pool.known_effects("s") == {}

    def test_a_file_from_before_effects_still_gives_the_schemas(self, tmp_path: Path) -> None:
        """The old ``{stack: {tool: schema}}`` shape: schemas known, effects not yet."""
        cache = tmp_path / "tool_schemas.json"
        cache.write_text(json.dumps({"s": {"t": {"type": "object"}}}))
        pool = BackendPool(resolve_spec=resolver({}), schema_cache=cache)
        assert pool.known_tools("s") == {"t": {"type": "object"}}
        assert pool.known_effects("s") == {}

    def test_an_odd_effect_in_the_file_is_ignored(self, tmp_path: Path) -> None:
        """A value outside the contract does not come back from disk."""
        cache = tmp_path / "tool_schemas.json"
        stacks = {"s": {"t": {"schema": {"type": "object"}, "effect": "harmless"}, "u": 3}}
        cache.write_text(json.dumps({"version": 2, "stacks": stacks}))
        pool = BackendPool(resolve_spec=resolver({}), schema_cache=cache)
        assert pool.known_tools("s") == {"t": {"type": "object"}}
        assert pool.tool_effect("s", "t") is None


# ── the approval card's labels ───────────────────────────────────────────────


class TestStackToolLabels:
    """A stack tool is labelled by its declaration, never by its argument names."""

    ARGS: JsonDict = {"input_path": "/w/t1.nii", "output_dir": "/w/out"}  # noqa: RUF012

    def test_untagged_gets_no_file_labels(self) -> None:
        """Before: ``file_read`` and, for an ``output_*`` argument, ``file_write``."""
        assert riskcheck.signals("medmcp-neuro_skull_strip", self.ARGS, stack_tool=True) == []

    @pytest.mark.parametrize("effect", ["read_only", "writes_new", "destructive"])
    def test_tagged_gets_exactly_its_effect(self, effect: str) -> None:
        """One label, the declared one."""
        keys = riskcheck.signals(
            "medmcp-neuro_skull_strip", self.ARGS, stack_tool=True, effect=effect
        )
        assert keys == [f"effect_{effect}"]

    def test_a_junk_effect_is_no_label(self) -> None:
        """Only contract values become a label."""
        assert riskcheck.signals("s_t", self.ARGS, stack_tool=True, effect="harmless") == []

    @pytest.mark.parametrize("effect", [None, "read_only"])
    def test_outside_workspace_still_follows(self, effect: str | None) -> None:
        """A read-only declaration cannot take the path label away; the effect comes first."""
        findings = [{"status": "outside_workspace", "severity": "warning"}]
        keys = riskcheck.signals(
            "medmcp-neuro_skull_strip", self.ARGS, findings, stack_tool=True, effect=effect
        )
        declared = [] if effect is None else ["effect_read_only"]
        assert keys == [*declared, "outside_workspace"]

    def test_builtin_tools_keep_their_rules(self) -> None:
        """Nothing changes for the built-in tools."""
        assert riskcheck.signals("write_file", {"path": "a.txt"}) == ["file_write"]
        assert riskcheck.signals("read_file", {"path": "a.txt"}) == ["file_read"]
        assert "file_delete" in riskcheck.signals("bash", {"command": "rm -rf x"})

    def test_effect_labels(self) -> None:
        """Short labels; the severity picks green, blue and red."""
        resolved = explain.resolve_risks(
            ["effect_read_only", "effect_writes_new", "effect_destructive", "network"]
        )
        assert [(r["label"], r["severity"]) for r in resolved] == [
            ("Read-only", "low"),
            ("Writes new files", "info"),
            ("Destructive", "high"),
            ("Contacts an external server or website", "medium"),
        ]

    def test_the_model_is_never_offered_an_effect(self) -> None:
        """They are the stack's word, not a risk for the explanation model to pick."""
        assert not set(explain.EFFECT_CATEGORIES) & set(explain.RISK_CATEGORIES)
        parsed = explain.parse_explanation_response(
            '{"explanation": "x", "risks": ["effect_read_only", "file_read"]}'
        )
        assert parsed == ("x", ["file_read"])


# ── the approval dialog and the destructive-tool policy ──────────────────────


class _EffectPool:
    """Stands in for the pool: declared effects, no backends."""

    def __init__(
        self, effects: dict[str, dict[str, str]], tools: dict[str, list[str]] | None = None
    ) -> None:
        self.effects = effects
        self.tools = tools or {name: list(effects[name]) for name in effects}

    def tool_effect(self, name: str, tool: str) -> str | None:
        return self.effects.get(name, {}).get(tool)

    def known_effects(self, name: str) -> dict[str, str] | None:
        return self.effects.get(name)

    def known_tools(self, name: str) -> dict[str, JsonDict] | None:
        tools = self.tools.get(name)
        return None if tools is None else {t: {} for t in tools}

    def known_stacks(self) -> list[str]:
        return list(self.effects)


STACK = "medmcp-fake"
_DOCKER_ARGS = ["run", "-i", "--rm", "--network", "none", "ghcr.io/x/fake:1"]


def _request(req_id: int, tool: str, **arguments: str) -> JsonDict:
    return {
        "id": req_id,
        "method": "session/request_permission",
        "params": {
            "toolCall": {
                "toolCallId": f"c{req_id}",
                "title": tool,
                "toolName": tool,
                "rawInput": arguments,
            },
            "options": [
                {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
                {"optionId": "reject_once", "name": "Deny", "kind": "reject_once"},
            ],
        },
    }


class TestApprovalDialog:
    """What the card carries for a stack tool, and when the chat-long option is withheld."""

    @pytest.fixture
    def pool(self, monkeypatch: pytest.MonkeyPatch) -> _EffectPool:
        """The core's pool, knowing what the fake stack declares."""
        pool = _EffectPool({STACK: dict(FAKE_EFFECTS)})
        monkeypatch.setattr(server, "_pool", pool)
        return pool

    @pytest.fixture
    def responses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pool: _EffectPool
    ) -> list[tuple[int, object]]:
        """Capture what is answered to vibe; provenance and explanations off."""
        del pool
        sent: list[tuple[int, object]] = []

        async def _respond(req_id: int, result: object) -> None:
            sent.append((req_id, result))

        monkeypatch.setattr(server._client, "respond", _respond)
        monkeypatch.setattr(settings, "load_provenance_enabled", lambda: False)
        monkeypatch.setattr(settings, "load_explain_enabled", lambda: False)
        monkeypatch.setattr(server, "WORKSPACE_ROOT", tmp_path)
        monkeypatch.delenv(server.STACK_TOOL_APPROVAL_ENV_VAR, raising=False)
        (tmp_path / "study").mkdir()
        return sent

    @staticmethod
    def _connection(args: list[str] | None = None) -> server._ChatConnection:
        queue: asyncio.Queue[JsonDict] = asyncio.Queue()
        servers: list[JsonDict] = [
            {"name": STACK, "command": "docker", "args": args or _DOCKER_ARGS, "env": {}}
        ]
        return server._ChatConnection(cast("Any", _StubWs()), "sid", queue, servers)

    @staticmethod
    def _dialogs(conn: server._ChatConnection) -> list[JsonDict]:
        sent = cast("list[JsonDict]", cast("_StubWs", conn.ws).sent)
        return [m for m in sent if m.get("type") == "permission_request"]

    async def _answer(self, conn: server._ChatConnection, req: JsonDict, option: str) -> None:
        task = asyncio.create_task(conn._handle_permission(req))
        await asyncio.sleep(0.05)
        conn._pending_perms[cast("int", req["id"])].set_result(option)
        await task

    @staticmethod
    def _keys(dialog: JsonDict) -> list[str]:
        return [r["key"] for r in cast("list[JsonDict]", dialog["risks"])]

    @staticmethod
    def _options(dialog: JsonDict) -> set[str]:
        return {o["optionId"] for o in cast("list[JsonDict]", dialog["options"])}

    @pytest.mark.parametrize(
        ("tool", "keys"),
        [
            ("inspect_image", ["effect_read_only"]),
            ("make_mask", ["effect_writes_new"]),
            ("erase_file", ["effect_destructive"]),
            ("echo", []),
        ],
    )
    async def test_the_card_shows_the_declared_effect_only(
        self, responses: list[tuple[int, object]], tmp_path: Path, tool: str, keys: list[str]
    ) -> None:
        """Each tool its one effect label (none untagged); no file_read/file_write guess."""
        del responses
        conn = self._connection()
        folder = str(tmp_path / "study")
        req = _request(1, f"{STACK}_{tool}", input_path=folder, output_dir=folder)
        await self._answer(conn, req, "allow_once")
        assert self._keys(self._dialogs(conn)[0]) == keys

    async def test_network_and_outside_workspace_follow_the_effect(
        self, responses: list[tuple[int, object]]
    ) -> None:
        """Effect first, then the path label, then the stack's network access."""
        del responses
        args = ["run", "-i", "--rm", "--network", "bridge", "ghcr.io/x/fake:1"]
        conn = self._connection(args)
        await self._answer(
            conn, _request(1, f"{STACK}_inspect_image", input_path="/etc"), "allow_once"
        )
        assert self._keys(self._dialogs(conn)[0]) == [
            "effect_read_only",
            "outside_workspace",
            "network",
        ]

    async def test_a_stack_without_network_gets_no_network_label(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """``--network none`` is the default install."""
        del responses
        conn = self._connection()
        req = _request(1, f"{STACK}_inspect_image", input_path=str(tmp_path / "study"))
        await self._answer(conn, req, "allow_once")
        assert "network" not in self._keys(self._dialogs(conn)[0])

    async def test_a_destructive_tool_is_never_offered_for_the_chat(
        self, responses: list[tuple[int, object]], tmp_path: Path
    ) -> None:
        """It asks on every call; read-only and writes-new keep the option."""
        conn = self._connection()
        folder = str(tmp_path / "study")
        await self._answer(conn, _request(1, f"{STACK}_erase_file", path=folder), "allow_once")
        await self._answer(conn, _request(2, f"{STACK}_erase_file", path=folder), "allow_once")
        await self._answer(conn, _request(3, f"{STACK}_make_mask", input_path=folder), "allow_once")
        dialogs = self._dialogs(conn)
        assert len(dialogs) == 3
        assert server._STACK_TOOL_CHAT_OPTION not in self._options(dialogs[0])
        assert server._STACK_TOOL_CHAT_OPTION not in self._options(dialogs[1])
        assert server._STACK_TOOL_CHAT_OPTION in self._options(dialogs[2])
        allow = {"outcome": {"outcome": "selected", "optionId": "allow_once"}}
        assert responses == [(1, allow), (2, allow), (3, allow)]

    async def test_an_existing_grant_never_covers_a_destructive_tool(
        self, responses: list[tuple[int, object]], tmp_path: Path, pool: _EffectPool
    ) -> None:
        """Granted while it said writes-new, then declared destructive: asked again."""
        conn = self._connection()
        folder = str(tmp_path / "study")
        pool.effects[STACK]["make_mask"] = "writes_new"
        await self._answer(
            conn,
            _request(1, f"{STACK}_make_mask", input_path=folder),
            server._STACK_TOOL_CHAT_OPTION,
        )
        assert (STACK, "make_mask") in conn._granted_stack_tools
        await conn._handle_permission(_request(2, f"{STACK}_make_mask", input_path=folder))
        assert len(self._dialogs(conn)) == 1  # covered while it is writes-new

        pool.effects[STACK]["make_mask"] = "destructive"
        await self._answer(
            conn, _request(3, f"{STACK}_make_mask", input_path=folder), "reject_once"
        )
        dialogs = self._dialogs(conn)
        assert len(dialogs) == 2
        assert server._STACK_TOOL_CHAT_OPTION not in self._options(dialogs[1])
        assert self._keys(dialogs[1]) == ["effect_destructive"]
        assert responses[-1] == (3, {"outcome": {"outcome": "selected", "optionId": "reject_once"}})

    def test_no_pool_means_no_declarations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Host-native default: every tool untagged, every stack without tools."""
        monkeypatch.setattr(server, "_pool", None)
        assert server.tool_effect(STACK, "erase_file") is None
        assert server.stack_tools(STACK) == []


# ── what the model sees ──────────────────────────────────────────────────────


class TestProxyDescriptions:
    """The proxy adds exactly one line to a tagged tool's description, nothing else."""

    @staticmethod
    async def _raw_descriptions() -> dict[str, str | None]:
        pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}))
        try:
            return {t.name: t.description for t in await pool.list_tools("fake")}
        finally:
            await pool.aclose()

    @staticmethod
    def _check(raw: dict[str, str | None], seen: dict[str, str | None]) -> None:
        for name, description in seen.items():
            effect = FAKE_EFFECTS.get(name)
            if effect is None:
                assert description == raw[name], name
            else:
                assert description == f"{raw[name]}\n\n{tooleffect.MODEL_LINES[effect]}", name
        assert set(FAKE_EFFECTS) | {"echo", "odd_effect"} <= set(seen)

    async def test_through_the_broker(self, tmp_path: Path) -> None:
        """The usual path: the broker relays ``_meta`` and the proxy reads it."""
        raw = await self._raw_descriptions()
        async with (
            broker(tmp_path) as sock,
            _proxy_client({"MEDMCP_BROKER_SOCK": str(sock)}) as session,
        ):
            seen = {t.name: t.description for t in (await session.list_tools()).tools}
        self._check(raw, seen)

    async def test_on_the_direct_spawn_fallback(self, tmp_path: Path) -> None:
        """No broker: the stack's own list, the same lines."""
        raw = await self._raw_descriptions()
        backends = tmp_path / "backends.json"
        spec: JsonDict = {
            "command": sys.executable,
            "args": [str(FAKE_STACK_SERVER)],
            "env": {},
            "cwd": str(tmp_path),  # not $MEDMCP_WORKSPACE, which another test may have set
        }
        backends.write_text(json.dumps({"fake": spec}))
        env = {
            "MEDMCP_BROKER_SOCK": str(tmp_path / "absent.sock"),
            "MEDMCP_BACKENDS_FILE": str(backends),
        }
        async with _proxy_client(env) as session:
            seen = {t.name: t.description for t in (await session.list_tools()).tools}
        self._check(raw, seen)

    def test_an_untagged_tool_is_returned_as_is(self) -> None:
        """Same object, nothing copied."""
        tool = mcp_types.Tool(name="t", description="x", inputSchema={"type": "object"})
        assert with_effect_line(tool) is tool


# ── the Tool stacks window ───────────────────────────────────────────────────


def test_api_stacks_lists_each_stacks_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tools: [{name, effect}]`` from the pool's last listing; empty when unknown."""
    installed: list[JsonDict] = [
        {"name": STACK, "image": "img", "gpu": False, "network": False, "jobs": []},
        {"name": "medmcp-other", "image": "img2", "gpu": False, "network": False, "jobs": []},
    ]
    monkeypatch.setattr(settings, "list_installed_stacks", lambda: installed)
    pool = _EffectPool({STACK: {"inspect_image": "read_only"}}, {STACK: ["inspect_image", "echo"]})
    monkeypatch.setattr(server, "_pool", pool)
    client = TestClient(server.app, base_url="http://127.0.0.1:8100", headers=AUTH_HEADERS)
    stacks = {s["name"]: s for s in client.get("/api/stacks").json()["stacks"]}
    assert stacks[STACK]["tools"] == [
        {"name": "inspect_image", "effect": "read_only"},
        {"name": "echo", "effect": None},
    ]
    assert stacks["medmcp-other"]["tools"] == []
    monkeypatch.setattr(server, "_pool", None)
    stacks = {s["name"]: s for s in client.get("/api/stacks").json()["stacks"]}
    assert stacks[STACK]["tools"] == []
