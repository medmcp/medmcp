"""The path guard turns back a call its tool's input schema rejects (`medmcp.pathguard`).

vibe 2.26 validates the arguments a ``pre_tool`` hook lets through once more,
and an invalid set there ends the chat turn instead of failing the call
(``Core rejected durable input``). A denial skips that check, so the guard
denies such a call itself, with what is wrong and what the argument takes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from medmcp import brokerclient, pathguard, server
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool

from ._pool import fake_spec, resolver

JsonDict = dict[str, Any]

# A tool as pydantic writes it when an argument is a nested model: a $ref into
# $defs, which the harness sandbox does not resolve before the hook.
PICK: JsonDict = {
    "type": "object",
    "$defs": {
        "Criteria": {
            "type": "object",
            "properties": {
                "labels": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "mode": {"enum": ["any", "all"]},
            },
        }
    },
    "properties": {
        "source": {"type": "string"},
        "criteria": {"$ref": "#/$defs/Criteria"},
    },
    "required": ["source", "criteria"],
}

ROUTE = "mcp_demo_stack.pick"
# A list inside the list of labels: what ended the turn.
NESTED: JsonDict = {"source": "alpha", "criteria": {"labels": [["a", "b"]]}}


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ledger and broker socket under the test's own VIBE_HOME (no core to ask)."""
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
    monkeypatch.delenv("MEDMCP_BROKER_SOCK", raising=False)
    monkeypatch.setenv("MEDMCP_WORKSPACE", str(tmp_path))


def _known(monkeypatch: pytest.MonkeyPatch, schema: JsonDict | None) -> list[str]:
    """Answer every schema lookup with *schema*; returns the routes asked about."""
    asked: list[str] = []

    def lookup(route: str) -> JsonDict | None:
        asked.append(route)
        return schema

    monkeypatch.setattr(pathguard, "_known_schema", lookup)
    return asked


def _invocation(tool_input: object, tool_name: str = ROUTE) -> JsonDict:
    return {
        "hook_event_name": "pre_tool",
        "session_id": "sess-1",
        "cwd": "/nonexistent",
        "tool_name": tool_name,
        "tool_call_id": "call-1",
        "tool_input": tool_input,
    }


class TestSchemaProblem:
    """The text the model receives for an invalid call."""

    def test_a_nested_list(self) -> None:
        """Names the argument, the error and every shape the argument may take."""
        reason = pathguard.schema_problem("pick", NESTED, PICK)
        assert reason is not None
        assert reason.startswith("pick arguments are invalid: criteria.labels[0]: ")
        assert "is not of type 'string'" in reason
        assert "Expected: criteria.labels takes string or array of string." in reason
        assert "call pick again" in reason

    def test_a_valid_call_has_no_problem(self) -> None:
        """Both shapes of the anyOf are accepted."""
        for labels in ("a", ["a", "b"]):
            args = {"source": "alpha", "criteria": {"labels": labels}}
            assert pathguard.schema_problem("pick", args, PICK) is None

    def test_an_enum_lists_its_values(self) -> None:
        """A value outside an enum is shown the values it may take."""
        args = {"source": "alpha", "criteria": {"mode": "some"}}
        reason = pathguard.schema_problem("pick", args, PICK)
        assert reason is not None
        assert 'criteria.mode takes one of "any", "all"' in reason

    def test_a_missing_parameter_names_the_required_ones(self) -> None:
        """A required parameter left out is reported at the root."""
        reason = pathguard.schema_problem("pick", {"criteria": {}}, PICK)
        assert reason is not None
        assert "(arguments): 'source' is a required property" in reason
        assert "requires 'source', 'criteria'" in reason

    def test_hint_follows_refs_and_stays_short(self) -> None:
        """A $ref is rendered as what it points at; deep nesting is cut off."""
        assert (
            pathguard.schema_hint({"$ref": "#/$defs/Criteria"}, PICK) == "object with labels, mode"
        )
        deep: JsonDict = {"type": "array", "items": {"type": "array", "items": {"type": "array"}}}
        nested: JsonDict = {"type": "array", "items": deep}
        assert pathguard.schema_hint(nested, {}).endswith("…")


class TestDecide:
    """Where the schema check sits in the guard's decision."""

    def test_the_invalid_call_is_denied_not_let_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call that ended the turn now comes back to the model as a denial."""
        asked = _known(monkeypatch, PICK)
        out = pathguard.decide(_invocation(NESTED))
        assert out["decision"] == "deny"
        assert out["reason"].startswith("pick arguments are invalid")
        assert asked == [ROUTE]

    def test_a_valid_call_goes_on_to_the_path_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid arguments are allowed as before (no paths to object to here)."""
        _known(monkeypatch, PICK)
        args = {"source": "alpha", "criteria": {"labels": "a"}}
        assert pathguard.decide(_invocation(args)) == {"decision": "allow"}

    def test_unknown_tool_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No schema known for the tool: allowed exactly as before."""
        _known(monkeypatch, None)
        assert pathguard.decide(_invocation(NESTED)) == {"decision": "allow"}

    def test_no_core_to_ask_is_allowed(self) -> None:
        """No broker socket: the lookup finds nothing and the call goes on."""
        assert pathguard.decide(_invocation(NESTED)) == {"decision": "allow"}

    def test_a_broken_lookup_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any error while checking lets the call through, as a broken guard must."""

        def boom(route: str) -> JsonDict | None:
            raise RuntimeError("broker exploded")

        monkeypatch.setattr(pathguard, "_known_schema", boom)
        assert pathguard.decide(_invocation(NESTED)) == {"decision": "allow"}

    def test_an_unusable_schema_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A schema jsonschema cannot use is no reason to refuse the call."""
        _known(monkeypatch, {"type": "object", "properties": {"x": {"pattern": "("}}})
        assert pathguard.decide(_invocation({"x": "a"})) == {"decision": "allow"}

    def test_the_denial_is_not_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unlike a path denial, giving up here would end the turn; the ledger stays out of it."""
        _known(monkeypatch, PICK)
        for _ in range(pathguard._MAX_DENIALS + 2):
            assert pathguard.decide(_invocation(NESTED))["decision"] == "deny"
        assert not pathguard._ledger_path().exists()

    def test_harness_tools_are_never_looked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The harness checks its own tools; the core is not asked about them."""

        def no_core(op: str, payload: JsonDict, **_kw: object) -> JsonDict | None:
            raise AssertionError(f"asked the core about {payload}")

        monkeypatch.setattr(brokerclient, "ask_sync", no_core)
        for route in ("file_system.bash", "vibe.todo", "run_typescript", "skill.read"):
            assert pathguard.check_schema(route, {"command": 3}) is None

    def test_main_prints_the_denial(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The hook's stdout carries the deny decision vibe maps to a skipped call."""
        _known(monkeypatch, PICK)
        monkeypatch.setattr("sys.stdin.read", lambda: json.dumps(_invocation(NESTED)))
        assert pathguard.main() == 0
        assert json.loads(capsys.readouterr().out)["decision"] == "deny"


class _KnownPool:
    """Stands in for the pool: known tool lists, no backends."""

    def __init__(self, known: dict[str, dict[str, JsonDict]]) -> None:
        self.known = known

    def known_stacks(self) -> list[str]:
        return list(self.known)

    def known_tools(self, name: str) -> dict[str, JsonDict] | None:
        return self.known.get(name)


_POOL = _KnownPool({"demo-stack": {"pick": PICK}, "demo": {"other": {"type": "object"}}})


class TestToolSchemaOp:
    """The broker op the guard asks: answered from the pool's known tool lists."""

    @pytest.fixture(autouse=True)
    def pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The core's pool, with one stack's list known."""
        monkeypatch.setattr(server, "_pool", _POOL)

    def test_registered(self) -> None:
        """The broker serves it."""
        assert server._core_ops["tool_schema"] is server._tool_schema_op

    def test_by_route(self) -> None:
        """The harness route finds the stack by its group spelling."""
        assert server._tool_schema_op({"tool": ROUTE}) == {"schema": PICK}

    def test_by_medmcp_name(self) -> None:
        """``<stack>_<tool>`` binds to the longest stack name, not a prefix of it."""
        assert server._tool_schema_op({"tool": "demo-stack_pick"}) == {"schema": PICK}

    def test_unknown_tool_or_stack(self) -> None:
        """Nothing known: no schema, and nothing is started to find one."""
        assert server._tool_schema_op({"tool": "mcp_demo_stack.nope"}) == {"schema": None}
        assert server._tool_schema_op({"tool": "mcp_elsewhere.pick"}) == {"schema": None}
        assert server._tool_schema_op({}) == {"schema": None}

    def test_no_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A core without a stack pool knows no schemas."""
        monkeypatch.setattr(server, "_pool", None)
        assert server._tool_schema_op({"tool": ROUTE}) == {"schema": None}


async def test_the_guard_asks_the_core_over_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the hook's lookup reaches tool_schema on the socket in $VIBE_HOME."""
    vibe_home = tmp_path / "vibe"
    vibe_home.mkdir()
    monkeypatch.setattr(server, "_pool", _POOL)
    pool = BackendPool(resolve_spec=resolver({}))
    broker = BackendBroker(pool, vibe_home / "backend.sock", ops=server._core_ops)
    await broker.start()
    try:
        # The hook runs synchronously (asyncio.run), as in its own process.
        denial = await asyncio.to_thread(pathguard.check_schema, ROUTE, NESTED)
        assert denial is not None
        assert denial["decision"] == "deny"
        valid = {"source": "alpha", "criteria": {"labels": ["a"]}}
        assert await asyncio.to_thread(pathguard.check_schema, ROUTE, valid) is None
    finally:
        await broker.aclose()
        await pool.aclose()


class TestKnownTools:
    """The pool keeps each stack's input schemas once listed, without warming it again."""

    async def test_listed_tools_are_kept_after_eviction_and_restart(self, tmp_path: Path) -> None:
        """Listing records them (warmup left out); a new pool reads them back from the file."""
        cache = tmp_path / "tool_schemas.json"
        pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}), schema_cache=cache)
        try:
            assert pool.known_tools("fake") is None
            await pool.list_tools("fake")
            await pool.evict("fake")
            known = pool.known_tools("fake")
            assert known is not None
            assert "echo" in known
            assert "warmup" not in known
            assert known["echo"]["properties"]["text"]["type"] == "string"
            assert pool.known_stacks() == ["fake"]
        finally:
            await pool.aclose()
        again = BackendPool(resolve_spec=resolver({}), schema_cache=cache)
        assert again.known_tools("fake") == known
        assert again.warm_names() == []

    async def test_prewarm_records_them_too(self) -> None:
        """Activation pre-warm lists the tools, so the schemas are known from then on."""
        pool = BackendPool(resolve_spec=resolver({"fake": fake_spec("fake")}))
        try:
            await pool.prewarm(["fake"])
            assert "echo" in (pool.known_tools("fake") or {})
        finally:
            await pool.aclose()

    def test_a_broken_cache_is_ignored(self, tmp_path: Path) -> None:
        """An unreadable or odd file is an empty memory, not an error."""
        cache = tmp_path / "tool_schemas.json"
        cache.write_text("not json")
        assert BackendPool(resolve_spec=resolver({}), schema_cache=cache).known_stacks() == []
        cache.write_text(json.dumps({"s": {"t": {"type": "object"}, "u": 3}, "v": []}))
        pool = BackendPool(resolve_spec=resolver({}), schema_cache=cache)
        assert pool.known_tools("s") == {"t": {"type": "object"}}
        assert pool.known_stacks() == ["s"]
