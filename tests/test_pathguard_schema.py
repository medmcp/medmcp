"""The path guard turns back a call its tool's input schema rejects (`medmcp.pathguard`).

vibe 2.26 validates the arguments a ``pre_tool`` hook lets through once more, and
an invalid set there kills the turn instead of failing the call (rehearsal D7:
``select_series`` with ``sequence: [["T2", "STIR"]]``). A denial skips that check,
so the guard denies such a call itself, with what is wrong and what the argument
takes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from medmcp import pathguard, plan, server
from medmcp.backend_broker import BackendBroker
from medmcp.backend_pool import BackendPool

from ._pool import resolver

JsonDict = dict[str, Any]

# select_series as pydantic writes it: the criteria are a $ref into $defs, which
# the harness sandbox does not resolve before the hook (so the call reached it).
SELECT_SERIES: JsonDict = {
    "type": "object",
    "$defs": {
        "SeriesCriteria": {
            "type": "object",
            "properties": {
                "sequence": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "contrast": {"enum": ["yes", "no"]},
            },
        }
    },
    "properties": {
        "index_path": {"type": "string"},
        "criteria": {"$ref": "#/$defs/SeriesCriteria"},
    },
    "required": ["index_path", "criteria"],
}

ROUTE = "mcp_medmcp_cohort.select_series"


@pytest.fixture(autouse=True)
def isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ledger and broker socket under the test's own VIBE_HOME (no core to ask)."""
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
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


D7_INPUT: JsonDict = {"index_path": "idx.csv", "criteria": {"sequence": [["T2", "STIR"]]}}


class TestSchemaProblem:
    """The text the model receives for an invalid call."""

    def test_the_d7_nested_list(self) -> None:
        """Names the argument, the error and every shape the argument may take."""
        reason = pathguard.schema_problem("select_series", D7_INPUT, SELECT_SERIES)
        assert reason is not None
        assert reason.startswith("select_series arguments are invalid: criteria.sequence[0]: ")
        assert "is not of type 'string'" in reason
        assert "Expected: criteria.sequence takes string or array of string." in reason
        assert "call select_series again" in reason

    def test_a_valid_call_has_no_problem(self) -> None:
        """Both shapes of the anyOf are accepted."""
        for sequence in ("T2", ["T2", "STIR"]):
            args = {"index_path": "idx.csv", "criteria": {"sequence": sequence}}
            assert pathguard.schema_problem("select_series", args, SELECT_SERIES) is None

    def test_an_enum_lists_its_values(self) -> None:
        """A value outside an enum is shown the values it may take."""
        args = {"index_path": "idx.csv", "criteria": {"contrast": "maybe"}}
        reason = pathguard.schema_problem("select_series", args, SELECT_SERIES)
        assert reason is not None
        assert 'criteria.contrast takes one of "yes", "no"' in reason

    def test_a_missing_parameter_names_the_required_ones(self) -> None:
        """A required parameter left out is reported at the root."""
        reason = pathguard.schema_problem("select_series", {"criteria": {}}, SELECT_SERIES)
        assert reason is not None
        assert "(arguments): 'index_path' is a required property" in reason
        assert "requires 'index_path', 'criteria'" in reason

    def test_hint_follows_refs_and_stays_short(self) -> None:
        """A $ref is rendered as what it points at; deep nesting is cut off."""
        assert (
            pathguard.schema_hint({"$ref": "#/$defs/SeriesCriteria"}, SELECT_SERIES)
            == "object with sequence, contrast"
        )
        deep: JsonDict = {"type": "array", "items": {"type": "array", "items": {"type": "array"}}}
        nested: JsonDict = {"type": "array", "items": deep}
        assert pathguard.schema_hint(nested, {}).endswith("…")


class TestDecide:
    """Where the schema check sits in the guard's decision."""

    def test_the_d7_call_is_denied_not_let_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The call that killed the turn now comes back to the model as a denial."""
        asked = _known(monkeypatch, SELECT_SERIES)
        out = pathguard.decide(_invocation(D7_INPUT))
        assert out["decision"] == "deny"
        assert out["reason"].startswith("select_series arguments are invalid")
        assert asked == [ROUTE]

    def test_a_valid_call_goes_on_to_the_path_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid arguments are allowed as before (no paths to object to here)."""
        _known(monkeypatch, SELECT_SERIES)
        args = {"index_path": "idx.csv", "criteria": {"sequence": "T2"}}
        assert pathguard.decide(_invocation(args)) == {"decision": "allow"}

    def test_unknown_tool_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No schema known for the tool: allowed exactly as before."""
        _known(monkeypatch, None)
        assert pathguard.decide(_invocation(D7_INPUT)) == {"decision": "allow"}

    def test_no_core_to_ask_is_allowed(self) -> None:
        """No broker socket: the lookup finds nothing and the call goes on."""
        assert pathguard.decide(_invocation(D7_INPUT)) == {"decision": "allow"}

    def test_a_broken_lookup_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Any error while checking lets the call through (strict = false semantics)."""

        def boom(route: str) -> JsonDict | None:
            raise RuntimeError("broker exploded")

        monkeypatch.setattr(pathguard, "_known_schema", boom)
        assert pathguard.decide(_invocation(D7_INPUT)) == {"decision": "allow"}

    def test_an_unusable_schema_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A schema jsonschema cannot use is no reason to refuse the call."""
        _known(monkeypatch, {"type": "object", "properties": {"x": {"pattern": "("}}})
        assert pathguard.decide(_invocation({"x": "a"})) == {"decision": "allow"}

    def test_the_denial_is_not_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unlike a path denial, giving up here would end the turn; the ledger stays out of it."""
        _known(monkeypatch, SELECT_SERIES)
        for _ in range(pathguard._MAX_DENIALS + 2):
            assert pathguard.decide(_invocation(D7_INPUT))["decision"] == "deny"
        assert not pathguard._ledger_path().exists()

    def test_vibe_tools_are_never_looked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The harness checks its own tools; the core is not asked about them."""

        async def no_core(op: str, payload: JsonDict, *, timeout: float = 5.0) -> JsonDict | None:
            raise AssertionError(f"asked the core about {payload}")

        monkeypatch.setattr(plan, "ask_core", no_core)
        for route in ("file_system.bash", "vibe.todo", "run_typescript"):
            assert pathguard.check_schema(route, {"command": 3}) is None

    def test_main_prints_the_denial(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The hook's stdout carries the deny decision vibe maps to a skipped call."""
        _known(monkeypatch, SELECT_SERIES)
        monkeypatch.setattr("sys.stdin.read", lambda: json.dumps(_invocation(D7_INPUT)))
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


_POOL = _KnownPool(
    {
        "medmcp-cohort": {"select_series": SELECT_SERIES},
        "medmcp": {"other": {"type": "object"}},
    }
)


class TestToolSchemaOp:
    """The broker op the guard asks: answered from the pool's known tool lists."""

    @pytest.fixture(autouse=True)
    def pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The core's pool, with the cohort stack's list known."""
        monkeypatch.setattr(server, "_pool", _POOL)

    def test_by_route(self) -> None:
        """The harness route finds the stack by its group spelling."""
        assert server._tool_schema_op({"tool": ROUTE}) == {"schema": SELECT_SERIES}

    def test_by_medmcp_name(self) -> None:
        """``<stack>_<tool>`` binds to the longest stack name, not a prefix of it."""
        assert server._tool_schema_op({"tool": "medmcp-cohort_select_series"}) == {
            "schema": SELECT_SERIES
        }

    def test_unknown_tool_or_stack(self) -> None:
        """Nothing known: no schema, and nothing is started to find one."""
        assert server._tool_schema_op({"tool": "mcp_medmcp_cohort.nope"}) == {"schema": None}
        assert server._tool_schema_op({"tool": "mcp_medmcp_neuro.skull_strip"}) == {"schema": None}
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
    broker = BackendBroker(
        pool, vibe_home / "backend.sock", ops={"tool_schema": server._tool_schema_op}
    )
    await broker.start()
    try:
        # The hook runs synchronously (asyncio.run), as in its own process.
        denial = await asyncio.to_thread(pathguard.check_schema, ROUTE, D7_INPUT)
        assert denial is not None
        assert denial["decision"] == "deny"
        valid = {"index_path": "idx.csv", "criteria": {"sequence": ["T2"]}}
        assert await asyncio.to_thread(pathguard.check_schema, ROUTE, valid) is None
    finally:
        await broker.aclose()
        await pool.aclose()
