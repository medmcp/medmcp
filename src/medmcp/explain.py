"""LLM-generated plain-language explanations and risk tags for tool calls.

When the user enables "Explain tool calls", each permission prompt is preceded
by a short explanation of what the tool call does, produced by the same local
Ollama model that powers the agent, via a lightweight direct API call.

UI-agnostic: callers invoke :func:`generate_explanation` and render the
result their own way.
"""

from __future__ import annotations

import json
import logging
import re
from typing import cast

import httpx

from medmcp.acp import JsonDict
from medmcp.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

_audit: logging.Logger = logging.getLogger("medmcp.audit")

# Timeout for the explanation call.  Local Ollama inference is fast but the
# model may be cold-starting; 20 s is generous without blocking the UI too long.
EXPLAIN_TIMEOUT: float = 20.0

# ── Risk taxonomy ──────────────────────────────────────────
# Predefined categories for tool-call risk assessment.  The LLM is instructed
# to pick from these keys only — it never invents new ones.
# Each value is (display_label, severity).  severity drives the icon shown in
# the permission dialog and the tool-call summary.
RISK_CATEGORIES: dict[str, tuple[str, str]] = {
    "file_read": ("Reads existing files", "low"),
    "file_write": ("Creates or modifies files", "medium"),
    "file_delete": ("Deletes files — may be irreversible", "high"),
    "network": ("Contacts an external server or website", "medium"),
    "code_exec": ("Runs a program or shell command", "high"),
    "data_exfil": ("Could send your data to an external service", "high"),
    "system_config": ("Changes system or application settings", "medium"),
    "privacy": ("Accesses personal or sensitive information", "high"),
    "skill_load": ("Loads external instructions into the agent context", "medium"),
}

SEVERITY_ICON: dict[str, str] = {"low": "🟢", "medium": "🟡", "high": "🔴"}


def resolve_risks(risks: list[str]) -> list[JsonDict]:
    """Expand risk keys into ``{key, label, severity}`` dicts for structured UIs."""
    resolved: list[JsonDict] = []
    for key in risks:
        entry = RISK_CATEGORIES.get(key)
        if entry is not None:
            resolved.append({"key": key, "label": entry[0], "severity": entry[1]})
    return resolved


def raw_input_to_str(raw_input_val: object) -> str:
    """Stringify a rawInput value for inclusion in the explanation prompt."""
    if isinstance(raw_input_val, dict):
        try:
            # pyright: ignore — json.dumps accepts Any-typed dict values at runtime
            return json.dumps(raw_input_val, indent=2)  # pyright: ignore[reportUnknownArgumentType]
        except (TypeError, ValueError):
            return str(raw_input_val)  # pyright: ignore[reportUnknownArgumentType]
    return str(raw_input_val)


async def generate_explanation(tc: JsonDict) -> tuple[str, list[str]] | None:
    """Ask the local Ollama model to explain a tool call for a non-technical user.

    Returns ``(explanation, risks)`` on success or ``None`` on failure/timeout.
    ``explanation`` is a single plain-language sentence aimed at a physician with
    no IT background.  ``risks`` is a (possibly empty) list of keys from
    :data:`RISK_CATEGORIES` that the model identified as applicable.

    Errors are logged but never propagated — the permission dialog renders
    without an explanation rather than blocking the user.
    """
    title = tc.get("title") or ""
    raw_input_str = raw_input_to_str(tc.get("rawInput") or "")

    # Keep the input snippet short so the prompt stays well within the model's
    # context window.  The full raw input is already shown in the JSON fence
    # inside the permission dialog, so truncating here is fine.
    if len(raw_input_str) > 400:
        raw_input_str = raw_input_str[:400] + "\n… (truncated)"

    valid_keys = ", ".join(RISK_CATEGORIES)
    prompt = (
        "You are a security-aware assistant helping a physician review an AI action "
        "before it runs on their computer. Your job is to explain what the action does "
        "and flag any risks — in plain language that requires no IT knowledge.\n\n"
        "Guidelines for the explanation:\n"
        "- Write ONE clear sentence a doctor with no computer background can understand.\n"
        "- Avoid all technical jargon. Translate terms like 'bash', 'stdin', 'API', "
        "'filesystem path', 'subprocess', or 'flag' into everyday language "
        "(e.g. 'runs a program', 'opens a file', 'contacts a website').\n"
        "- State what the action will DO and what will CHANGE as a result.\n\n"
        "Then select every applicable risk from this fixed list (use the exact keys):\n"
        f"{valid_keys}\n\n"
        "Respond with ONLY a JSON object — no markdown fences, no extra text:\n"
        '{"explanation": "<one sentence>", "risks": ["<key>", ...]}\n\n'
        f"Tool: {title}\n"
        f"Input: {raw_input_str}"
    )

    try:
        async with httpx.AsyncClient(timeout=EXPLAIN_TIMEOUT) as client:
            # Use the native Ollama /api/chat endpoint: the OpenAI-compatible
            # endpoint ignores think:false, causing thinking models to emit output
            # into "reasoning" only and leave "content" empty.
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.2, "num_predict": 1024},
                },
            )
            resp.raise_for_status()
            data = cast("JsonDict", resp.json())
            message = cast("JsonDict", data.get("message") or {})
            raw_text = str(message.get("content") or "").strip()
    except Exception:
        _audit.warning("failed to generate tool-call explanation", exc_info=True)
        return None

    return parse_explanation_response(raw_text)


def parse_explanation_response(raw_text: str) -> tuple[str, list[str]] | None:
    """Parse the LLM's JSON response into ``(explanation, risks)``.

    Handles models that wrap JSON in markdown code fences.  Returns ``None`` if
    no valid JSON object can be extracted.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Greedy match is intentional: .*? would stop at the first } and break
        # on nested objects like {"risks": ["file_read"]}.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    try:
        payload_raw: object = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        _audit.warning("could not parse explanation JSON: %r", raw_text[:200])
        return None

    if not isinstance(payload_raw, dict):
        return None
    payload = cast("JsonDict", payload_raw)

    explanation = str(payload.get("explanation") or "").strip()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    if not explanation:
        return None

    raw_risks = payload.get("risks")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    risks: list[str] = (
        [k for k in cast("list[object]", raw_risks) if isinstance(k, str) and k in RISK_CATEGORIES]
        if isinstance(raw_risks, list)
        else []
    )

    return explanation, risks
