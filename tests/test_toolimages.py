"""Stack-result images reach a local model that can see them (``toolimages`` and shim repair 8).

vibe 2.26.1 drops the image blocks of an MCP result. The broker hook stores them and
leaves markers; the LLM shim attaches them to the request for a vision model.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

import mcp.types as mcp_types
import pytest
from starlette.responses import JSONResponse

from medmcp import llmshim, server, toolimages
from medmcp.toolimages import ImageStore

JsonDict = dict[str, Any]

PNG_A = b"\x89PNG\r\n\x1a\nfirst render"
PNG_B = b"\x89PNG\r\n\x1a\nsecond render"


def _image(data: bytes, mime: str = "image/png") -> mcp_types.ImageContent:
    return mcp_types.ImageContent(
        type="image", data=base64.b64encode(data).decode("ascii"), mimeType=mime
    )


def _result(*blocks: mcp_types.ContentBlock) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(content=list(blocks))


def _texts(result: mcp_types.CallToolResult) -> list[str]:
    return [b.text for b in result.content if isinstance(b, mcp_types.TextContent)]


def _ids(text: str) -> list[str]:
    return toolimages.MARKER_RE.findall(text)


@pytest.fixture
def store(tmp_path: Path) -> ImageStore:
    """An empty store in a fresh directory."""
    return ImageStore(tmp_path / "images")


class TestStore:
    """Content-addressed, bounded, and safe to look up by any string."""

    def test_round_trip_and_same_bytes_same_id(self, store: ImageStore) -> None:
        """An image comes back with its type; storing it again gives the same id."""
        image_id = store.put(PNG_A, "image/png")
        assert store.get(image_id) == (PNG_A, "image/png")
        assert store.put(PNG_A, "image/png") == image_id
        assert len(list(store.root.iterdir())) == 1

    def test_unknown_and_malformed_ids_are_absent(self, store: ImageStore) -> None:
        """Only a stored hex id resolves; nothing else touches the file system."""
        store.put(PNG_A, "image/png")
        for image_id in ("0" * 32, "../../etc/passwd", "", "A" * 32):
            assert store.get(image_id) is None
            assert not store.has(image_id)

    def test_oldest_images_go_when_the_store_is_full(self, tmp_path: Path) -> None:
        """Past its bound the store drops the least recently used, never the new one."""
        small = ImageStore(tmp_path / "images", max_bytes=2 * len(PNG_A) + 1)
        first = small.put(PNG_A, "image/png")
        second = small.put(PNG_B, "image/png")
        small.get(first)  # used again: now the more recent of the two
        third = small.put(b"\x89PNG\r\n\x1a\nthird render", "image/png")
        assert small.has(first) and small.has(third)
        assert not small.has(second)


class TestExtract:
    """The broker side: image blocks become markers, images go to the store."""

    def test_images_become_markers_in_place(self, store: ImageStore) -> None:
        """Text keeps its place; each image is replaced by a marker naming it."""
        result = _result(
            mcp_types.TextContent(type="text", text="central slices"),
            _image(PNG_A),
            _image(PNG_B, "image/jpeg"),
        )
        out = toolimages.extract(result, store)
        texts = _texts(out)
        assert texts[0] == "central slices"
        assert not any(isinstance(b, mcp_types.ImageContent) for b in out.content)
        first, second = (_ids(t)[0] for t in texts[1:])
        assert texts[1] == toolimages.marker(first)
        assert texts[1].endswith(f"{toolimages.RETURNED}]")
        assert store.get(first) == (PNG_A, "image/png")
        assert store.get(second) == (PNG_B, "image/jpeg")

    def test_a_result_without_images_is_untouched(self, store: ImageStore) -> None:
        """No copy, no store directory."""
        result = _result(mcp_types.TextContent(type="text", text="ok"))
        assert toolimages.extract(result, store) is result
        assert not store.root.exists()

    def test_error_flag_and_structured_content_survive(self, store: ImageStore) -> None:
        """Only the image blocks change."""
        result = mcp_types.CallToolResult(
            content=[_image(PNG_A)], structuredContent={"k": 1}, isError=True
        )
        out = toolimages.extract(result, store)
        assert out.isError is True
        assert out.structuredContent == {"k": 1}

    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            (_image(PNG_A, "image/svg+xml"), "of type image/svg+xml"),
            (
                mcp_types.ImageContent(type="image", data="not base64!", mimeType="image/png"),
                "unreadable",
            ),
        ],
    )
    def test_images_that_cannot_be_sent_are_described(
        self, store: ImageStore, block: mcp_types.ImageContent, expected: str
    ) -> None:
        """An unsupported or broken image is named, not stored, and has no id."""
        (text,) = _texts(toolimages.extract(_result(block), store))
        assert expected in text and toolimages.NOT_SHOWN in text
        assert not _ids(text)
        assert not store.root.exists()

    def test_an_oversized_image_is_described(
        self, store: ImageStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Past the size bound the image is named with its size and not kept."""
        monkeypatch.setattr(toolimages, "MAX_IMAGE_BYTES", 4)
        (text,) = _texts(toolimages.extract(_result(_image(PNG_A)), store))
        assert "too large" in text and not _ids(text)


def _chat(*tool_texts: str, store: ImageStore | None = None) -> JsonDict:
    """A chat: one user message, then per text an assistant call and its tool result."""
    messages: list[JsonDict] = [{"role": "user", "content": "segment the liver"}]
    for n, text in enumerate(tool_texts):
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{n}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{n}", "content": text})
    return {"model": "muse-medmcp", "messages": messages}


def _image_parts(message: JsonDict) -> list[bytes]:
    parts = cast("list[JsonDict]", message["content"])
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    return [base64.b64decode(u.split(",", 1)[1]) for u in urls]


class TestAttach:
    """The shim side: markers in tool results become images in the request."""

    def test_an_image_goes_after_its_tool_result(self, store: ImageStore) -> None:
        """A user-role message from MedMCP carries it; the marker says so."""
        image_id = store.put(PNG_A, "image/png")
        payload = _chat(f"view\n{toolimages.marker(image_id)}")
        assert toolimages.attach_images(payload, store, vision=True, max_images=8) == 1
        messages = payload["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
        assert (
            messages[2]["content"] == f"view\n{toolimages.marker(image_id, 'attached as image 1')}"
        )
        assert messages[3]["content"][0]["text"] == toolimages.ATTACHMENT_NOTE
        assert _image_parts(messages[3]) == [PNG_A]
        assert messages[3]["content"][-1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_only_the_most_recent_images_are_sent_again(self, store: ImageStore) -> None:
        """Older images are named as not attached; numbering follows reading order."""
        a, b = store.put(PNG_A, "image/png"), store.put(PNG_B, "image/png")
        payload = _chat(toolimages.marker(a), toolimages.marker(b))
        assert toolimages.attach_images(payload, store, vision=True, max_images=1) == 1
        messages = payload["messages"]
        assert messages[2]["content"] == toolimages.marker(a, toolimages.EARLIER)
        assert messages[4]["content"] == toolimages.marker(b, "attached as image 1")
        assert [m["role"] for m in messages][-2:] == ["tool", "user"]
        assert _image_parts(messages[-1]) == [PNG_B]

    def test_results_of_parallel_calls_share_one_attachment(self, store: ImageStore) -> None:
        """Tool results that follow one another stay together; the images follow them."""
        a, b = store.put(PNG_A, "image/png"), store.put(PNG_B, "image/png")
        payload: JsonDict = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "check both"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "x"}, {"id": "y"}]},
                {"role": "tool", "tool_call_id": "x", "content": toolimages.marker(a)},
                {"role": "tool", "tool_call_id": "y", "content": toolimages.marker(b)},
            ],
        }
        assert toolimages.attach_images(payload, store, vision=True, max_images=8) == 2
        messages = payload["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "user"]
        assert _image_parts(messages[4]) == [PNG_A, PNG_B]

    def test_a_model_without_vision_gets_text_only(self, store: ImageStore) -> None:
        """Nothing is attached and the marker says why."""
        image_id = store.put(PNG_A, "image/png")
        payload = _chat(toolimages.marker(image_id))
        assert toolimages.attach_images(payload, store, vision=False, max_images=8) == 0
        assert len(payload["messages"]) == 3
        assert payload["messages"][2]["content"] == toolimages.marker(
            image_id, toolimages.NO_VISION
        )

    def test_a_missing_image_is_named_as_gone(self, store: ImageStore) -> None:
        """An image pruned since (or from another core) cannot be sent."""
        payload = _chat(toolimages.marker("ab" * 16))
        assert toolimages.attach_images(payload, store, vision=True, max_images=8) == 0
        assert payload["messages"][2]["content"] == toolimages.marker("ab" * 16, toolimages.GONE)

    def test_markers_outside_tool_results_are_ignored(self, store: ImageStore) -> None:
        """A user or the model writing a marker does not attach anything."""
        image_id = store.put(PNG_A, "image/png")
        payload = _chat()
        payload["messages"][0]["content"] = toolimages.marker(image_id)
        assert toolimages.attach_images(payload, store, vision=True, max_images=8) == 0
        assert payload["messages"][0]["content"] == toolimages.marker(image_id)

    def test_list_content_and_rewritten_markers_are_read(self, store: ImageStore) -> None:
        """Text parts are searched, and a marker rewritten before still resolves."""
        image_id = store.put(PNG_A, "image/png")
        payload = _chat()
        payload["messages"] += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": [{"type": "text", "text": toolimages.marker(image_id, "old status")}],
            },
        ]
        assert toolimages.attach_images(payload, store, vision=True, max_images=8) == 1
        assert payload["messages"][2]["content"][0]["text"] == toolimages.marker(
            image_id, "attached as image 1"
        )

    def test_zero_images_attaches_none_and_says_so(self, store: ImageStore) -> None:
        """Switched off, the model still learns that it was not shown the image."""
        image_id = store.put(PNG_A, "image/png")
        payload = _chat(toolimages.marker(image_id))
        assert toolimages.attach_images(payload, store, vision=True, max_images=0) == 0
        assert len(payload["messages"]) == 3
        assert payload["messages"][2]["content"] == toolimages.marker(
            image_id, toolimages.NOT_SHOWN
        )


async def test_the_broker_hook_takes_the_images_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_after_stack_call``: what the proxy relays to vibe has markers, not images."""
    store = ImageStore(tmp_path / "images")
    monkeypatch.setattr(server, "_tool_images", store)
    monkeypatch.setattr(server, "_jobs", None)
    out = await server._after_stack_call("medmcp-x", "view", {}, object(), _result(_image(PNG_A)))
    (text,) = _texts(out)
    assert store.get(_ids(text)[0]) == (PNG_A, "image/png")


def _forwarding(monkeypatch: pytest.MonkeyPatch) -> list[JsonDict]:
    forwarded: list[JsonDict] = []

    async def complete(
        url: str, payload: JsonDict, *args: object, **kwargs: object
    ) -> JSONResponse:
        forwarded.append(payload)
        return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llmshim, "_complete_with_retry", complete)
    return forwarded


@pytest.mark.parametrize("vision", [True, False])
def test_the_shim_forwards_the_images_to_a_vision_model(
    store: ImageStore, monkeypatch: pytest.MonkeyPatch, vision: bool
) -> None:
    """Through the endpoint: the request upstream carries the image only for a vision model."""
    from starlette.testclient import TestClient

    forwarded = _forwarding(monkeypatch)
    asked: list[str] = []

    async def sees_images(model: str) -> bool:
        asked.append(model)
        return vision

    image_id = store.put(PNG_A, "image/png")
    app = llmshim.create_app(
        "http://upstream.invalid", tool_mode="ollama", image_store=store, sees_images=sees_images
    )
    resp = TestClient(app).post(
        "/v1/chat/completions", json=_chat(f"view\n{toolimages.marker(image_id)}")
    )
    assert resp.status_code == 200
    assert asked == ["muse-medmcp"]
    messages = forwarded[0]["messages"]
    if vision:
        assert _image_parts(messages[-1]) == [PNG_A]
    else:
        assert messages[-1]["role"] == "tool"
        assert toolimages.NO_VISION in messages[-1]["content"]


def test_switched_off_the_shim_does_not_ask_about_vision(
    store: ImageStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_images=0``: no capability lookup; the marker says the image is not shown."""
    from starlette.testclient import TestClient

    forwarded = _forwarding(monkeypatch)

    async def sees_images(model: str) -> bool:
        raise AssertionError("asked while switched off")

    image_id = store.put(PNG_A, "image/png")
    app = llmshim.create_app(
        "http://upstream.invalid",
        tool_mode="ollama",
        image_store=store,
        sees_images=sees_images,
        max_images=0,
    )
    resp = TestClient(app).post("/v1/chat/completions", json=_chat(toolimages.marker(image_id)))
    assert resp.status_code == 200
    assert forwarded[0]["messages"][-1]["content"] == toolimages.marker(
        image_id, toolimages.NOT_SHOWN
    )


def test_the_shim_does_not_ask_about_a_chat_without_images(
    store: ImageStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No marker, no capability lookup and no change."""
    from starlette.testclient import TestClient

    forwarded = _forwarding(monkeypatch)

    async def sees_images(model: str) -> bool:
        raise AssertionError("asked without a marker")

    app = llmshim.create_app(
        "http://upstream.invalid", tool_mode="ollama", image_store=store, sees_images=sees_images
    )
    payload = _chat("plain text")
    assert TestClient(app).post("/v1/chat/completions", json=payload).status_code == 200
    assert forwarded[0]["messages"] == payload["messages"]


def test_a_failing_lookup_never_fails_the_request(
    store: ImageStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Images are a bonus: a broken capability check sends the turn without them."""
    from starlette.testclient import TestClient

    forwarded = _forwarding(monkeypatch)

    async def sees_images(model: str) -> bool:
        raise RuntimeError("upstream gone")

    image_id = store.put(PNG_A, "image/png")
    app = llmshim.create_app(
        "http://upstream.invalid", tool_mode="ollama", image_store=store, sees_images=sees_images
    )
    resp = TestClient(app).post("/v1/chat/completions", json=_chat(toolimages.marker(image_id)))
    assert resp.status_code == 200
    assert forwarded[0]["messages"][-1]["role"] == "tool"


async def test_ollama_capabilities_are_asked_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/api/show`` decides; a vision answer is kept, a failure is not."""
    import httpx

    calls: list[str] = []
    answers = iter([httpx.Response(500), httpx.Response(200, json={"capabilities": ["vision"]})])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return next(answers)

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def client(*args: object, **kwargs: Any) -> httpx.AsyncClient:  # noqa: ANN401
        return real_client(transport=transport)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    sees = llmshim.ollama_capabilities("http://ollama.invalid")
    assert await sees("m") is False  # 500: not known, not kept
    assert await sees("m") is True
    assert await sees("m") is True
    assert calls == ["m", "m"]
