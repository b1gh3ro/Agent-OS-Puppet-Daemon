"""OpenRouter speaks OpenAI chat; the brain speaks google-genai Content/Part.

These cover the translation between them, and the seam that translation cannot
paper over: Gemini's computer-use tool is a server-side built-in with no
OpenRouter equivalent, so the UI vocabulary is declared by hand and must stay in
step with what GeminiBrain._execute can actually dispatch.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestServer
from google.genai import types

from agentos import openrouter
from agentos.brain import GeminiBrain, OpenRouterBrain
from agentos.models import Task

PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


def _blob():
    return types.Blob(data=PNG, mime_type="image/png")


def _conversation() -> list[types.Content]:
    """One full round: goal+screenshot, a batched model turn, its answers."""
    return [
        types.Content(role="user", parts=[
            types.Part(text="operate this desktop"),
            types.Part(inline_data=_blob()),
        ]),
        types.Content(role="model", parts=[
            types.Part(text="I will click then type."),
            types.Part(function_call=types.FunctionCall(name="click_at", args={"x": 500, "y": 400})),
            types.Part(function_call=types.FunctionCall(name="type_text", args={"text": "hi"})),
        ]),
        types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(name="click_at", response={"ok": True})),
            types.Part(function_response=types.FunctionResponse(
                name="type_text", response={"ok": True},
                parts=[types.FunctionResponsePart(
                    inline_data=types.FunctionResponseBlob(mime_type="image/png", data=PNG))])),
            types.Part(text="[budget: 12 of 15 actions remaining]"),
        ]),
    ]


def test_messages_pair_every_tool_call_with_its_result():
    messages = openrouter._to_messages(_conversation(), "be careful")

    assert messages[0] == {"role": "system", "content": "be careful"}
    assistant = next(m for m in messages if m["role"] == "assistant")
    ids = [c["id"] for c in assistant["tool_calls"]]
    assert len(ids) == len(set(ids)) == 2
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["click_at", "type_text"]
    # Arguments cross the wire as a JSON *string* on this API, not an object.
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"x": 500, "y": 400}

    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ids
    # ...and they must directly follow the assistant turn that called them.
    start = messages.index(assistant)
    assert messages[start + 1: start + 3] == tool_messages


def test_screenshot_inside_a_tool_result_becomes_its_own_user_turn():
    """A `tool` message is text-only, so the frame Gemini nests in the function
    response has to be re-emitted as a user turn or the model goes blind."""
    messages = openrouter._to_messages(_conversation(), None)

    last = messages[-1]
    assert last["role"] == "user"
    images = [c for c in last["content"] if c["type"] == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    # The budget note still rides along, after the frame it describes.
    assert any("budget" in c.get("text", "") for c in last["content"])
    assert all("image_url" not in json.dumps(m) for m in messages if m["role"] == "tool")


def test_opening_turn_keeps_text_before_its_screenshot():
    messages = openrouter._to_messages(_conversation()[:1], None)
    kinds = [c["type"] for c in messages[0]["content"]]
    assert kinds == ["text", "image_url"]


def test_orphaned_result_is_folded_into_text_not_sent_as_a_tool_message():
    """A response with no surviving call would be a 400. History trimming moves
    in call/response blocks so it should not happen — but a hand-built or
    hand-edited history must degrade into context, not kill the run."""
    contents = [types.Content(role="user", parts=[
        types.Part(function_response=types.FunctionResponse(
            name="click_at", response={"ok": True}))])]

    messages = openrouter._to_messages(contents, None)

    assert not [m for m in messages if m["role"] == "tool"]
    assert "click_at" in messages[0]["content"][0]["text"]


def test_reply_becomes_a_model_turn_the_loop_can_append():
    content = openrouter._to_content({
        "content": "clicking now",
        "tool_calls": [{"id": "call_0", "type": "function", "function": {
            "name": "click_at", "arguments": '{"x": 10, "y": 20}'}}],
    })

    assert content.role == "model"
    assert content.parts[0].text == "clicking now"
    call = content.parts[1].function_call
    assert (call.name, call.args) == ("click_at", {"x": 10, "y": 20})


def test_unparsable_arguments_survive_as_a_dispatchable_call():
    """Truncated JSON must not raise inside the loop; _execute will reject the
    action and the error goes back to the model as a normal function response."""
    content = openrouter._to_content({"tool_calls": [{"function": {
        "name": "click_at", "arguments": '{"x": 10, "y":'}}]})
    assert content.parts[0].function_call.args == {"_unparsed_arguments": '{"x": 10, "y":'}


def test_empty_reply_yields_no_candidate():
    """None routes into the brain's empty-response recovery. Returning an empty
    Content instead would read as 'no calls' — i.e. a finished task."""
    assert openrouter._to_content({"content": "", "tool_calls": []}) is None


def test_http_errors_are_classified_by_the_brains_retry_predicates():
    throttled = openrouter.OpenRouterError("OpenRouter 429: rate limit exceeded")
    assert GeminiBrain._is_rate_limit(throttled)
    assert GeminiBrain._is_transient(openrouter.OpenRouterError("OpenRouter 503: upstream"))
    assert not GeminiBrain._is_rate_limit(openrouter.OpenRouterError("OpenRouter 400: bad request"))
    # A Retry-After is rewritten into the spelling _retry_after already parses.
    assert GeminiBrain._retry_after(
        openrouter.OpenRouterError("OpenRouter 429: retryDelay: 30.0s body"), fallback=99) == 30.0


def test_usage_maps_onto_the_fields_the_run_log_records():
    fields = GeminiBrain._usage_fields(openrouter._Response([], {
        "prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
        "prompt_tokens_details": {"cached_tokens": 400},
        "completion_tokens_details": {"reasoning_tokens": 20},
    }, None))

    assert fields["prompt_tokens"] == 1000
    assert fields["cached_tokens"] == 400
    assert fields["uncached_prompt_tokens"] == 600
    assert fields["output_tokens"] == 50
    assert fields["thoughts_tokens"] == 20


class _RecordingSandbox:
    """Accepts every Sandbox call and records the name."""

    width, height = 1280, 800

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        async def record(*args, **kwargs):
            self.calls.append(name)
            if name == "exec_shell":
                return 0, ""
            if name == "screenshot":
                return PNG
        return record


def _declared_tools() -> list[dict]:
    brain = OpenRouterBrain.__new__(OpenRouterBrain)  # __init__ wants an API key
    brain._tools = []
    return brain._config().tools


def test_every_declared_ui_tool_is_one_execute_can_dispatch(monkeypatch):
    """The built-in computer-use tool guaranteed the action names matched what
    the model emits. Declaring them by hand removes that guarantee, so pin it:
    a rename in _execute that misses openrouter.UI_TOOLS is a run that dies on
    'unsupported action'."""
    brain = OpenRouterBrain.__new__(OpenRouterBrain)
    sandbox, task = _RecordingSandbox(), Task(goal="x")

    async def instant(_seconds):  # the launch/settle waits are real seconds
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)

    async def dispatch_all():
        for tool in openrouter.UI_TOOLS:
            args = {"x": 1, "y": 2, "destination_x": 3, "destination_y": 4,
                    "text": "t", "keys": "ctrl+a", "url": "http://x", "seconds": 0}
            await brain._execute(tool["name"], args, sandbox, task)

    asyncio.run(dispatch_all())


def test_config_exposes_both_the_ui_and_the_custom_toolset():
    brain = OpenRouterBrain.__new__(OpenRouterBrain)
    brain._tools = [{"name": "run_command", "description": "d", "parameters": {}}]
    names = [t["name"] for t in brain._config("house rules").tools]

    assert "click_at" in names and "run_command" in names
    assert brain._config("house rules").system_instruction.endswith("house rules")


def test_declared_tools_are_valid_json_schema_function_declarations():
    for tool in _declared_tools():
        assert set(tool) >= {"name", "description", "parameters"}
        assert tool["parameters"]["type"] == "object"
        for prop in tool["parameters"]["properties"].values():
            assert "type" in prop and "description" in prop


def test_coordinates_stay_on_the_grid_denormalize_expects():
    """The tools advertise a 0-1000 grid because scaling.denormalize divides by
    1000. Advertising pixels here would put every click at a fraction of its
    intended position."""
    for tool in openrouter.UI_TOOLS:
        x = tool["parameters"]["properties"].get("x")
        if x:
            assert "0-1000" in x["description"]


def test_request_and_response_round_trip_over_real_http():
    """Exercise the HTTP layer end to end against a local stand-in: the body we
    build, the headers we send, and the reply we parse."""
    from aiohttp import web

    seen: dict = {}

    async def handler(request: web.Request) -> web.Response:
        seen["body"] = await request.json()
        seen["auth"] = request.headers.get("Authorization")
        return web.json_response({
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "click_at", "arguments": '{"x": 500, "y": 500}'}}]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        })

    async def exercise():
        app = web.Application()
        app.router.add_post("/chat/completions", handler)
        server = TestServer(app)
        await server.start_server()
        client = openrouter.OpenRouterClient(
            "test-key", url=str(server.make_url("/chat/completions")))
        try:
            return await client.generate_content(
                "google/gemini-3.7-flash", _conversation(),
                openrouter.Config(system_instruction="rules",
                                  tools=list(openrouter.UI_TOOLS)))
        finally:
            await client.aclose()
            await server.close()

    response = asyncio.run(exercise())

    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "google/gemini-3.7-flash"
    assert seen["body"]["tools"][0]["type"] == "function"
    assert seen["body"]["tools"][0]["function"]["name"] == "click_at"
    assert response.candidates[0].content.parts[0].function_call.name == "click_at"
    assert response.usage_metadata.prompt_token_count == 12


def test_api_error_inside_a_200_body_still_raises():
    """OpenRouter can report a provider failure with HTTP 200 and an `error`
    object. Returning that as a normal reply would look like an empty turn and
    silently burn the step budget instead of retrying."""
    from aiohttp import web

    async def handler(request):
        return web.json_response({"error": {"code": 429, "message": "rate limited"}})

    async def exercise():
        app = web.Application()
        app.router.add_post("/x", handler)
        server = TestServer(app)
        await server.start_server()
        client = openrouter.OpenRouterClient("k", url=str(server.make_url("/x")))
        try:
            with pytest.raises(openrouter.OpenRouterError) as excinfo:
                await client.generate_content("m", [], openrouter.Config())
            return excinfo.value
        finally:
            await client.aclose()
            await server.close()

    assert GeminiBrain._is_rate_limit(asyncio.run(exercise()))
