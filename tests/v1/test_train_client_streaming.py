import json
from unittest.mock import AsyncMock

import aiohttp
import pytest
from openai.types.chat import ChatCompletionChunk

from verifiers.v1.clients import ModelContext, TrainClient, TrainClientConfig
from verifiers.v1.clients.train import serialize_completion_stream
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.dialects.base import is_sse_done_event, parse_sse_event
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    SamplingConfig,
    ToolCall,
    TurnTokens,
    Usage,
)


def _response(*, tool_call: bool = False) -> Response:
    return Response(
        id="completion-1",
        created=123,
        model="test-model",
        message=AssistantMessage(
            content=None if tool_call else "READY",
            tool_calls=(
                [
                    ToolCall(
                        id="call-1",
                        name="ipython",
                        arguments=json.dumps({"code": "6 * 7"}),
                    )
                ]
                if tool_call
                else None
            ),
        ),
        finish_reason="tool_calls" if tool_call else "stop",
        usage=Usage(prompt_tokens=2, completion_tokens=2),
        tokens=TurnTokens(
            prompt_ids=[10, 11],
            completion_ids=[12, 13],
            completion_logprobs=[-0.1, -0.2],
        ),
    )


def test_completion_stream_is_valid_and_round_trips_tool_calls() -> None:
    response = _response(tool_call=True)
    chunks = serialize_completion_stream(response, "fallback-model")

    payloads = [
        parse_sse_event(chunk) for chunk in chunks if not is_sse_done_event(chunk)
    ]
    native = [ChatCompletionChunk.model_validate(payload) for payload in payloads]
    assert native[0].choices[0].delta.tool_calls[0].index == 0
    assert native[1].choices[0].finish_reason == "tool_calls"

    parser = ChatDialect().stream_parser()
    for chunk in chunks:
        parser.feed(chunk)
    parsed = parser.finish()

    assert parsed.message == response.message
    assert parsed.finish_reason == response.finish_reason
    assert parsed.usage == response.usage


@pytest.mark.asyncio
async def test_interception_commits_renderer_tokens_for_streaming_request() -> None:
    response = _response()
    client = object.__new__(TrainClient)
    client.get_response = AsyncMock(return_value=response)
    trace = Trace.model_construct(
        id="trace-1", nodes=[], calls=[], tools=[], stop_condition=None
    )
    context = ModelContext(
        model="test-model",
        client=TrainClientConfig(
            base_url="http://127.0.0.1:8000",
            renderer_model_name="test-model",
        ),
        sampling=SamplingConfig(max_tokens=16),
    )
    session = RolloutSession(ctx=context, client=client, trace=trace)

    async with (
        InterceptionServer() as server,
        server.acquire(session) as (base_url, secret, _),
        aiohttp.ClientSession() as http,
    ):
        result = await http.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {secret}"},
            json={
                "model": "ignored",
                "messages": [{"role": "user", "content": "Return READY"}],
                "stream": True,
            },
        )
        body = await result.read()

    assert result.status == 200
    assert body.endswith(b"data: [DONE]\n\n")
    assert trace.num_turns == 1
    assert trace.nodes[-1].token_ids[-2:] == [12, 13]
    assert trace.nodes[-1].logprobs == [-0.1, -0.2]
    assert trace.calls[0].node == len(trace.nodes) - 1
    assert client.get_response.await_args.kwargs["turn"].trace is trace
