import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from verifiers.v1.clients.train import TrainClient
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    SamplingConfig,
    ToolCall,
    TurnTokens,
    Usage,
)


def _event(chunk: bytes) -> dict:
    return json.loads(chunk.removeprefix(b"data: ").strip())


@pytest.mark.asyncio
async def test_train_client_synthesizes_stream_and_preserves_response() -> None:
    response = Response(
        id="request-1",
        created=0,
        model="model",
        message=AssistantMessage(
            content="working",
            reasoning_content="checking",
            tool_calls=[ToolCall(id="call-1", name="python", arguments='{"x":1}')],
        ),
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=3, completion_tokens=4),
        tokens=TurnTokens(
            prompt_ids=[1, 2, 3],
            completion_ids=[4, 5, 6, 7],
            completion_logprobs=[-0.1, -0.2, -0.3, -0.4],
        ),
    )
    client = TrainClient.__new__(TrainClient)
    client.get_response = AsyncMock(return_value=response)
    turn = MagicMock(spec=PendingTurn)
    dialect = ChatDialect()
    sampling = SamplingConfig(max_tokens=8)

    reply = await client.relay(
        dialect,
        {"messages": [], "stream": True},
        "model",
        sampling,
        session_id="trace-1",
        turn=turn,
    )
    chunks = [chunk async for chunk in reply.chunks]
    parser = dialect.stream_parser()
    for chunk in chunks:
        parser.feed(chunk)
    parsed = parser.finish()

    assert reply.response is response
    assert reply.response.tokens is response.tokens
    assert reply.content_type == "text/event-stream"
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert _event(chunks[0])["choices"][0] == {
        "index": 0,
        "delta": {
            "role": "assistant",
            "content": "working",
            "reasoning_content": "checking",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"x":1}'},
                }
            ],
        },
        "finish_reason": None,
    }
    assert _event(chunks[1])["choices"][0]["finish_reason"] == "tool_calls"
    assert parsed.message == response.message
    assert parsed.finish_reason == response.finish_reason
    client.get_response.assert_awaited_once_with(
        dialect,
        {"messages": [], "stream": True},
        "model",
        sampling,
        session_id="trace-1",
        turn=turn,
        headers=None,
    )
