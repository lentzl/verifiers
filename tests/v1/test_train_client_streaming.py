import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from renderers import MultiModalData, PlaceholderRange, RenderedTokens

import verifiers.v1 as vf
from verifiers.v1.clients.train import (
    TrainClient,
    _prepend_system_tokens,
    _task_system_prefix,
)
from verifiers.v1.configs.client import TrainClientConfig
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


def test_task_system_prefix_uses_task_data_and_template() -> None:
    data = vf.WireTaskData(prompt="question", demonstration="expert actions")
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=data),
    )
    turn = PendingTurn(trace=trace, prompt=[], prefix_node_ids=[], path_len=0)
    config = TrainClientConfig(
        base_url="http://localhost:8000/v1",
        task_system_prefix_field="demonstration",
        task_system_prefix_template="<demo>\n{value}\n</demo>",
    )

    assert _task_system_prefix(config, turn) == "<demo>\nexpert actions\n</demo>"


def test_prepend_system_tokens_preserves_prompt_attribution_and_shifts_media() -> None:
    prefix = RenderedTokens(token_ids=[1, 2, 3])
    prompt = RenderedTokens(
        token_ids=[4, 5],
        message_indices=[0, -1],
        sampled_mask=[False, False],
        is_content=[True, False],
        message_roles=["user"],
        message_tool_names=[None],
        multi_modal_data=MultiModalData(
            mm_hashes={"image": ["hash"]},
            mm_items={"image": [{"pixels": "data"}]},
            mm_placeholders={"image": [PlaceholderRange(offset=1, length=1)]},
        ),
    )

    combined = _prepend_system_tokens(prefix, prompt)

    assert combined.token_ids == [1, 2, 3, 4, 5]
    assert combined.message_indices == [-1, -1, -1, 0, -1]
    assert combined.sampled_mask == [False] * 5
    assert combined.is_content == [False, False, False, True, False]
    assert combined.message_roles == ["user"]
    assert combined.multi_modal_data is not None
    assert combined.multi_modal_data.mm_placeholders["image"][0].offset == 4


@pytest.mark.asyncio
async def test_train_client_sends_task_system_prefix_as_separate_rendered_block(
    monkeypatch,
) -> None:
    class FakeRenderer:
        def render(self, messages, *, tools=None, add_generation_prompt=False):
            if messages == [{"role": "system", "content": "<demo>expert</demo>"}]:
                assert tools is None
                assert add_generation_prompt is False
                return RenderedTokens(token_ids=[90, 91])
            assert messages == [{"role": "user", "content": "question"}]
            assert add_generation_prompt is True
            return RenderedTokens(
                token_ids=[1, 2, 3],
                message_indices=[0, 0, -1],
                sampled_mask=[False, False, False],
                is_content=[False, True, False],
                message_roles=["user"],
                message_tool_names=[None],
            )

    class FakeSlot:
        renderer = FakeRenderer()

        async def run(self, fn):
            return fn()

    class FakePool:
        @asynccontextmanager
        async def acquire(self):
            yield FakeSlot()

    captured = {}

    async def fake_generate(**kwargs):
        captured.update(kwargs)
        return {
            "request_id": "request-prefix",
            "prompt_ids": kwargs["prompt_ids"],
            "completion_ids": [4],
            "completion_logprobs": [-0.1],
            "content": "answer",
            "finish_reason": "stop",
            "prompt_attribution": kwargs["prompt_attribution"],
        }

    monkeypatch.setattr("verifiers.v1.clients.train.ElasticRendererPool", lambda *args, **kwargs: FakePool())
    monkeypatch.setattr("renderers.client.generate", fake_generate)
    client = TrainClient.__new__(TrainClient)
    client.config = TrainClientConfig(
        base_url="http://localhost:8000/v1",
        task_system_prefix_field="demonstration",
        task_system_prefix_template="<demo>{value}</demo>",
    )
    client.client = object()
    data = vf.WireTaskData(prompt="question", demonstration="expert")
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=data),
    )
    prompt = [vf.UserMessage(content="question")]
    turn = PendingTurn(trace=trace, prompt=prompt, prefix_node_ids=[], path_len=0)

    response = await client.get_response(
        ChatDialect(),
        {"messages": [{"role": "user", "content": "question"}]},
        "model",
        SamplingConfig(max_tokens=8),
        turn=turn,
    )

    assert captured["prompt_ids"] == [90, 91, 1, 2, 3]
    assert captured["prompt_attribution"].message_indices == [-1, -1, 0, 0, -1]
    assert response.tokens is not None
    assert response.tokens.prompt_ids == [90, 91, 1, 2, 3]
