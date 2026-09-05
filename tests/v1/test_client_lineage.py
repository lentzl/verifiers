from types import SimpleNamespace

import pytest

from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.clients.client import CLIENT_SESSION_ID_HEADER, SESSION_ID_HEADER
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.clients.train import forwarded_session_headers
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.types import SamplingConfig


@pytest.mark.asyncio
async def test_eval_client_replaces_intercepted_affinity_headers() -> None:
    client = EvalClient(
        EvalClientConfig(
            base_url="http://provider.test/v1",
            api_key_var="UNSET_TEST_API_KEY",
        )
    )
    try:
        headers = client._headers(
            ChatDialect(),
            {
                "session_id": "prime-agent-child",
                "x-client-request-id": "prime-agent-child",
                "x-session-affinity": "prime-agent-child",
            },
            "rollout-trace",
        )
    finally:
        await client.close()

    assert "session_id" not in headers
    assert "x-client-request-id" not in headers
    assert "x-session-affinity" not in headers
    assert headers["x-session-id"] == "rollout-trace"


def test_record_call_preserves_client_session_id() -> None:
    server = InterceptionServer()
    session = SimpleNamespace(
        released=False,
        trace=SimpleNamespace(id="trace", calls=[]),
    )

    server.record_call(
        session,
        ChatDialect(),
        {"model": "test-model", "messages": []},
        1.0,
        client_session_id="prime-agent-child",
    )

    assert session.trace.calls[0].client_session_id == "prime-agent-child"


def test_train_client_forwards_branch_id_separately_from_rollout_id() -> None:
    headers = forwarded_session_headers(
        session_id="rollout-trace",
        headers={"session_id": "prime-agent-child"},
    )

    assert headers == {
        SESSION_ID_HEADER: "rollout-trace",
        CLIENT_SESSION_ID_HEADER: "prime-agent-child",
    }


def test_chat_eval_max_tokens_removes_competing_program_alias() -> None:
    body = {
        "model": "program-model",
        "messages": [],
        "max_tokens": 8192,
        "max_completion_tokens": 24576,
    }

    steered = ChatDialect().apply_overrides(
        body,
        "eval-model",
        SamplingConfig(max_tokens=3072),
    )

    assert steered["model"] == "eval-model"
    assert steered["max_tokens"] == 3072
    assert "max_completion_tokens" not in steered


def test_chat_preserves_program_max_tokens_when_eval_has_no_limit() -> None:
    body = {
        "model": "program-model",
        "messages": [],
        "max_completion_tokens": 24576,
    }

    steered = ChatDialect().apply_overrides(
        body,
        "eval-model",
        SamplingConfig(),
    )

    assert steered["max_completion_tokens"] == 24576
