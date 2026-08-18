from types import SimpleNamespace

import pytest

from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.interception.server import InterceptionServer


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
