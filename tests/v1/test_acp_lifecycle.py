import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.acp import ACPHarnessSession, _record_acp_stop


@pytest.mark.asyncio
@pytest.mark.parametrize("visible", ["Done.", ""])
@pytest.mark.parametrize("reason,condition,truncated", [
    ("end_turn", None, False),
    ("max_tokens", "acp_max_tokens", True),
    ("max_turn_requests", "acp_max_turn_requests", True),
    ("refusal", "refusal", False),
    ("cancelled", "cancelled", False),
])
async def test_acp_stop_reason_survives_visible_reply(reason, condition, truncated, visible):
    from verifiers.v1.acp.runner import VerifiersACPClient, prompt

    client = VerifiersACPClient()

    class Connection:
        async def prompt(self, **kwargs):
            client.visible_reply = visible
            return SimpleNamespace(stop_reason=reason)

    config = {"system_prompt": "", "user_contents": ["Summarize the chapter."]}
    if reason == "end_turn" and not visible:
        with pytest.raises(RuntimeError, match="no visible reply"):
            await prompt(client, Connection(), None, "test", config, is_new=True)
        return
    response = await prompt(client, Connection(), None, "test", config, is_new=True)
    assert response == {"reply": visible, "stop_reason": reason}
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0)),
    )
    _record_acp_stop(trace, response["stop_reason"])
    assert trace.info["acp_stop_reasons"] == [reason]
    assert trace.stop_condition == condition
    assert trace.is_truncated == truncated
    trace.stop("agent_completed")
    assert trace.stop_condition == (condition or "agent_completed")
    with pytest.raises(ValueError, match="unknown ACP stop reason"):
        _record_acp_stop(trace, "unexpected")


class _HungTerminateProcess:
    def __init__(self) -> None:
        self.kill_called = asyncio.Event()

    async def wait(self) -> None:
        await self.kill_called.wait()

    async def terminate(self) -> None:
        await asyncio.Future()

    async def kill(self) -> None:
        self.kill_called.set()


class _ExitedProcess:
    async def wait(self) -> None:
        return None

    async def terminate(self) -> None:
        raise AssertionError("terminate should not be called after process exit")

    async def kill(self) -> None:
        raise AssertionError("kill should not be called after process exit")


@pytest.mark.asyncio
async def test_acp_stop_escalates_when_terminate_hangs(monkeypatch):
    monkeypatch.setattr("verifiers.v1.acp.PROCESS_EXIT_POLL_TIMEOUT", 0.01)
    monkeypatch.setattr("verifiers.v1.acp.PROCESS_SIGNAL_TIMEOUT", 0.01)
    process = _HungTerminateProcess()
    session = object.__new__(ACPHarnessSession)
    session._process = process
    session._reader = None
    session._stderr_task = None

    await asyncio.wait_for(session._stop(graceful=False), timeout=0.1)

    assert process.kill_called.is_set()
    assert session._process is None


@pytest.mark.asyncio
async def test_acp_stop_does_not_wait_forever_for_stderr_drain(monkeypatch):
    monkeypatch.setattr("verifiers.v1.acp.PROCESS_SIGNAL_TIMEOUT", 0.01)
    release = asyncio.Event()

    async def resist_cancellation() -> None:
        while True:
            try:
                await release.wait()
                return
            except asyncio.CancelledError:
                continue

    stderr_task = asyncio.create_task(resist_cancellation())
    await asyncio.sleep(0)
    session = object.__new__(ACPHarnessSession)
    session._process = _ExitedProcess()
    session._reader = None
    session._stderr_task = stderr_task

    await asyncio.wait_for(session._stop(graceful=False), timeout=0.1)

    assert not stderr_task.done()
    assert session._stderr_task is None
    release.set()
    await asyncio.wait_for(stderr_task, timeout=0.1)


@pytest.mark.asyncio
async def test_acp_stderr_drain_rejects_unbounded_output(monkeypatch):
    monkeypatch.setattr("verifiers.v1.acp.MAX_STDERR_BYTES", 5)

    async def chunks():
        yield b"abc"
        yield b"def"

    session = object.__new__(ACPHarnessSession)
    session._stderr_tail = bytearray()

    with pytest.raises(RuntimeError, match="stderr exceeded 5 bytes"):
        await session._drain_stderr(chunks())

    assert session._stderr() == "abcdef"
