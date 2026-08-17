import asyncio

import pytest

from verifiers.v1.acp import ACPHarnessSession


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
