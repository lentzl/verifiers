import asyncio
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from verifiers.v1.agent import Agent
from verifiers.v1.rollout import Rollout


class _Run:
    instances: ClassVar[list["_Run"]] = []

    def __init__(self, **kwargs) -> None:
        del kwargs
        self.closed = False
        self.trace = SimpleNamespace(agent=SimpleNamespace(runtime=None))
        self.cleanup_started = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.abort_task = None
        self.__class__.instances.append(self)

    async def open(self) -> bool:
        return True

    def start_abort(self):
        self.closed = True
        self.abort_task = asyncio.create_task(self._abort())
        return self.abort_task

    async def _abort(self) -> None:
        self.cleanup_started.set()
        await self.cleanup_release.wait()


@pytest.mark.asyncio
async def test_interaction_timeout_does_not_wait_for_teardown(monkeypatch):
    monkeypatch.setattr("verifiers.v1.agent.Rollout", _Run)
    agent = object.__new__(Agent)
    agent._closed = False
    agent._gate = None
    agent._check_resume_support = lambda: None
    agent._rollout_params = lambda *args, **kwargs: {}

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            async with agent.interaction(object()):
                await asyncio.Future()

    run = _Run.instances[-1]
    await run.cleanup_started.wait()
    assert run.closed
    assert not run.abort_task.done()

    run.cleanup_release.set()
    await run.abort_task


@pytest.mark.asyncio
async def test_start_abort_is_idempotent_and_retains_cleanup():
    rollout = object.__new__(Rollout)
    rollout._closed = False
    rollout._abort_task = None
    rollout._harness_session = None
    rollout._stack = SimpleNamespace(aclose=AsyncMock())
    rollout.runtime = None
    rollout._owns_runtime = False

    first = rollout.start_abort()
    second = rollout.start_abort()

    assert rollout.closed
    assert first is second
    await first
