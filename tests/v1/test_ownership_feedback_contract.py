import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from ownership_invariant_v1.feedback import (
    FEEDBACK_SCHEMA_VERSION,
    OwnershipFailureCode,
    OwnershipFailureSignals,
    diagnose_ownership_failure,
    feedback_contract_payload,
    render_ownership_feedback,
)
from ownership_invariant_v1.taskset import (
    OwnershipInvariantData,
    OwnershipInvariantEnv,
    _ownership_causal_diagnostic,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage
from verifiers.v1.utils.loaders import environment_class


def _signals(**updates) -> OwnershipFailureSignals:
    values = {
        "ownership": "child",
        "resource_family": "json_sum",
        "expected_child": "ledger-worker",
        "resource_path": "/workspace/data.json",
        "coordinator_ipython_calls": 1,
        "spawn_calls": 1,
        "strict_success": False,
        "state_retained": True,
        "state_precedes_spawn": True,
        "retained_handle": True,
        "expected_child_selected": True,
        "delegated_path": True,
        "parent_path_access": False,
        "local_state_leaked": False,
        "prohibited_control": False,
        "post_spawn_action": False,
        "direct_answer_accurate": False,
    }
    values.update(updates)
    return OwnershipFailureSignals(**values)


def _data() -> OwnershipInvariantData:
    return OwnershipInvariantData(
        idx=0,
        name="test",
        prompt="coordinate",
        ownership="child",
        split="admission",
        resource_family="json_sum",
        phrasing_variant=0,
        state_name="request_tag",
        state_value="coord-json-sum",
        expected_child="ledger-worker",
        resource_path="/workspace/data.json",
        resource_content="[1,2]",
        expected_result=3,
        operation="sum JSON integers",
        files={"/workspace/data.json": "[1,2]"},
    )


def _trace(code: str) -> vf.Trace:
    call_id = "call-0"
    return vf.Trace(
        id="ownership-feedback-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="OwnershipInvariantTask", data=vf.TaskData(idx=0)),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="coordinate"), sampled=False),
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            name="ipython",
                            arguments=json.dumps({"code": code}),
                        )
                    ],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=1,
                message=ToolMessage(
                    tool_call_id=call_id,
                    content="RLMSpawnHandle(name='ledger-worker')",
                ),
                sampled=False,
            ),
            MessageNode(parent=2, message=AssistantMessage(content="waiting"), sampled=True),
        ],
    )


def test_success_has_no_diagnostic() -> None:
    assert diagnose_ownership_failure(_signals(strict_success=True)) is None


def test_child_resource_access_has_priority_over_partial_successes() -> None:
    diagnostic = diagnose_ownership_failure(
        _signals(
            parent_path_access=True,
            state_retained=False,
            retained_handle=False,
        )
    )

    assert diagnostic is not None
    assert diagnostic.code is OwnershipFailureCode.CHILD_RESOURCE_ACCESSED_BY_COORDINATOR
    assert "/workspace/data.json" in render_ownership_feedback(diagnostic)


def test_direct_arm_rejects_delegation_without_revealing_result() -> None:
    diagnostic = diagnose_ownership_failure(
        _signals(
            ownership="coordinator",
            spawn_calls=1,
            parent_path_access=True,
        )
    )

    assert diagnostic is not None
    assert diagnostic.code is OwnershipFailureCode.UNNECESSARY_DELEGATION
    assert "handling it locally" in render_ownership_feedback(diagnostic)


def test_contract_is_typed_exact_and_answer_free() -> None:
    secret_answer = "NEVER-LEAK-EXPECTED-RESULT-7f42"
    diagnostic = diagnose_ownership_failure(_signals(retained_handle=False))

    assert diagnostic is not None
    payload = feedback_contract_payload(diagnostic)
    assert payload["schema_version"] == FEEDBACK_SCHEMA_VERSION
    assert payload["answer_free"] is True
    assert payload["retryable"] is True
    assert payload["message"] == render_ownership_feedback(diagnostic)
    assert secret_answer not in json.dumps(payload, sort_keys=True)
    assert set(payload["resource"]) == {"path", "expected_child"}


def test_trace_diagnostic_uses_observed_ast_violation() -> None:
    code = (
        "request_tag = 'coord-json-sum'\n"
        "raw = open('/workspace/data.json').read()\n"
        "child = await rlm('Read /workspace/data.json', name='ledger-worker')"
    )

    diagnostic = _ownership_causal_diagnostic(_trace(code), _data(), "semantic")

    assert diagnostic is not None
    assert diagnostic.code is OwnershipFailureCode.CHILD_RESOURCE_ACCESSED_BY_COORDINATOR
    assert diagnostic.signals.parent_path_access is True


def test_taskset_exports_its_diagnostic_environment() -> None:
    assert environment_class("ownership-invariant-v1") is OwnershipInvariantEnv


class _FakeAgent:
    def __init__(self, trace: vf.Trace) -> None:
        self.trace = trace

    @asynccontextmanager
    async def interaction(self, task):
        async def turn():
            return SimpleNamespace(terminated=False)

        yield SimpleNamespace(trace=self.trace, turn=turn)


def test_environment_records_exact_typed_feedback_for_failed_decision() -> None:
    code = (
        "request_tag = 'coord-json-sum'\n"
        "raw = open('/workspace/data.json').read()\n"
        "child = await rlm('Read /workspace/data.json', name='ledger-worker')"
    )
    trace = _trace(code)
    env = object.__new__(OwnershipInvariantEnv)
    env.taskset = SimpleNamespace(config=SimpleNamespace(record_causal_feedback=True))
    task = SimpleNamespace(data=_data(), config=SimpleNamespace(yield_policy="semantic"))

    asyncio.run(env.run(task, SimpleNamespace(agent=_FakeAgent(trace))))

    contract = trace.info["feedback_contract"]
    assert contract["schema_version"] == FEEDBACK_SCHEMA_VERSION
    assert contract["code"] == "child_resource_accessed_by_coordinator"
    assert contract["message"] == trace.info["feedback"]
    assert trace.info["ownership_causal_feedback_contracts"] == [contract]
