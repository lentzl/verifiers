import json
from types import SimpleNamespace

import pytest
from programmatic_episodic_memory_v2.feedback import (
    FEEDBACK_SCHEMA_VERSION,
    MemoryFailureCode,
    feedback_contract_payload,
)
from programmatic_episodic_memory_v2.taskset import (
    DEMONSTRATION_TEMPLATE,
    ProgrammaticEpisodicMemoryConfig,
    ProgrammaticEpisodicMemoryData,
    ProgrammaticEpisodicMemoryEnv,
    ProgrammaticEpisodicMemoryTaskset,
    _behavior,
    _causal_diagnostic,
    _causal_feedback,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


def _row(*, family: str = "latest_state", requires_history: bool = True) -> dict:
    history_path = "/workspace/history.log"
    code = (
        f"from pathlib import Path\nrows = Path({history_path!r}).read_text().splitlines()\nrows[-1]"
        if requires_history
        else "sum([20, 22])"
    )
    messages = [
        {"role": "system", "content": "Use evidence and answer concisely."},
        {"role": "user", "content": "What is the value?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "demo-1",
                    "type": "function",
                    "function": {
                        "name": "ipython",
                        "arguments": json.dumps({"code": code}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "demo-1", "content": "'42'"},
        {"role": "assistant", "content": "42"},
    ]
    metadata = {
        "split": "familiar_heldout",
        "family": family,
        "domain": "project",
        "instance": 7,
        "instruction_level": "natural",
        "history_path": history_path,
        "history_format": "kv",
        "requires_history": requires_history,
        "uses_ipython": True,
        "retrieval_policy": "latest_valid",
        "expected_answer": "42",
    }
    return {
        "messages_json": json.dumps(messages),
        "tools": "[]",
        "workspace_files_json": json.dumps({history_path: "old\n42\n"}),
        "metadata_json": json.dumps(metadata),
    }


def _data(
    *,
    family: str = "latest_state",
    requires_history: bool = True,
    uses_ipython: bool = True,
    expected_answers: tuple[str, ...] = ("42",),
) -> ProgrammaticEpisodicMemoryData:
    return ProgrammaticEpisodicMemoryData(
        idx=0,
        name="test",
        prompt="What is the value?",
        split="familiar_heldout",
        family=family,
        domain="project",
        instruction_level="natural",
        history_path="/workspace/history.log",
        history_format="kv",
        requires_history=requires_history,
        uses_ipython=uses_ipython,
        retrieval_policy="latest_valid",
        expected_answers=expected_answers,
        demonstration="assistant tool: ipython\n{}\ntool result: 42\nassistant: 42",
        files={"/workspace/history.log": "old\n42\n"},
    )


def _trace(
    calls: list[tuple[str, str]],
    *,
    answers: tuple[str, ...] = ("42",),
    outputs: tuple[str, ...] | None = None,
) -> vf.Trace:
    nodes = [
        MessageNode(parent=None, message=UserMessage(content="question"), sampled=False)
    ]
    parent = 0
    outputs = outputs or tuple("42" for _ in calls)
    for index, ((call_id, code), output) in enumerate(zip(calls, outputs, strict=True)):
        nodes.append(
            MessageNode(
                parent=parent,
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
            )
        )
        parent = len(nodes) - 1
        nodes.append(
            MessageNode(
                parent=parent,
                message=ToolMessage(tool_call_id=call_id, content=output),
                sampled=False,
            )
        )
        parent = len(nodes) - 1
        if index < len(answers) - 1:
            nodes.append(
                MessageNode(
                    parent=parent,
                    message=AssistantMessage(content=answers[index]),
                    sampled=True,
                )
            )
            parent = len(nodes) - 1
            nodes.append(
                MessageNode(
                    parent=parent,
                    message=UserMessage(content="follow up"),
                    sampled=False,
                )
            )
            parent = len(nodes) - 1
    for answer in answers[len(calls) - 1 :]:
        nodes.append(
            MessageNode(
                parent=parent,
                message=AssistantMessage(content=answer),
                sampled=True,
            )
        )
        parent = len(nodes) - 1
    return vf.Trace(
        id="programmatic-memory-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="ProgrammaticEpisodicMemoryTask", data=vf.TaskData(idx=0)
        ),
        nodes=nodes,
    )


def test_loader_preserves_frozen_row_and_exposes_demonstration(tmp_path) -> None:
    path = tmp_path / "familiar_heldout.jsonl"
    path.write_text(json.dumps(_row()) + "\n")

    task = ProgrammaticEpisodicMemoryTaskset(
        ProgrammaticEpisodicMemoryConfig(
            dataset_path=str(path),
            split="familiar_heldout",
        )
    ).load()[0]

    assert task.data.system_prompt == "Use evidence and answer concisely."
    assert task.data.expected_answers == ("42",)
    assert task.data.files == {"/workspace/history.log": "old\n42\n"}
    assert "assistant tool: ipython" in task.data.demonstration
    assert "reasoning_content" not in task.data.demonstration


def test_conditioned_admission_changes_only_system_context(tmp_path) -> None:
    path = tmp_path / "familiar_heldout.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    base = ProgrammaticEpisodicMemoryTaskset(
        ProgrammaticEpisodicMemoryConfig(
            dataset_path=str(path),
            split="familiar_heldout",
        )
    ).load()[0]
    conditioned = ProgrammaticEpisodicMemoryTaskset(
        ProgrammaticEpisodicMemoryConfig(
            dataset_path=str(path),
            split="familiar_heldout",
            condition_on_demonstration=True,
        )
    ).load()[0]

    assert conditioned.data.prompt == base.data.prompt
    assert conditioned.data.files == base.data.files
    assert conditioned.data.expected_answers == base.data.expected_answers
    assert conditioned.data.system_prompt == (
        f"{DEMONSTRATION_TEMPLATE.format(demonstration=base.data.demonstration)}\n\n"
        f"{base.data.system_prompt}"
    )


def test_per_family_window_is_balanced(tmp_path) -> None:
    path = tmp_path / "familiar_heldout.jsonl"
    rows = []
    for family in ("latest_state", "checkpoint_resume"):
        for instance in range(4):
            row = _row(family=family)
            metadata = json.loads(row["metadata_json"])
            metadata["instance"] = instance
            row["metadata_json"] = json.dumps(metadata)
            rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    tasks = ProgrammaticEpisodicMemoryTaskset(
        ProgrammaticEpisodicMemoryConfig(
            dataset_path=str(path),
            split="familiar_heldout",
            instance_offset=2,
            instances_per_family=1,
        )
    ).load()

    assert len(tasks) == 2
    assert {task.data.family for task in tasks} == {
        "latest_state",
        "checkpoint_resume",
    }


def test_loader_rejects_split_mismatch_and_reasoning_content(tmp_path) -> None:
    mismatch = tmp_path / "mismatch.jsonl"
    mismatch.write_text(json.dumps(_row()) + "\n")
    with pytest.raises(ValueError, match="does not match configured split"):
        ProgrammaticEpisodicMemoryTaskset(
            ProgrammaticEpisodicMemoryConfig(dataset_path=str(mismatch), split="train")
        ).load()

    row = _row()
    messages = json.loads(row["messages_json"])
    messages[2]["reasoning_content"] = "fabricated"
    row["messages_json"] = json.dumps(messages)
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="reasoning_content"):
        ProgrammaticEpisodicMemoryTaskset(
            ProgrammaticEpisodicMemoryConfig(
                dataset_path=str(invalid),
                split="familiar_heldout",
            )
        ).load()


def test_history_task_requires_grounded_retrieval() -> None:
    correct = _behavior(
        _trace(
            [
                (
                    "call-1",
                    "from pathlib import Path\nPath('/workspace/history.log').read_text()",
                )
            ]
        ),
        _data(),
    )
    shortcut = _behavior(
        _trace([("call-1", "print(42)")]),
        _data(),
    )

    assert correct["strict_success"] == 1.0
    assert shortcut["answer_correct"] == 1.0
    assert shortcut["retrieval_decision"] == 0.0
    assert shortcut["strict_success"] == 0.0


def test_current_turn_override_requires_not_reading_history() -> None:
    data = _data(
        family="prompt_override_control",
        requires_history=False,
        uses_ipython=False,
    )
    direct = _behavior(_trace([], answers=("42",)), data)
    stale_lookup = _behavior(
        _trace([("call-1", "open('/workspace/history.log').read()")]),
        data,
    )

    assert direct["strict_success"] == 1.0
    assert direct["current_turn_override"] == 1.0
    assert stale_lookup["retrieval_decision"] == 0.0
    assert stale_lookup["current_turn_override"] == 0.0


def test_repeated_lookup_requires_cross_turn_state_reuse() -> None:
    data = _data(
        family="repeated_lookup_index",
        expected_answers=("alpha-42", "beta-17"),
    )
    first = (
        "from pathlib import Path\n"
        "bindings = {'alpha': 'alpha-42', 'beta': 'beta-17'}\n"
        "Path('/workspace/history.log').read_text()\n"
        "bindings['alpha']"
    )
    reused = _behavior(
        _trace(
            [("call-1", first), ("call-2", "bindings['beta']")],
            answers=("alpha-42", "beta-17"),
        ),
        data,
    )
    reread = _behavior(
        _trace(
            [
                ("call-1", first),
                ("call-2", "open('/workspace/history.log').read(); 'beta-17'"),
            ],
            answers=("alpha-42", "beta-17"),
        ),
        data,
    )

    assert reused["persistent_index_reuse"] == 1.0
    assert reused["strict_success"] == 1.0
    assert reread["persistent_index_reuse"] == 0.0
    assert reread["strict_success"] == 0.0


def test_identical_repeated_cells_are_rejected() -> None:
    code = "from pathlib import Path\nPath('/workspace/history.log').read_text()"
    behavior = _behavior(
        _trace([("call-1", code), ("call-2", code)]),
        _data(),
    )

    assert behavior["no_repeated_cell"] == 0.0
    assert behavior["strict_success"] == 0.0


def test_history_dump_is_not_selective_retrieval() -> None:
    behavior = _behavior(
        _trace(
            [("call-1", "open('/workspace/history.log').read()")],
            outputs=("x" * 5000,),
        ),
        _data(),
    )

    assert behavior["answer_correct"] == 1.0
    assert behavior["retrieval_decision"] == 1.0
    assert behavior["bounded_retrieval"] == 0.0
    assert behavior["observation_chars"] == 5000.0
    assert behavior["strict_success"] == 0.0


def test_causal_feedback_reports_missing_history_without_revealing_answer() -> None:
    trace = _trace([("call-1", "print('guess')")], answers=("wrong",))

    feedback = _causal_feedback(
        trace,
        _data(expected_answers=("secret-42",)),
        expected="secret-42",
        call_start=0,
        turn_index=0,
    )

    assert feedback is not None
    assert "append-only history" in feedback
    assert "secret-42" not in feedback


def test_trace_adapter_emits_typed_missing_history_contract() -> None:
    secret = "secret-42"
    diagnostic = _causal_diagnostic(
        _trace([("call-1", "print('guess')")], answers=("wrong",)),
        _data(expected_answers=(secret,)),
        expected=secret,
        call_start=0,
        turn_index=0,
    )

    assert diagnostic is not None
    assert diagnostic.code is MemoryFailureCode.REQUIRED_HISTORY_NOT_RETRIEVED
    payload = feedback_contract_payload(diagnostic)
    assert payload["schema_version"] == FEEDBACK_SCHEMA_VERSION
    assert payload["code"] == "required_history_not_retrieved"
    assert payload["evidence"]["history_reads"] == 0
    assert secret not in json.dumps(payload, sort_keys=True)


def test_trace_adapter_classifies_real_tool_traceback() -> None:
    diagnostic = _causal_diagnostic(
        _trace(
            [
                (
                    "call-1",
                    (
                        "from pathlib import Path\n"
                        "Path('/workspace/history.log').read_text()"
                    ),
                )
            ],
            answers=("wrong",),
            outputs=(
                (
                    "Traceback (most recent call last):\n"
                    "  File \"<stdin>\", line 1, in <module>\n"
                    "NameError: name 'missing' is not defined"
                ),
            ),
        ),
        _data(),
        expected="42",
        call_start=0,
        turn_index=0,
    )

    assert diagnostic is not None
    assert diagnostic.code is MemoryFailureCode.TOOL_EXECUTION_ERROR
    assert diagnostic.tool_error is True
    assert "actual traceback" in feedback_contract_payload(diagnostic)["message"]


def test_causal_feedback_distinguishes_output_contract_from_semantics() -> None:
    trace = _trace(
        [
            (
                "call-1",
                "from pathlib import Path\nPath('/workspace/history.log').read_text()",
            )
        ],
        answers=("The value is 42.",),
    )

    feedback = _causal_feedback(
        trace,
        _data(),
        expected="42",
        call_start=0,
        turn_index=0,
    )

    assert feedback is not None
    assert "output contract" in feedback


def test_behavior_scores_only_final_answers_after_causal_retry() -> None:
    trace = _trace(
        [
            (
                "call-1",
                "from pathlib import Path\nPath('/workspace/history.log').read_text()",
            )
        ],
        answers=("The value is 42.", "42"),
    )
    trace.info["memory_final_answers"] = ["42"]
    trace.info["memory_causal_feedback"] = ["Return only the value."]

    behavior = _behavior(trace, _data())

    assert behavior["answer_correct"] == 1.0
    assert behavior["causal_feedback_count"] == 1.0
    assert behavior["causal_feedback_repair"] == 1.0


def test_causal_feedback_recording_is_opt_in_and_retry_independent() -> None:
    base = {
        "dataset_path": "/tmp/memory.jsonl",
        "split": "train",
    }

    default = ProgrammaticEpisodicMemoryConfig(**base)
    recorded = ProgrammaticEpisodicMemoryConfig(
        **base,
        record_causal_feedback=True,
        causal_feedback_retries=0,
    )

    assert default.record_causal_feedback is False
    assert recorded.record_causal_feedback is True
    assert recorded.causal_feedback_retries == 0


class _InteractionContext:
    def __init__(self, interaction) -> None:
        self.interaction = interaction

    async def __aenter__(self):
        return self.interaction

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False


class _RecordedFailureInteraction:
    def __init__(self, trace: vf.Trace, completed_trace: vf.Trace) -> None:
        self.trace = trace
        self.completed_trace = completed_trace

    async def turn(self, prompt=None):
        self.trace.nodes = self.completed_trace.nodes
        return SimpleNamespace(last_reply="wrong", terminated=False)


class _RetryInteraction:
    def __init__(
        self,
        trace: vf.Trace,
        failed_trace: vf.Trace,
        repaired_trace: vf.Trace,
    ) -> None:
        self.trace = trace
        self.traces = [failed_trace, repaired_trace]
        self.replies = ["wrong", "42"]

    async def turn(self, prompt=None):
        self.trace.nodes = self.traces.pop(0).nodes
        return SimpleNamespace(last_reply=self.replies.pop(0), terminated=False)


@pytest.mark.asyncio
async def test_environment_records_legacy_and_typed_feedback_without_answer_leak() -> None:
    secret = "NEVER-LEAK-42"
    trace = _trace([], answers=())
    completed_trace = _trace(
        [("call-1", "print('guess')")], answers=("wrong",)
    )
    interaction = _RecordedFailureInteraction(trace, completed_trace)
    agents = SimpleNamespace(
        agent=SimpleNamespace(
            interaction=lambda task: _InteractionContext(interaction)
        )
    )
    env = object.__new__(ProgrammaticEpisodicMemoryEnv)
    env.taskset = SimpleNamespace(
        config=ProgrammaticEpisodicMemoryConfig(
            dataset_path="/tmp/not-read.jsonl",
            split="train",
            record_causal_feedback=True,
            causal_feedback_retries=0,
        )
    )
    task = SimpleNamespace(data=_data(expected_answers=(secret,)))

    await env.run(task, agents)

    contract = trace.info["feedback_contract"]
    assert trace.info["feedback"] == contract["message"]
    assert trace.info["memory_causal_feedback"] == [contract["message"]]
    assert trace.info["memory_causal_feedback_contracts"] == [contract]
    assert contract["code"] == "required_history_not_retrieved"
    assert secret not in json.dumps(trace.info, sort_keys=True)


@pytest.mark.asyncio
async def test_retry_feedback_and_contract_lists_remain_aligned() -> None:
    trace = _trace([], answers=())
    failed_trace = _trace([("call-1", "print('guess')")], answers=("wrong",))
    repaired_trace = _trace(
        [
            ("call-1", "print('guess')"),
            (
                "call-2",
                (
                    "from pathlib import Path\n"
                    "Path('/workspace/history.log').read_text()"
                ),
            ),
        ],
        answers=("wrong", "42"),
    )
    interaction = _RetryInteraction(trace, failed_trace, repaired_trace)
    agents = SimpleNamespace(
        agent=SimpleNamespace(
            interaction=lambda task: _InteractionContext(interaction)
        )
    )
    env = object.__new__(ProgrammaticEpisodicMemoryEnv)
    env.taskset = SimpleNamespace(
        config=ProgrammaticEpisodicMemoryConfig(
            dataset_path="/tmp/not-read.jsonl",
            split="train",
            causal_feedback_retries=1,
        )
    )

    await env.run(SimpleNamespace(data=_data()), agents)

    feedback = trace.info["memory_causal_feedback"]
    contracts = trace.info["memory_causal_feedback_contracts"]
    assert len(feedback) == len(contracts) == 1
    assert feedback[0] == contracts[0]["message"]
    assert contracts[0]["code"] == "required_history_not_retrieved"
    assert trace.info["memory_final_answers"] == ["42"]
