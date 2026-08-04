"""Compositional reasoning-offload task generation and metrics."""

import ast
import json
import subprocess
import sys

import pytest

import verifiers.v1 as vf
from reasoning_offload_composition_v1.generators import generate
from reasoning_offload_composition_v1.taskset import (
    INCORRECT_ANSWER_FEEDBACK,
    ReasoningOffloadCompositionConfig,
    ReasoningOffloadCompositionTaskset,
    _mutates_operations_file,
    _provided_operations_reused,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage


def _agent_info() -> vf.AgentInfo:
    return vf.AgentInfo(
        config=vf.AgentConfig(model="test", harness={"id": "null"})
    )


def test_train_and_eval_use_disjoint_pipeline_variants():
    train = ReasoningOffloadCompositionTaskset(
        ReasoningOffloadCompositionConfig(split="train", instances_per_template=1)
    ).load()
    evaluation = ReasoningOffloadCompositionTaskset(
        ReasoningOffloadCompositionConfig(split="eval", instances_per_template=1)
    ).load()

    assert len(train) == 4
    assert len(evaluation) == 2
    assert {task.data.template_variant for task in train} == {0, 1, 2, 3}
    assert {task.data.template_variant for task in evaluation} == {4, 5}


@pytest.mark.parametrize("split", ["train", "eval"])
def test_generated_answer_matches_provided_pipeline(split):
    variants = range(4) if split == "train" else range(4, 6)
    for instance in range(2):
        for variant in variants:
            generated = generate(variant, instance, 20260722)
            namespace = {}
            exec(generated.correct_source, namespace)
            value = json.loads(generated.files["inputs/target.json"])
            manifest = json.loads(generated.files["inputs/pipeline.json"])
            for step in manifest["steps"]:
                value = namespace[step](value)
            assert (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                == generated.answer
            )


@pytest.mark.parametrize("variant", range(6))
def test_injected_defect_fails_at_least_one_checker_case(variant):
    generated = generate(variant, 0, 20260722)
    namespace = {}
    exec(generated.files["inputs/operations.py"], namespace)
    manifest = json.loads(generated.files["inputs/pipeline.json"])
    cases = json.loads(generated.files["inputs/cases.json"])

    actual = []
    for case in cases:
        value = case["input"]
        for step in manifest["steps"]:
            value = namespace[step](value)
        actual.append(value)

    assert any(
        value != case["expected"] for value, case in zip(actual, cases, strict=True)
    )


@pytest.mark.parametrize("variant", range(6))
def test_checker_fails_then_verifies_from_workspace_root(tmp_path, variant):
    generated = generate(variant, 0, 20260722)
    for relative_path, content in generated.files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    failed = subprocess.run(
        [sys.executable, "inputs/check.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert failed.returncode == 1
    assert "FAILED" in failed.stdout
    assert "ModuleNotFoundError" not in failed.stderr

    (tmp_path / "inputs/operations.py").write_text(generated.correct_source)
    verified = subprocess.run(
        [sys.executable, "inputs/check.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert verified.returncode == 0
    assert verified.stdout.strip() == "VERIFIED"


@pytest.mark.parametrize(
    "code",
    [
        "from pathlib import Path\nPath('inputs/operations.py').write_text(source)",
        "open('inputs/operations.py', 'w').write(source)",
        "%%bash\nsed -i 's/wrong/right/' inputs/operations.py",
    ],
)
def test_operation_mutation_recognizes_file_writes(code):
    assert _mutates_operations_file(code)


def test_operation_mutation_rejects_notebook_only_repair():
    assert not _mutates_operations_file("def operation(value): return value")


def test_operation_reuse_recognizes_module_and_function_imports():
    assert _provided_operations_reused(
        [ast.parse("import inputs.operations as ops\nvalue = ops.normalize(value)")]
    )
    assert _provided_operations_reused(
        [ast.parse("from inputs.operations import normalize\nvalue = normalize(value)")]
    )
    assert _provided_operations_reused(
        [ast.parse("import operations\nvalue = getattr(operations, step)(value)")]
    )
    assert not _provided_operations_reused(
        [ast.parse("def normalize(value): return value\nvalue = normalize(value)")]
    )


@pytest.mark.asyncio
async def test_unquoted_string_answer_is_accepted_as_json_string():
    task = ReasoningOffloadCompositionTaskset(
        ReasoningOffloadCompositionConfig(
            split="eval", instances_per_template=1, seed=20260722
        )
    ).load()[1]
    expected = json.loads(task.data.answer)
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=_agent_info(),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content=task.data.prompt),
                sampled=False,
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(content=f"<answer>{expected}</answer>"),
                sampled=True,
            ),
        ],
    )

    await task.score(trace)

    assert trace.reward == 1.0


@pytest.mark.asyncio
async def test_incorrect_answer_adds_non_leaking_feedback():
    task = ReasoningOffloadCompositionTaskset(
        ReasoningOffloadCompositionConfig(instances_per_template=1)
    ).load()[0]
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        agent=_agent_info(),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content=task.data.prompt),
                sampled=False,
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="<answer>wrong</answer>"),
                sampled=True,
            ),
        ],
    )

    await task.score(trace)

    assert trace.reward == 0.0
    assert trace.info["feedback"] == INCORRECT_ANSWER_FEEDBACK
    assert task.data.answer not in trace.info["feedback"]
