import json
from types import SimpleNamespace

import pytest
from adaptive_skill_stream_v1.taskset import (
    SYSTEM_PROMPT,
    AdaptiveSkillStreamConfig,
    AdaptiveSkillStreamData,
    AdaptiveSkillStreamTask,
    AdaptiveSkillStreamTaskset,
    SkillStreamRound,
    _extract_answer,
    _round_prompt,
    _round_score,
    _valid_frontier_skill,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.prime_agent.harness import PrimeAgentHarness
from verifiers.v1.types import AssistantMessage, ToolCall, UserMessage


def _task(family="stable"):
    return AdaptiveSkillStreamTask(
        AdaptiveSkillStreamData(
            idx=0,
            name=f"{family}-test",
            prompt=None,
            family=family,
            template_variant=0,
            project_context="test context",
            rounds=(
                SkillStreamRound(
                    instruction="Return the result.",
                    answer=["ready"],
                    files={"/workspace/inbox/input.json": "{}"},
                ),
            ),
        )
    )


def _trace(task, code=""):
    messages = [
        MessageNode(
            parent=None,
            message=UserMessage(content="solve"),
            sampled=False,
        )
    ]
    if code:
        messages.append(
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="ipython",
                            arguments=json.dumps({"code": code}),
                        )
                    ],
                ),
                sampled=True,
            )
        )
    return vf.Trace(
        id="adaptive-skill-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        nodes=messages,
    )


def test_taskset_balances_lifecycle_families_and_holds_out_variants():
    train = AdaptiveSkillStreamTaskset(
        AdaptiveSkillStreamConfig(split="train", instances_per_template=1)
    ).load()
    evaluation = AdaptiveSkillStreamTaskset(
        AdaptiveSkillStreamConfig(split="eval", instances_per_template=1)
    ).load()

    assert len(train) == 12
    assert len(evaluation) == 6
    assert {task.data.family for task in train} == {
        "installed",
        "stable",
        "ephemeral",
    }
    assert {task.data.template_variant for task in train} == {0, 1, 2, 3}
    assert {task.data.template_variant for task in evaluation} == {4, 5}
    assert all(len(task.data.rounds) == 4 for task in [*train, *evaluation])


def test_generated_streams_have_distinct_persistence_contracts():
    tasks = AdaptiveSkillStreamTaskset(
        AdaptiveSkillStreamConfig(split="train", instances_per_template=1)
    ).load()
    by_family = {task.data.family: task for task in tasks[:3]}

    assert "installed skill" in by_family["installed"].data.project_context
    assert "stable across projects" in by_family["stable"].data.project_context
    assert "temporary mapping" in by_family["ephemeral"].data.project_context
    ephemeral_codebooks = [
        json.loads(round_.files["/workspace/inbox/codebook.json"])["codebook"]
        for round_ in by_family["ephemeral"].data.rounds
    ]
    assert len({json.dumps(value, sort_keys=True) for value in ephemeral_codebooks}) > 1


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('<answer>["a","b"]</answer>', ["a", "b"]),
        ('analysis\n<answer>{"b":2,"a":1}</answer>', {"a": 1, "b": 2}),
        ("not json", None),
    ],
)
def test_extract_answer_requires_json(reply, expected):
    assert _extract_answer(reply) == expected


def test_system_prompt_uses_plain_json_without_colliding_with_tool_parser():
    assert "JSON value only" in SYSTEM_PROMPT
    assert "<answer>" not in SYSTEM_PROMPT
    assert "future batches do not exist" in SYSTEM_PROMPT
    assert "never call tools named answer" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("family", "actual", "expected", "score"),
    [
        ("installed", {"a": 1, "b": 2}, {"a": 1, "b": 2}, 1.0),
        ("installed", {"result": {"a": 1}}, {"a": 1, "b": 2}, 0.5),
        ("stable", ["a", "b"], ["a", "b"], 1.0),
        ("stable", [{"name": "a"}], ["a", "b"], 0.5),
        ("ephemeral", ["x", "wrong", "z"], ["x", "y", "z"], 2 / 3),
        ("ephemeral", None, ["x"], 0.0),
    ],
)
def test_round_score_provides_semantic_family_specific_credit(
    family, actual, expected, score
):
    assert _round_score(family, actual, expected) == pytest.approx(score)


def test_frontier_promotion_requires_portable_metadata_and_procedure():
    valid = """---
name: project-frontier
description: Compute the reusable ready-task frontier from project dependency state.
---

Read requirements, completed, and blocked. The frontier contains unfinished,
unblocked tasks whose requirements are all completed.
"""
    invalid_metadata = valid.replace("name: project-frontier", "title: frontier")
    invalid_body = valid.replace("requirements", "dependencies").replace(
        "completed", "finished"
    )

    assert _valid_frontier_skill(valid)
    assert not _valid_frontier_skill(invalid_metadata)
    assert not _valid_frontier_skill(invalid_body)


def test_round_feedback_reports_outcome_without_leaking_answer():
    task = _task()

    failed = _round_prompt(task, 0, False)

    assert "failed validation" in failed
    assert "ready" not in failed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("family", "code", "paths", "valid_paths", "expected"),
    [
        (
            "installed",
            "portable_record_normalization.summarize(records)",
            [],
            [],
            1.0,
        ),
        ("installed", "result = {}", [], [], 0.0),
        ("stable", "result = []", ["/workspace/.agents/skills/x/SKILL.md"], [], 0.0),
        (
            "stable",
            "result = []",
            ["/workspace/.agents/skills/x/SKILL.md"],
            ["/workspace/.agents/skills/x/SKILL.md"],
            1.0,
        ),
        ("ephemeral", "result = []", [], [], 1.0),
        ("ephemeral", "result = []", ["/workspace/.agents/skills/x/SKILL.md"], [], 0.0),
    ],
)
async def test_lifecycle_reward_distinguishes_reuse_promotion_and_restraint(
    family, code, paths, valid_paths, expected
):
    task = _task(family)
    trace = _trace(task, code)
    trace.info["adaptive_skill_artifacts"] = {
        "portable_skill_paths": paths,
        "valid_frontier_skill_paths": valid_paths,
    }

    await task.score(trace)

    assert trace.rewards["skill_lifecycle"].score == expected


@pytest.mark.asyncio
async def test_finalize_harvests_portable_and_local_harness_state_before_cleanup():
    task = _task()
    trace = _trace(task)
    skill_path = "/workspace/.agents/skills/project-frontier/SKILL.md"
    skill = b"""---
name: project-frontier
description: Compute the reusable ready-task frontier from project dependency state.
---
Read requirements, completed, and blocked. The frontier contains tasks whose
requirements are completed and which are not blocked.
"""
    local_path = PrimeAgentHarness.harness_state_path(trace)
    session_path = (
        f"{PrimeAgentHarness.agent_dir(trace)}/session-artifacts/session-1/"
        "harness/harness_state.json"
    )
    local_state = json.dumps(
        {
            "schema": 1,
            "entries": {"memory": {"m1": {}}, "skill": {"s1": {}}},
            "refinements": [],
        }
    ).encode()
    global_state = json.dumps(
        {
            "schema": 1,
            "entries": {"prompt": {"p1": {}}},
            "refinements": [],
        }
    ).encode()

    class Runtime:
        async def run(self, argv, env):
            del env
            if argv[0] == "find":
                if argv[1] == PrimeAgentHarness.agent_dir(trace):
                    return SimpleNamespace(
                        exit_code=0,
                        stdout=f"{local_path}\n{session_path}\n",
                        stderr="",
                    )
                return SimpleNamespace(exit_code=0, stdout=f"{skill_path}\n", stderr="")
            if argv[:2] == ["test", "-f"]:
                return SimpleNamespace(
                    exit_code=0 if argv[2] in {local_path, session_path} else 1,
                    stdout="",
                    stderr="",
                )
            raise AssertionError(argv)

        async def read(self, path, max_bytes=None):
            del max_bytes
            if path == skill_path:
                return skill
            if path == local_path:
                return global_state
            if path == session_path:
                return local_state
            raise AssertionError(path)

    await task.finalize(trace, Runtime())

    artifacts = trace.info["adaptive_skill_artifacts"]
    assert artifacts["portable_skill_paths"] == [skill_path]
    assert artifacts["valid_frontier_skill_paths"] == [skill_path]
    assert artifacts["local_harness_entries"] == 2
    assert artifacts["local_harness_kinds"] == {"memory": 1, "skill": 1}
    assert artifacts["global_harness_entries"] == 1
    assert artifacts["global_harness_kinds"] == {"prompt": 1}
