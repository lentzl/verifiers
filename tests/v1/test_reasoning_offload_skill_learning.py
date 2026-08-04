"""Portable Agent Skill validation and downstream-utility judgement."""

import json

import pytest
from reasoning_offload_skill_acquisition_v1.taskset import (
    ReasoningOffloadSkillAcquisitionConfig,
    ReasoningOffloadSkillAcquisitionData,
    ReasoningOffloadSkillAcquisitionTask,
    ReasoningOffloadSkillAcquisitionTaskset,
    SkillExperience,
)
from reasoning_offload_skill_learning_v1.env import (
    ReasoningOffloadSkillLearningEnv,
    _score_bounded_failure,
    author_task,
    consumer_task,
)
from reasoning_offload_skill_learning_v1.skill_package import (
    SkillPackageError,
    install_candidate,
    parse_candidate,
    render_candidate,
)

import verifiers.v1 as vf

VALID_SCRIPT = """from __future__ import annotations

import json
import sys


def compact(path: str) -> str:
    with open(path) as handle:
        value = json.load(handle)
    return json.dumps(value, separators=(",", ":"))


if __name__ == "__main__":
    print(compact(sys.argv[1]))
"""

VALID_SKILL_MD = """---
name: compact-json
description: Render a JSON file in a stable compact form.
---

Use this skill for deterministic compact JSON serialization.
Run 'python scripts/compact_json.py <input-path>' from this skill directory.
"""


def _reply(
    *,
    name: str = "compact-json",
    skill_md: str = VALID_SKILL_MD,
    files: dict[str, str] | None = None,
) -> str:
    payload = {
        "name": name,
        "skill_md": skill_md,
        "files": files or {"scripts/compact_json.py": VALID_SCRIPT},
    }
    return f"<skill_package>{json.dumps(payload)}</skill_package>"


def _task(answer: str = "held-out-answer") -> ReasoningOffloadSkillAcquisitionTask:
    return ReasoningOffloadSkillAcquisitionTask(
        ReasoningOffloadSkillAcquisitionData(
            idx=7,
            name="held-out-task",
            prompt="Read inputs/unseen.json and return held-out-answer.",
            system_prompt="Keep the answer exact.",
            family="canonicalization",
            split="validation",
            template_variant=7,
            answer=answer,
            files={"inputs/unseen.json": '"held-out-private-input"'},
            experiences=(
                SkillExperience(
                    prompt="Compact inputs/past.json.",
                    answer='["past-answer"]',
                    files={"inputs/past.json": '[ "past-answer" ]'},
                ),
            ),
        )
    )


def _trace(role: str, *, score: float | None = None) -> vf.Trace:
    data = vf.TaskData(idx=0, name=f"{role}-task", prompt="task")
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=data),
        agent=vf.AgentInfo(
            config=vf.AgentConfig(model="test", harness={"id": "null"}),
            name=role,
            trainable=role == "author",
        ),
        ok=True,
        is_completed=True,
    )
    if score is not None:
        trace.record_reward("exact_match", score)
    return trace


def test_bounded_consumer_failure_becomes_a_scored_miss():
    trace = _trace("baseline_user")
    trace.ok = False
    trace.errors.append(vf.Error(type="HarnessError", message="agent timeout"))

    normalized = _score_bounded_failure(trace, 2)

    assert normalized.ok
    assert not normalized.errors
    assert normalized.info["utility_case_index"] == 2
    assert normalized.info["utility_arm_errors"] == [
        {
            "type": "HarnessError",
            "message": "agent timeout",
            "status_code": None,
            "traceback": None,
        }
    ]


def test_candidate_renders_only_the_portable_agent_skill_core():
    candidate = parse_candidate(_reply())
    files = render_candidate(candidate)

    assert set(files) == {"SKILL.md", "scripts/compact_json.py"}
    rendered = b"\n".join(files.values())
    assert b"pyproject.toml" not in rendered
    assert b"rlm.skill" not in rendered
    assert b"Prime" not in rendered


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (_reply(name="compact_json"), "single hyphens"),
        (
            _reply(
                skill_md=VALID_SKILL_MD.replace(
                    "description:", "allowed-tools: Bash\ndescription:"
                )
            ),
            "non-portable frontmatter",
        ),
        (_reply(files={"../escape.py": VALID_SCRIPT}), "relative path under"),
        (
            _reply(files={"pyproject.toml": "[project]\nname='vendor-package'\n"}),
            "relative path under",
        ),
        (
            _reply(files={"references/guide.md": "Read me.\n"}),
            "at least one file under scripts",
        ),
        (
            _reply(files={"scripts/run.py": "import subprocess\n"}),
            "forbidden module",
        ),
        (
            _reply(files={"scripts/run.py": "def broken(:\n"}),
            "invalid Python",
        ),
    ],
)
def test_candidate_rejects_non_portable_or_unsafe_packages(reply, message):
    with pytest.raises(SkillPackageError, match=message):
        parse_candidate(reply)


def test_author_sees_past_examples_but_not_the_held_out_task():
    authored = author_task(_task())

    assert "past-answer" in authored.data.prompt
    assert "held-out-answer" not in authored.data.prompt
    assert "held-out-private-input" not in authored.data.prompt
    assert "Prime" not in authored.data.prompt
    assert "Anthropic" not in authored.data.prompt
    assert "OpenAI" not in authored.data.prompt
    assert '"name":"example-skill"' in authored.data.prompt
    assert "---\\nname: example-skill\\ndescription:" in authored.data.prompt
    assert "task file paths as command-line arguments" in authored.data.prompt
    assert authored.data.source_task_name == "held-out-task"


def test_consumer_discovers_the_skill_without_changing_the_task():
    task = _task()
    adapted = consumer_task(task, "/task/agent-skills/compact-json")

    assert adapted.data.prompt.endswith(task.data.prompt)
    assert "/task/agent-skills/compact-json/SKILL.md" in adapted.data.prompt
    assert adapted.data.answer == task.data.answer
    assert "Keep the answer exact." in adapted.data.system_prompt
    assert "/task/agent-skills/compact-json/SKILL.md" in adapted.data.system_prompt
    assert "must inspect SKILL.md before solving" in adapted.data.system_prompt
    assert "never the task working directory" in adapted.data.system_prompt
    assert "next assistant turn must be a genuine IPython tool call" in (
        adapted.data.system_prompt
    )
    assert adapted.data is not task.data


def test_taskset_carries_disjoint_bootstrap_evidence():
    tasks = ReasoningOffloadSkillAcquisitionTaskset(
        ReasoningOffloadSkillAcquisitionConfig(
            split="validation", families=("ledger",), experience_examples=2
        )
    ).load()

    assert len(tasks) == 3
    assert all(len(task.data.experiences) == 2 for task in tasks)
    assert all(
        experience.files != task.data.files
        for task in tasks
        for experience in task.data.experiences
    )


def test_taskset_builds_a_hidden_same_family_utility_panel():
    tasks = ReasoningOffloadSkillAcquisitionTaskset(
        ReasoningOffloadSkillAcquisitionConfig(
            split="validation",
            families=("ledger",),
            experience_examples=2,
            utility_panel_size=3,
        )
    ).load()

    assert len(tasks) == 3
    assert all(len(task.data.utility_examples) == 3 for task in tasks)
    for task in tasks:
        authored = author_task(task)
        for utility_example in task.data.utility_examples:
            assert utility_example.answer not in authored.data.prompt
            assert all(
                utility_example.files != experience.files
                for experience in task.data.experiences
            )


def test_taskset_can_limit_author_prompts_without_shrinking_utility_panel():
    tasks = ReasoningOffloadSkillAcquisitionTaskset(
        ReasoningOffloadSkillAcquisitionConfig(
            split="validation",
            families=("ledger", "frontier"),
            instances_per_template=2,
            variants_per_family=1,
            utility_panel_size=3,
        )
    ).load()

    assert len(tasks) == 4
    assert [task.data.family for task in tasks] == [
        "ledger",
        "ledger",
        "frontier",
        "frontier",
    ]
    assert all(task.data.template_variant == 6 for task in tasks)
    assert all(len(task.data.utility_examples) == 3 for task in tasks)
    assert all(
        {example.template_variant for example in task.data.utility_examples}
        == {6, 7, 8}
        for task in tasks
    )


class RecordingRuntime:
    def __init__(self):
        self.files = {}

    async def write(self, path, content):
        self.files[path] = content


@pytest.mark.asyncio
async def test_candidate_is_materialized_under_a_vendor_neutral_path():
    runtime = RecordingRuntime()
    candidate = parse_candidate(_reply())

    root = await install_candidate(runtime, candidate)

    assert root == "/task/agent-skills/compact-json"
    assert set(runtime.files) == {
        "/task/agent-skills/compact-json/SKILL.md",
        "/task/agent-skills/compact-json/scripts/compact_json.py",
    }


@pytest.mark.asyncio
async def test_author_reward_is_marginal_downstream_utility():
    author = _trace("author")
    author.info["skill_name"] = "compact-json"
    skill_user = _trace("skill_user", score=1.0)
    baseline = _trace("baseline_user", score=0.0)
    episode = vf.Episode(traces=[author, skill_user, baseline])
    env = object.__new__(ReasoningOffloadSkillLearningEnv)

    await env.finalize(_task(), episode)

    assert author.rewards["skill_utility"].score == 1.0
    assert author.metrics["skill_package_valid"] == 1.0
    assert author.metrics["skill_consulted"] == 0.0
    assert "feedback" not in author.info


@pytest.mark.asyncio
async def test_author_reward_averages_marginal_utility_across_hidden_panel():
    author = _trace("author")
    author.info["skill_name"] = "compact-json"
    skill_users = [_trace("skill_user", score=1.0), _trace("skill_user", score=0.0)]
    baselines = [_trace("baseline_user", score=0.0), _trace("baseline_user", score=1.0)]
    for index, trace in enumerate(skill_users):
        trace.info["utility_case_index"] = index
    for index, trace in enumerate(baselines):
        trace.info["utility_case_index"] = index
    episode = vf.Episode(traces=[author, *skill_users, *baselines])
    env = object.__new__(ReasoningOffloadSkillLearningEnv)

    await env.finalize(_task(), episode)

    assert author.rewards["skill_utility"].score == 0.0
    assert author.metrics["utility_panel_size"] == 2.0
    assert author.metrics["baseline_correct"] == 0.5
    assert author.metrics["skill_user_correct"] == 0.5
    assert author.metrics["utility_evaluation_complete"] == 1.0


@pytest.mark.asyncio
async def test_incomplete_baseline_suppresses_candidate_reward():
    author = _trace("author")
    author.info["skill_name"] = "compact-json"
    skill_user = _trace("skill_user", score=1.0)
    baseline = _trace("baseline_user", score=0.0)
    baseline.info["utility_arm_errors"] = [
        {"type": "HarnessError", "message": "agent timeout"}
    ]
    episode = vf.Episode(traces=[author, skill_user, baseline])
    env = object.__new__(ReasoningOffloadSkillLearningEnv)

    await env.finalize(_task(), episode)

    assert author.rewards["skill_utility"].score == 0.0
    assert author.metrics["utility_evaluation_complete"] == 0.0
    assert "received no learning signal" in author.info["feedback"]


@pytest.mark.asyncio
async def test_redundant_unconsulted_skill_gets_no_utility_reward():
    author = _trace("author")
    author.info["skill_name"] = "compact-json"
    skill_user = _trace("skill_user", score=1.0)
    baseline = _trace("baseline_user", score=1.0)
    episode = vf.Episode(traces=[author, skill_user, baseline])
    env = object.__new__(ReasoningOffloadSkillLearningEnv)

    await env.finalize(_task(), episode)

    assert author.rewards["skill_utility"].score == 0.0
    assert "did not consult" in author.info["feedback"]
