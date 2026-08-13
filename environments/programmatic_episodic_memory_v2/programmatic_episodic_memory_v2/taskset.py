"""Executable programmatic episodic-memory tasks backed by frozen JSONL splits."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage, content_text

Split = Literal["train", "familiar_heldout", "semantic_ood"]
RewardShape = Literal["strict", "dense"]

DEMONSTRATION_TEMPLATE = (
    "Here is an example of an expert response:\n"
    "<demonstration>\n{demonstration}\n</demonstration>"
)

_ERROR_PATTERN = re.compile(
    r"(?:Traceback \(most recent call last\)|\b(?:NameError|TypeError|KeyError|"
    r"FileNotFoundError|SyntaxError|ImportError|ModuleNotFoundError)\b)",
)


class ProgrammaticEpisodicMemoryData(vf.TaskData):
    split: Split
    family: str
    domain: str
    instruction_level: str
    history_path: str
    history_format: str
    requires_history: bool
    uses_ipython: bool
    retrieval_policy: str
    expected_answers: tuple[str, ...]
    demonstration: str
    followup_prompts: tuple[str, ...] = ()
    files: dict[str, str] = Field(default_factory=dict)


class ProgrammaticEpisodicMemoryTaskConfig(vf.TaskConfig):
    reward_shape: RewardShape = "strict"


def _ipython_calls(trace: vf.Trace) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    for node in trace.nodes:
        message = node.message
        if not isinstance(message, AssistantMessage) or not node.sampled:
            continue
        for call in message.tool_calls or []:
            if call.name != "ipython":
                continue
            try:
                arguments = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError):
                calls.append((call.id, ""))
                continue
            calls.append((call.id, arguments.get("code", "")))
    return calls


def _tool_outputs(trace: vf.Trace) -> dict[str, str]:
    return {
        node.message.tool_call_id: content_text(node.message.content)
        for node in trace.nodes
        if isinstance(node.message, ToolMessage)
    }


def _text_answers(trace: vf.Trace) -> list[str]:
    answers: list[str] = []
    for node in trace.nodes:
        message = node.message
        if not isinstance(message, AssistantMessage) or not node.sampled:
            continue
        text = content_text(message.content).strip()
        if text:
            answers.append(text)
    return answers


def _assigned_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _loaded_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _behavior(
    trace: vf.Trace, data: ProgrammaticEpisodicMemoryData
) -> dict[str, float]:
    calls = _ipython_calls(trace)
    outputs = _tool_outputs(trace)
    codes = [source for _, source in calls]
    answers = _text_answers(trace)
    expected = list(data.expected_answers)
    observed = answers[-len(expected) :] if expected else []
    answer_correct = float(observed == expected)

    history_reads = [data.history_path in source for source in codes]
    touched_history = any(history_reads)
    retrieval_decision = float(touched_history == data.requires_history)
    grounded_answer = float(
        answer_correct == 1.0 and (not data.requires_history or touched_history)
    )

    valid_calls = bool(calls) == data.uses_ipython and all(
        source.strip() for source in codes
    )
    unresolved_errors = (
        any(_ERROR_PATTERN.search(outputs.get(call_id, "")) for call_id, _ in calls)
        and answer_correct == 0.0
    )
    valid_tool_behavior = float(valid_calls and not unresolved_errors)
    observation_chars = sum(len(outputs.get(call_id, "")) for call_id, _ in calls)
    bounded_retrieval = float(not data.requires_history or observation_chars <= 4096)
    no_repeat = float(
        len(codes) == len(dict.fromkeys(source.strip() for source in codes))
    )

    persistent_required = data.family == "repeated_lookup_index"
    persistent_reuse = True
    if persistent_required:
        first_history = next(
            (index for index, read in enumerate(history_reads) if read), None
        )
        persistent_reuse = False
        if first_history is not None and first_history + 1 < len(codes):
            retained = _assigned_names(codes[first_history])
            persistent_reuse = any(
                not history_reads[index]
                and bool(retained & _loaded_names(codes[index]))
                for index in range(first_history + 1, len(codes))
            )

    stale_required = data.family == "stale_note_override"
    stale_resolution = not stale_required or grounded_answer == 1.0
    reset_required = data.instruction_level == "context_reset"
    context_reset_recovery = not reset_required or grounded_answer == 1.0
    override_required = data.family == "prompt_override_control"
    current_turn_override = not override_required or (
        answer_correct == 1.0 and not touched_history
    )

    expected_calls = max(
        1 if data.uses_ipython else 0,
        sum(
            1
            for marker in data.demonstration.splitlines()
            if marker == "assistant tool: ipython"
        ),
    )
    efficient = float(len(calls) <= expected_calls + 1)

    required_components = (
        answer_correct == 1.0,
        retrieval_decision == 1.0,
        grounded_answer == 1.0,
        valid_tool_behavior == 1.0,
        bounded_retrieval == 1.0,
        no_repeat == 1.0,
        persistent_reuse,
        stale_resolution,
        context_reset_recovery,
        current_turn_override,
    )
    strict_success = float(all(required_components))
    dense_core = sum(float(value) for value in required_components) / len(
        required_components
    )
    dense_reward = 0.9 * dense_core + 0.1 * efficient
    return {
        "strict_success": strict_success,
        "dense_reward": dense_reward,
        "answer_correct": answer_correct,
        "retrieval_decision": retrieval_decision,
        "grounded_answer": grounded_answer,
        "valid_tool_behavior": valid_tool_behavior,
        "bounded_retrieval": bounded_retrieval,
        "observation_chars": float(observation_chars),
        "no_repeated_cell": no_repeat,
        "persistent_index_required": float(persistent_required),
        "persistent_index_reuse": float(persistent_reuse),
        "stale_note_required": float(stale_required),
        "stale_note_resolution": float(stale_resolution),
        "context_reset_required": float(reset_required),
        "context_reset_recovery": float(context_reset_recovery),
        "current_turn_override_required": float(override_required),
        "current_turn_override": float(current_turn_override),
        "efficient_calls": efficient,
        "ipython_calls": float(len(calls)),
    }


class ProgrammaticEpisodicMemoryTask(
    vf.Task[
        ProgrammaticEpisodicMemoryData,
        vf.State,
        ProgrammaticEpisodicMemoryTaskConfig,
    ]
):
    NEEDS_CONTAINER = True

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        directories = sorted({str(Path(path).parent) for path in self.data.files})
        created = await runtime.run(["mkdir", "-p", *directories], {})
        if created.exit_code != 0:
            raise RuntimeError(
                f"memory workspace setup failed: {created.stderr[-500:]}"
            )
        for path, contents in self.data.files.items():
            await runtime.write(path, contents.encode())

    @vf.reward(weight=1.0)
    async def memory_reward(self, trace: vf.Trace) -> float:
        behavior = _behavior(trace, self.data)
        return behavior[
            f"{self.config.reward_shape}_success"
            if self.config.reward_shape == "strict"
            else "dense_reward"
        ]

    @vf.metric
    async def memory_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _behavior(trace, self.data)


class ProgrammaticEpisodicMemoryConfig(vf.TasksetConfig):
    task: ProgrammaticEpisodicMemoryTaskConfig = ProgrammaticEpisodicMemoryTaskConfig()
    dataset_path: str
    split: Split
    condition_on_demonstration: bool = False
    instance_offset: int = Field(0, ge=0)
    instances_per_family: int | None = Field(None, ge=1)
    offset: int = Field(0, ge=0)
    limit: int | None = Field(None, ge=1)
    families: tuple[str, ...] | None = None


def _demonstration(messages: list[dict]) -> str:
    first_user = next(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )
    lines: list[str] = []
    for message in messages[first_user + 1 :]:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                function = call["function"]
                lines.append(f"assistant tool: {function['name']}")
                lines.append(function["arguments"])
        elif role == "tool":
            lines.append(f"tool result: {message.get('content', '')}")
        elif role in {"assistant", "user"}:
            lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)


def _row_to_task(
    row: dict,
    *,
    idx: int,
    split: Split,
    condition_on_demonstration: bool,
    task_config: ProgrammaticEpisodicMemoryTaskConfig,
) -> ProgrammaticEpisodicMemoryTask:
    messages = json.loads(row["messages_json"])
    metadata = json.loads(row["metadata_json"])
    files = json.loads(row["workspace_files_json"])
    if metadata["split"] != split:
        raise ValueError(
            f"row split {metadata['split']!r} does not match configured split {split!r}"
        )
    if any("reasoning_content" in message for message in messages):
        raise ValueError("fabricated or embedded reasoning_content is not accepted")

    system = next(
        message["content"] for message in messages if message.get("role") == "system"
    )
    user_messages = [
        message["content"] for message in messages if message.get("role") == "user"
    ]
    expected_answers = tuple(
        str(message["content"]).strip()
        for message in messages
        if message.get("role") == "assistant" and message.get("content") is not None
    )
    demonstration = _demonstration(messages)
    if condition_on_demonstration:
        system = (
            f"{DEMONSTRATION_TEMPLATE.format(demonstration=demonstration)}\n\n{system}"
        )

    data = ProgrammaticEpisodicMemoryData(
        idx=idx,
        name=f"{split}-{metadata['family']}-{metadata['instance']}",
        prompt=user_messages[0],
        system_prompt=system,
        split=split,
        family=metadata["family"],
        domain=metadata["domain"],
        instruction_level=metadata["instruction_level"],
        history_path=metadata["history_path"],
        history_format=metadata["history_format"],
        requires_history=metadata["requires_history"],
        uses_ipython=metadata["uses_ipython"],
        retrieval_policy=metadata["retrieval_policy"],
        expected_answers=expected_answers,
        demonstration=demonstration,
        followup_prompts=tuple(user_messages[1:]),
        files=files,
    )
    return ProgrammaticEpisodicMemoryTask(data, task_config)


class ProgrammaticEpisodicMemoryEnv(vf.SingleAgentEnv):
    async def run(self, task, agents) -> None:
        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn()
            for prompt in task.data.followup_prompts:
                if segment.terminated:
                    break
                segment = await interaction.turn([UserMessage(content=prompt)])


class ProgrammaticEpisodicMemoryTaskset(
    vf.Taskset[ProgrammaticEpisodicMemoryTask, ProgrammaticEpisodicMemoryConfig]
):
    def load(self) -> list[ProgrammaticEpisodicMemoryTask]:
        path = Path(self.config.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"programmatic memory split not found: {path}")
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if self.config.instances_per_family is not None:
            start = self.config.instance_offset
            stop = start + self.config.instances_per_family
            rows = [
                row
                for row in rows
                if start <= json.loads(row["metadata_json"])["instance"] < stop
            ]
        if self.config.families is not None:
            allowed = set(self.config.families)
            rows = [
                row
                for row in rows
                if json.loads(row["metadata_json"])["family"] in allowed
            ]
        selected = rows[self.config.offset :]
        if self.config.limit is not None:
            selected = selected[: self.config.limit]
        return [
            _row_to_task(
                row,
                idx=index,
                split=self.config.split,
                condition_on_demonstration=self.config.condition_on_demonstration,
                task_config=self.config.task,
            )
            for index, row in enumerate(selected)
        ]
