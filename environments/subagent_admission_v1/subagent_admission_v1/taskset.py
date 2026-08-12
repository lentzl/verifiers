"""Spawn-first admission control layered over subagent-communication-v1."""

from __future__ import annotations

import ast
from typing import Literal

import verifiers.v1 as vf
from subagent_communication_v1.taskset import (
    WEIGHTED_CHECKSUM_FORMULA,
    SubagentCommunicationConfig,
    SubagentCommunicationData,
    SubagentCommunicationTask,
    SubagentCommunicationTaskset,
    _answer_score,
    _assigned_call_names,
    _branch_root,
    _call_name,
    _ipython_events,
    _keyword,
    _protocol_behavior,
    _spawn_prompt,
)


def _path_used_outside_spawn(tree: ast.AST, path: str) -> bool:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if path not in node.value:
            continue
        ancestor: ast.AST | None = node
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, ast.Call) and _call_name(ancestor) == "rlm":
                break
        else:
            return True
    return False


def _statement_index(tree: ast.AST, target: ast.AST) -> int | None:
    for index, statement in enumerate(getattr(tree, "body", [])):
        if any(node is target for node in ast.walk(statement)):
            return index
    return None


def _local_sum(statement: ast.AST) -> bool:
    return "local" in ast.unparse(statement) and any(
        isinstance(node, ast.Call) and _call_name(node) == "sum" for node in ast.walk(statement)
    )


def _admission_behavior(
    trace: vf.Trace,
    data: SubagentCommunicationData,
    reward_shape: Literal["strict", "dense"] = "strict",
) -> dict[str, float]:
    empty = {
        "admission_score": 0.0,
        "spawn_first_cell": 0.0,
        "spawn_precedes_local": 0.0,
        "retained_admission_handle": 0.0,
        "exact_admission_payload": 0.0,
        "self_contained_formula": 0.0,
        "parent_remote_read": 0.0,
        "parent_remote_read_before_admission": 0.0,
        "local_work_after_admission": 0.0,
        "admission_aligned": 0.0,
    }
    if data.family != "single":
        return empty

    events = _ipython_events(trace)
    if not events:
        return empty
    coordinator_root = _branch_root(trace, 0)
    branch_aware = any(_branch_root(trace, event.node_index) != coordinator_root for event in events)
    coordinator_events = [
        event for event in events if not branch_aware or _branch_root(trace, event.node_index) == coordinator_root
    ]
    if not coordinator_events:
        return empty

    parsed_events: list[tuple[int, ast.Module]] = []
    for event_index, event in enumerate(coordinator_events):
        try:
            parsed_events.append((event_index, ast.parse(event.code)))
        except SyntaxError:
            continue
    if not parsed_events:
        return empty

    spawn_locations = [
        (event_index, tree, node)
        for event_index, tree in parsed_events
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "rlm"
    ]
    if reward_shape == "strict":
        spawn_locations = [location for location in spawn_locations if location[0] == 0]
    single_spawn = len(spawn_locations) == 1
    spawn_event_index, spawn_tree, spawn_call = spawn_locations[0] if single_spawn else (-1, parsed_events[0][1], None)
    spawn_first_cell = single_spawn and spawn_event_index == 0
    assigned = _assigned_call_names(spawn_tree)
    retained = bool(single_spawn and spawn_call is not None and id(spawn_call) in assigned)
    path = data.child_paths["shard-worker"]
    prompt = _spawn_prompt(spawn_call) if spawn_call is not None else None
    exact_payload = bool(
        single_spawn
        and spawn_call is not None
        and _keyword(spawn_call, "name") == "shard-worker"
        and prompt
        and path in prompt
        and "agent_message" in prompt
        and "parent" in prompt
    )
    formula_bound = bool(
        exact_payload and prompt and ("Delegation contract:" not in data.prompt or WEIGHTED_CHECKSUM_FORMULA in prompt)
    )
    remote_read = any(_path_used_outside_spawn(tree, path) for _, tree in parsed_events)

    spawn_index = _statement_index(spawn_tree, spawn_call) if spawn_call is not None else None
    statements = getattr(spawn_tree, "body", [])
    local_before = any(
        _local_sum(statement)
        for event_index, tree in parsed_events
        if event_index < spawn_event_index
        for statement in getattr(tree, "body", [])
    ) or bool(spawn_index is not None and any(_local_sum(statement) for statement in statements[:spawn_index]))
    spawn_precedes_local = bool(single_spawn and spawn_index is not None and not local_before)
    local_after = bool(
        spawn_index is not None and any(_local_sum(statement) for statement in statements[spawn_index + 1 :])
    )
    if not local_after and single_spawn:
        for event_index, tree in parsed_events:
            if (
                event_index > spawn_event_index
                and (reward_shape == "dense" or not _path_used_outside_spawn(tree, path))
                and any(_local_sum(statement) for statement in getattr(tree, "body", []))
            ):
                local_after = True
                break

    components = (
        spawn_first_cell,
        spawn_precedes_local,
        retained,
        exact_payload,
        single_spawn and not remote_read,
        local_after,
    )
    aligned = all(components)
    return {
        "admission_score": sum(components) / len(components),
        "spawn_first_cell": float(spawn_first_cell),
        "spawn_precedes_local": float(spawn_precedes_local),
        "retained_admission_handle": float(retained),
        "exact_admission_payload": float(exact_payload),
        "self_contained_formula": float(formula_bound),
        "parent_remote_read": float(remote_read),
        # Keep the legacy key readable in historical result summaries.
        "parent_remote_read_before_admission": float(remote_read),
        "local_work_after_admission": float(local_after),
        "admission_aligned": float(aligned),
    }


def _causal_behavior(
    trace: vf.Trace,
    data: SubagentCommunicationData,
) -> dict[str, float]:
    admission = _admission_behavior(trace, data, reward_shape="strict")
    protocol = _protocol_behavior(
        trace,
        data.family,
        data.expected_children,
        data.child_paths,
        data.followup_secret,
    )
    answer_accuracy = _answer_score(trace.last_reply, data.answer)

    spawn_first = bool(admission["spawn_first_cell"])
    contract_bound = bool(
        spawn_first
        and admission["spawn_precedes_local"]
        and admission["retained_admission_handle"]
        and admission["exact_admission_payload"]
        and not admission["parent_remote_read"]
    )
    local_work = bool(contract_bound and admission["local_work_after_admission"])
    child_reply = bool(local_work and protocol["messages_to_parent"] >= 1)
    completed = bool(child_reply and protocol["protocol_aligned"] and answer_accuracy == 1.0)
    clean_child_reply = bool(
        child_reply
        and protocol["explicit_messages_to_parent"] >= len(data.expected_children)
        and protocol["roster_calls"] == 0
        and protocol["observation_calls"] == 0
        and protocol["failed_cells"] == 0
        and protocol.get("non_ipython_tool_calls", 0) == 0
        and protocol.get("inert_cells", 0) == 0
        and protocol.get("duplicate_cells", 0) == 0
        and protocol.get("post_parent_send_tool_calls", 0) == 0
        and protocol.get("post_fan_in_cells", 0) == 0
        and admission["self_contained_formula"] == 1.0
    )
    clean_completed = bool(clean_child_reply and protocol["clean_protocol_aligned"] and answer_accuracy == 1.0)
    stages = (spawn_first, contract_bound, local_work, child_reply, completed)
    return {
        "causal_score": sum(stages) / len(stages),
        "causal_spawn_first": float(spawn_first),
        "causal_contract_bound": float(contract_bound),
        "causal_local_work": float(local_work),
        "causal_child_reply": float(child_reply),
        "causal_completed": float(completed),
        "causal_clean_child_reply": float(clean_child_reply),
        "causal_clean_completed": float(clean_completed),
    }


def _clean_causal_behavior(
    trace: vf.Trace,
    data: SubagentCommunicationData,
) -> dict[str, float]:
    admission = _admission_behavior(trace, data, reward_shape="strict")
    protocol = _protocol_behavior(
        trace,
        data.family,
        data.expected_children,
        data.child_paths,
        data.followup_secret,
    )
    answer_accuracy = _answer_score(trace.last_reply, data.answer)

    raw_spawn_first = bool(admission["spawn_first_cell"])
    raw_contract_bound = bool(
        raw_spawn_first
        and admission["spawn_precedes_local"]
        and admission["retained_admission_handle"]
        and admission["exact_admission_payload"]
        and admission["self_contained_formula"]
        and not admission["parent_remote_read"]
    )
    raw_local_work = bool(raw_contract_bound and admission["local_work_after_admission"])
    trace_admissible = bool(
        admission["parent_remote_read"] == 0
        and protocol["roster_calls"] == 0
        and protocol["observation_calls"] == 0
        and protocol["failed_cells"] == 0
        and protocol.get("non_ipython_tool_calls", 0) == 0
        and protocol.get("inert_cells", 0) == 0
        and protocol.get("duplicate_cells", 0) == 0
        and protocol.get("post_parent_send_tool_calls", 0) == 0
        and protocol.get("post_fan_in_cells", 0) == 0
    )
    spawn_first = bool(trace_admissible and raw_spawn_first)
    contract_bound = bool(trace_admissible and raw_contract_bound)
    local_work = bool(trace_admissible and raw_local_work)
    tool_discipline = local_work
    child_reply = bool(tool_discipline and protocol["explicit_messages_to_parent"] >= len(data.expected_children))
    completed = bool(child_reply and protocol["clean_protocol_aligned"] and answer_accuracy == 1.0)
    stages = (spawn_first, contract_bound, local_work, tool_discipline, child_reply, completed)
    return {
        "clean_causal_score": sum(stages) / len(stages),
        "clean_causal_trace_admissible": float(trace_admissible),
        "clean_causal_raw_spawn_first": float(raw_spawn_first),
        "clean_causal_raw_contract_bound": float(raw_contract_bound),
        "clean_causal_raw_local_work": float(raw_local_work),
        "clean_causal_spawn_first": float(spawn_first),
        "clean_causal_contract_bound": float(contract_bound),
        "clean_causal_local_work": float(local_work),
        "clean_causal_tool_discipline": float(tool_discipline),
        "clean_causal_child_reply": float(child_reply),
        "clean_causal_completed": float(completed),
    }


class SubagentAdmissionTaskConfig(vf.TaskConfig):
    reward_mode: Literal["admission", "causal", "clean_causal", "mixed"] = "admission"
    reward_shape: Literal["strict", "dense"] = "strict"


class SubagentAdmissionTask(SubagentCommunicationTask):
    def _include_base_rewards(self) -> bool:
        return self.config.reward_mode == "mixed"

    @vf.reward(weight=1.0)
    async def protocol_gated_accuracy(self, trace: vf.Trace) -> float:
        if not self._include_base_rewards():
            return 0.0
        return await super().protocol_gated_accuracy(trace)

    @vf.reward(weight=1.0)
    async def delegation_protocol(self, trace: vf.Trace) -> float:
        if not self._include_base_rewards():
            return 0.0
        return await super().delegation_protocol(trace)

    @vf.reward(weight=1.0)
    async def stateful_control_progress(self, trace: vf.Trace) -> float:
        if not self._include_base_rewards():
            return 0.0
        return await super().stateful_control_progress(trace)

    @vf.reward(weight=1.0)
    async def post_fan_in_control_reward(self, trace: vf.Trace) -> float:
        if not self._include_base_rewards():
            return 0.0
        return await super().post_fan_in_control_reward(trace)

    @vf.reward(weight=1.0)
    async def admission_control(self, trace: vf.Trace) -> float:
        if self.config.reward_mode in {"causal", "clean_causal"}:
            return 0.0
        return _admission_behavior(
            trace,
            self.data,
            self.config.reward_shape,
        )["admission_score"]

    @vf.reward(weight=1.0)
    async def causal_control(self, trace: vf.Trace) -> float:
        if self.config.reward_mode == "clean_causal":
            return _clean_causal_behavior(trace, self.data)["clean_causal_score"]
        if self.config.reward_mode != "causal":
            return 0.0
        return _causal_behavior(trace, self.data)["causal_score"]

    @vf.metric
    async def admission_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _admission_behavior(trace, self.data, self.config.reward_shape)

    @vf.metric
    async def causal_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _causal_behavior(trace, self.data)

    @vf.metric
    async def clean_causal_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _clean_causal_behavior(trace, self.data)


class SubagentAdmissionConfig(SubagentCommunicationConfig):
    task: SubagentAdmissionTaskConfig = SubagentAdmissionTaskConfig()
    self_contained_child_contract: bool = False


def _self_contained_prompt(prompt: str) -> str:
    formula = "sum((index + 1) * value for index, value in enumerate(values))"
    prompt = prompt.replace(
        "compute its weighted checksum",
        f"compute its weighted checksum using exactly {formula}",
    )
    return (
        f"{prompt}\n\nDelegation contract: the child does not inherit this parent question. "
        "Its spawn prompt must explicitly define the weighted checksum as "
        f"{formula}."
    )


def _self_contained_data(data: SubagentCommunicationData) -> SubagentCommunicationData:
    prompt = _self_contained_prompt(data.prompt)
    updates: dict[str, object] = {"prompt": prompt}
    for field in (
        "demonstrations",
        "turn_demonstrations",
        "child_request_demonstrations",
        "coordinator_demonstrations",
    ):
        mapping = getattr(data, field)
        if mapping is None:
            continue
        if data.prompt not in mapping:
            raise ValueError(f"{field} is missing the parent prompt")
        updates[field] = {
            prompt if question == data.prompt else question: demonstration
            for question, demonstration in mapping.items()
        }
    return data.model_copy(update=updates)


class SubagentAdmissionTaskset(vf.Taskset[SubagentAdmissionTask, SubagentAdmissionConfig]):
    def load(self) -> list[SubagentAdmissionTask]:
        base_config = SubagentCommunicationConfig.model_validate(
            self.config.model_dump(exclude={"task", "self_contained_child_contract"})
        )
        tasks = SubagentCommunicationTaskset(base_config).load()
        data = [task.data for task in tasks]
        if self.config.self_contained_child_contract:
            data = [_self_contained_data(item) if item.expected_children else item for item in data]
        return [SubagentAdmissionTask(item, self.config.task) for item in data]
