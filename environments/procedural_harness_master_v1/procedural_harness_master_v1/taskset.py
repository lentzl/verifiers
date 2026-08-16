"""Executable procedural benchmark for end-to-end Prime Agent coordination."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import Field
from subagent_communication_v1.taskset import (
    _assigned_call_names,
    _branch_root,
    _call_name,
    _completion_gate_source,
    _delegated_path_used_outside_spawn,
    _failed,
    _ipython_events,
    _message_sent,
    _spawn_name,
    _spawn_prompt,
)

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, UserMessage, content_text

Split = Literal["train_gen", "valid_gen", "ood_gen"]
COMPLETION_GATE_PATH = "/workspace/.procedural-harness-master/completion_gate.py"
COMPLETION_GATE_FEEDBACK = (
    "completion gate: the end-to-end coordinator task is not complete. Preserve successful "
    "state and existing child handles. React only to visible child messages; do not poll, "
    "invent a result, or spawn replacement children. If a visible failure explicitly transfers "
    "resource ownership, continue from that transition. Otherwise end the next waiting turn "
    "briefly so this active autonomous run can accept the child reply. Return bare strict JSON "
    "only after all required child evidence has arrived."
)


@lru_cache(maxsize=1)
def _generator() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[3]
        / "datasets"
        / "procedural_harness_master_v1"
        / "generate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_procedural_harness_master_v1_generator", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load procedural generator from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProceduralHarnessMasterData(vf.TaskData):
    episode_id: str
    split: Split
    family: str
    workspace_files: dict[str, str] = Field(default_factory=dict)
    oracle: dict[str, Any]
    generation_metadata: dict[str, Any] = Field(default_factory=dict)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _keyword(call: ast.Call, name: str) -> Any:
    value = next((item.value for item in call.keywords if item.arg == name), None)
    return _literal(value) if value is not None else None


def _assigned_state(statement: ast.stmt, name: str, value: Any) -> bool:
    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        assigned_value = _literal(statement.value)
        assigned = assigned_value == value or str(assigned_value) == str(value)
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and assigned:
                return True
            if (
                isinstance(target, ast.Subscript)
                and _literal(target.slice) == name
                and assigned
            ):
                return True
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "os"
                and target.value.attr == "environ"
                and str(_literal(target.slice)).lower() == name.lower()
                and assigned
            ):
                return True
        if isinstance(statement.value, ast.Dict):
            pairs = zip(statement.value.keys, statement.value.values, strict=True)
            return any(_literal(key) == name and _literal(item) == value for key, item in pairs)
    return False


def _incoming_messages(trace: vf.Trace) -> list[tuple[int, str, str]]:
    messages = []
    for index, node in enumerate(trace.nodes):
        if not isinstance(node.message, UserMessage):
            continue
        text = content_text(node.message.content)
        match = re.match(r"\[from child:([^\]]+)\]", text)
        if match:
            messages.append((index, match.group(1), text.rsplit("\n\n", 1)[-1].strip()))
    return messages


def _is_request_message(body: str) -> bool:
    lowered = body.lower()
    return lowered.startswith("need ") or bool(
        re.search(
            r"\b(?:please\s+provide|could\s+you\s+provide|send\s+me|share|supply|requesting)\b"
            r".{0,80}\bmultiplier\b",
            lowered,
        )
    )


def _is_awaited(statement: ast.stmt, call: ast.Call) -> bool:
    return any(
        isinstance(node, ast.Await) and node.value is call for node in ast.walk(statement)
    )


def _final_node(trace: vf.Trace) -> int:
    return next(
        (
            index
            for index in range(len(trace.nodes) - 1, -1, -1)
            if isinstance(trace.nodes[index].message, AssistantMessage)
            and trace.nodes[index].sampled
            and content_text(trace.nodes[index].message.content).strip()
        ),
        len(trace.nodes),
    )


def _answer_exact(reply: str, expected: Any) -> bool:
    try:
        return json.loads(reply.strip()) == expected
    except (AttributeError, json.JSONDecodeError):
        return False


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, float):
        return "float"
    if isinstance(value, int):
        return "int"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "str"
    raise TypeError(f"unsupported JSON answer type: {type(value).__name__}")


def _contract_behavior(trace: vf.Trace, data: ProceduralHarnessMasterData) -> dict[str, float]:
    oracle = data.oracle
    contract = oracle["trajectory_contract"]
    ownership = oracle["resource_ownership"]
    children = oracle["children"]
    expected_names = [child["name"] for child in children]
    child_paths = {
        path for path, item in ownership.items() if str(item["owner"]).startswith("child:")
    }
    local_paths = {
        path for path, item in ownership.items() if item["owner"] == "coordinator"
    }
    state = oracle.get("coordinator_state", {})

    events = _ipython_events(trace)
    coordinator_root = _branch_root(trace, 0) if trace.nodes else -1
    branch_aware = any(_branch_root(trace, event.node_index) != coordinator_root for event in events)
    coordinator_events = [
        event
        for event in events
        if not branch_aware or _branch_root(trace, event.node_index) == coordinator_root
    ]
    observed: dict[str, list[float]] = {}
    counts: Counter[str] = Counter()

    def mark(atom: str, position: float) -> None:
        observed.setdefault(atom, []).append(position)

    spawn_records: list[tuple[str | None, str, float, bool]] = []
    poll_positions: list[float] = []
    child_access_positions: list[float] = []
    local_access_positions: list[tuple[str, float]] = []
    parent_sends: list[tuple[float, str | None]] = []

    for event in coordinator_events:
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned_calls = _assigned_call_names(tree)
        for statement_index, statement in enumerate(tree.body):
            position = event.node_index + statement_index / 1000
            statement_source = ast.unparse(statement)
            for name, value in state.items():
                if _assigned_state(statement, name, value):
                    mark("retain_state", position)
            for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
                call_name = _call_name(call) or ""
                if call_name == "rlm":
                    child_name = _spawn_name(call, event.output)
                    prompt = _spawn_prompt(call) or ""
                    successful = not _failed(event.output)
                    spawn_records.append((child_name, prompt, position, successful))
                    if successful:
                        counts["spawn_child"] += 1
                        mark("spawn_child", position)
                        if child_name:
                            mark(f"spawn:{child_name}", position)
                        if id(call) in assigned_calls:
                            mark("retain_handle", position)
                    if any(path in prompt for path in local_paths):
                        mark("delegate_coordinator_owned", position)
                    if any(name in prompt for name in state):
                        mark("delegate_coordinator_state", position)
                    continue
                if call_name.startswith("agent_observe.") or call_name == "agent_observe":
                    poll_positions.append(position)
                    mark("poll", position)
                if call_name in {
                    "rlm.list_subagents",
                    "agent_message.list_agents",
                    "agent_message.recv",
                    "agent_message.list_messages",
                }:
                    poll_positions.append(position)
                    mark("poll", position)
                    mark("discover_child", position)
                if call_name == "sleep" or call_name.endswith(".sleep"):
                    poll_positions.append(position)
                    mark("poll", position)
                send_succeeded = _message_sent(event.output) or (
                    _is_awaited(statement, call) and not _failed(event.output)
                )
                if call_name == "agent_message.send" and send_succeeded:
                    receiver = _keyword(call, "receiver_name")
                    parent_sends.append((position, receiver if isinstance(receiver, str) else None))
                    counts["parent_to_child_message"] += 1
            for path in child_paths:
                if _delegated_path_used_outside_spawn(statement_source, path):
                    child_access_positions.append(position)
            for path in local_paths:
                if _delegated_path_used_outside_spawn(statement_source, path):
                    local_access_positions.append((path, position))

    for _, position in local_access_positions:
        mark("coordinator_read_local", position)
    manifest_path = next((path for path in local_paths if "verification" in path), None)
    if manifest_path is not None:
        for path, position in local_access_positions:
            if path == manifest_path:
                mark("coordinator_read_verification_manifest", position)

    incoming = _incoming_messages(trace)
    failure_messages = [item for item in incoming if "RESOURCE_UNAVAILABLE" in item[2] or "RLM child failure" in item[2]]
    request_messages = [
        item
        for item in incoming
        if item not in failure_messages and _is_request_message(item[2])
    ]
    result_messages = [item for item in incoming if item not in failure_messages and item not in request_messages]
    counts["child_result_message"] = len(result_messages)
    counts["child_to_parent_message"] = len(request_messages) + len(result_messages)

    for position, name, _ in result_messages:
        mark(f"receive:{name}", float(position))
        mark(f"receive_result:{name}", float(position))
    for position, name, _ in request_messages:
        mark(f"receive_request:{name}", float(position))
    for position, name, _ in failure_messages:
        mark(f"receive_failure:{name}", float(position))

    successful_spawns = [item for item in spawn_records if item[3]]
    first_incoming = min((item[0] for item in incoming), default=None)
    all_expected_spawned = set(expected_names) <= {
        name for name, _, _, successful in successful_spawns if successful and name
    }
    if first_incoming is not None and successful_spawns and all_expected_spawned:
        last_spawn = max(item[2] for item in successful_spawns)
        if last_spawn < first_incoming and not any(last_spawn < pos < first_incoming for pos in poll_positions):
            mark("yield", first_incoming - 0.1)
    if first_incoming is not None and not all(position < first_incoming for _, _, position, ok in successful_spawns if ok):
        mark("serialized_fanout_wait", float(first_incoming))

    for send_position, receiver in parent_sends:
        prior_requests = [item for item in request_messages if item[0] < send_position]
        target = receiver or (prior_requests[-1][1] if prior_requests else (expected_names[0] if len(expected_names) == 1 else None))
        if target:
            mark(f"send_followup:{target}", send_position)
        if not prior_requests:
            mark("guess_followup_value", send_position)
        later_results = [item for item in result_messages if item[0] > send_position]
        if later_results and not any(send_position < pos < later_results[0][0] for pos in poll_positions):
            mark("yield_after_followup", later_results[0][0] - 0.1)

    first_failure = min((item[0] for item in failure_messages), default=None)
    if child_access_positions:
        for position in child_access_positions:
            if first_failure is None or position < first_failure:
                mark("coordinator_read_child_owned", position)
                mark("coordinator_read_child_owned_before_reclaim", position)
                mark("reclaim_without_failure", position)
            else:
                mark("explicit_reclaim", first_failure + 0.1)
                mark("coordinator_read_after_reclaim", position)
                counts["reclaim"] += 1
    child_spawn_counts = Counter(name for name, _, _, ok in successful_spawns if ok and name)
    if first_failure is not None and any(count > 1 for count in child_spawn_counts.values()):
        mark("respawn_same_failed_child", float(first_failure) + 0.2)

    if manifest_path is not None and result_messages:
        try:
            digest = json.loads(data.workspace_files[manifest_path])["expected_digest"]
        except (KeyError, TypeError, json.JSONDecodeError):
            digest = None
        matching = next((item for item in result_messages if digest and digest in item[2]), None)
        if matching is not None:
            mark("verify_child_digest", matching[0] + 0.1)

    final_position = float(_final_node(trace))
    if (trace.last_reply or "").strip():
        mark("final_answer", final_position)
    final_exact = _answer_exact(trace.last_reply, oracle["final_answer"])
    if final_exact and "verify_child_digest" not in observed and oracle["expected_route"] == "single_verify":
        mark("accept_unverified_child_result", final_position)

    required = contract["required_atoms"]
    forbidden = contract["forbidden_atoms"]
    missing = [atom for atom in required if atom not in observed]
    violations = [atom for atom in forbidden if atom in observed]
    ordering_failures = []
    for edge in contract["ordering"]:
        before, after = edge["before"], edge["after"]
        if before not in observed or after not in observed or min(observed[before]) >= max(observed[after]):
            ordering_failures.append(f"{before}->{after}")
    cardinality_failures = [
        key
        for key, expected in contract["cardinality"].items()
        if counts[key] != expected
    ]
    required_fraction = (len(required) - len(missing)) / len(required) if required else 1.0
    ordering_fraction = (
        (len(contract["ordering"]) - len(ordering_failures)) / len(contract["ordering"])
        if contract["ordering"]
        else 1.0
    )
    cardinality_fraction = (
        (len(contract["cardinality"]) - len(cardinality_failures))
        / len(contract["cardinality"])
        if contract["cardinality"]
        else 1.0
    )
    bootstrap_progress = (
        float(final_exact)
        * float(not violations)
        * required_fraction
        * ordering_fraction
        * cardinality_fraction
    )
    hard_gate = bool(
        final_exact
        and not missing
        and not violations
        and not ordering_failures
        and not cardinality_failures
    )
    return {
        "harness_score": float(hard_gate),
        "final_answer_exact": float(final_exact),
        "all_required_atoms": float(not missing),
        "no_forbidden_atoms": float(not violations),
        "ordering_satisfied": float(not ordering_failures),
        "cardinality_exact": float(not cardinality_failures),
        "required_atoms_fraction": required_fraction,
        "ordering_fraction": ordering_fraction,
        "cardinality_fraction": cardinality_fraction,
        "bootstrap_progress": bootstrap_progress,
        "missing_required_atoms": float(len(missing)),
        "forbidden_atom_violations": float(len(violations)),
        "ordering_failures": float(len(ordering_failures)),
        "cardinality_failures": float(len(cardinality_failures)),
        "observed_atoms": float(len(observed)),
    }


class ProceduralHarnessMasterTaskConfig(vf.TaskConfig):
    reward_mode: Literal["hard", "bootstrap"] = "hard"


class ProceduralHarnessMasterTask(
    vf.Task[ProceduralHarnessMasterData, vf.State, ProceduralHarnessMasterTaskConfig]
):
    NEEDS_CONTAINER = True

    @property
    def key(self) -> str:
        return self.data.episode_id

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        deferred = set()
        if self.data.family == "reclaim":
            deferred = {child["resource_path"] for child in self.data.oracle["children"]}
        directories = sorted(
            {
                str(Path(COMPLETION_GATE_PATH).parent),
                *(str(Path(path).parent) for path in self.data.workspace_files),
            }
        )
        result = await runtime.run(["mkdir", "-p", *directories], {})
        if result.exit_code != 0:
            raise RuntimeError(f"procedural workspace setup failed: {result.stderr[-500:]}")
        for path, contents in self.data.workspace_files.items():
            if path not in deferred:
                await runtime.write(path, contents.encode())
        required_child_messages = {}
        for child in self.data.oracle["children"]:
            contract = child.get("message_contract")
            count = len(contract) if isinstance(contract, list) else 1
            required_child_messages[child["name"]] = count
        family = "followup" if self.data.family == "followup" else "direct"
        final_answer = self.data.oracle["final_answer"]
        await runtime.write(
            COMPLETION_GATE_PATH,
            _completion_gate_source(
                tuple(final_answer),
                family,
                expected_types={key: _json_type(value) for key, value in final_answer.items()},
                required_child_messages=required_child_messages,
                feedback=COMPLETION_GATE_FEEDBACK,
            ).encode(),
        )

    @vf.reward(weight=1.0)
    async def harness_score(self, trace: vf.Trace) -> float:
        behavior = _contract_behavior(trace, self.data)
        if self.config.reward_mode == "bootstrap":
            return behavior["harness_score"] + 0.1 * behavior["bootstrap_progress"]
        return behavior["harness_score"]

    @vf.metric
    async def harness_contract(self, trace: vf.Trace) -> dict[str, float]:
        return _contract_behavior(trace, self.data)


class ProceduralHarnessMasterConfig(vf.TasksetConfig):
    task: ProceduralHarnessMasterTaskConfig = ProceduralHarnessMasterTaskConfig()
    split: Split = "train_gen"
    count: int = Field(64, ge=1)
    start_index: int = Field(0, ge=0)
    master_seed: int = 20260816
    families: tuple[str, ...] | None = None


class ProceduralHarnessMasterTaskset(
    vf.Taskset[ProceduralHarnessMasterTask, ProceduralHarnessMasterConfig]
):
    def load(self) -> list[ProceduralHarnessMasterTask]:
        generator = _generator()
        tasks = []
        index = self.config.start_index
        while len(tasks) < self.config.count:
            row = generator.generate_episode(self.config.split, index, self.config.master_seed)
            index += 1
            family = row["metadata"]["episode_family"]
            if self.config.families is not None and family not in self.config.families:
                continue
            public = row["public"]
            tasks.append(
                ProceduralHarnessMasterTask(
                    ProceduralHarnessMasterData(
                        idx=len(tasks),
                        name=row["episode_id"],
                        prompt=public["user_prompt"],
                        system_prompt=public["system_prompt"],
                        episode_id=row["episode_id"],
                        split=row["split"],
                        family=family,
                        workspace_files=public["workspace_files"],
                        oracle=row["oracle"],
                        generation_metadata=row["metadata"],
                    ),
                    self.config.task,
                )
            )
        return tasks


class ProceduralHarnessMasterEnv(vf.SingleAgentEnv):
    """Inject the OOD reclaim transition without exposing hidden oracle values."""

    async def run(self, task, agents) -> None:
        if task.data.family != "reclaim":
            await agents.agent.run(task)
            return
        child = task.data.oracle["children"][0]
        path = child["resource_path"]
        fault = task.data.oracle["fault_plan"]
        async with (
            agents.agent.provision(task) as runtime,
            agents.agent.interaction(task, runtime=runtime) as interaction,
        ):
                first = await interaction.turn()
                if first.terminated:
                    return
                await runtime.write(path, task.data.workspace_files[path].encode())
                existing_failure = any(
                    "RESOURCE_UNAVAILABLE" in body
                    for _, _, body in _incoming_messages(interaction.trace)
                )
                if existing_failure:
                    message = (
                        f"[benchmark runtime]\nThe explicit failure transferred ownership of {path} "
                        "to the coordinator. Continue from retained state."
                    )
                else:
                    message = (
                        f"[from child:{child['name']}]\n"
                        "Agent-to-agent message received.\n"
                        "Source: benchmark fault injector\n"
                        f"Message id: fault-{task.data.episode_id}\n\n"
                        f"{fault['message']}"
                    )
                await interaction.turn(message)
