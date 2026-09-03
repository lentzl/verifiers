"""Native Prime Agent depth-one delegation and messaging tasks."""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage, content_text

Family = Literal[
    "direct",
    "single",
    "parallel",
    "handshake",
    "followup",
    "document_direct",
    "document_flat",
    "document_hierarchical",
    "document_free",
    "document_utility_direct",
    "document_utility_flat",
    "document_utility_hierarchical",
    "document_utility_depth3",
    "document_adaptive_d0",
    "document_adaptive_d1",
    "document_adaptive_d2",
    "document_adaptive_d3",
    "specialist_local",
    "specialist_generic",
    "specialist_table_join",
    "specialist_table_reconcile",
    "specialist_source_ast",
    "specialist_source_config",
    "specialist_recursive_table",
    "specialist_recursive_source",
]
InstructionLevel = Literal["standard", "guided"]
PromptContract = Literal["historical_v1", "explicit_bidirectional_v2"]
UtilityPolicyProfile = Literal[
    "historical_v1",
    "causal_matched_v2",
    "causal_action_boundary_v3",
    "causal_decision_boundary_v4",
]
WEIGHTED_CHECKSUM_FORMULA = "sum((index + 1) * value for index, value in enumerate(values))"

FAMILIES: tuple[Family, ...] = ("direct", "single", "parallel", "followup")
DOCUMENT_FAMILIES: tuple[Family, ...] = (
    "document_direct",
    "document_flat",
    "document_hierarchical",
    "document_free",
    "document_utility_direct",
    "document_utility_flat",
    "document_utility_hierarchical",
    "document_utility_depth3",
    "document_adaptive_d0",
    "document_adaptive_d1",
    "document_adaptive_d2",
    "document_adaptive_d3",
)
FREE_DOCUMENT_TOPOLOGY_FAMILIES: tuple[Family, ...] = (
    "document_free",
    "document_utility_direct",
    "document_utility_flat",
    "document_utility_hierarchical",
    "document_utility_depth3",
    "document_adaptive_d0",
    "document_adaptive_d1",
    "document_adaptive_d2",
    "document_adaptive_d3",
)
UTILITY_DOCUMENT_TOPOLOGIES: dict[Family, str] = {
    "document_utility_direct": "direct",
    "document_utility_flat": "flat",
    "document_utility_hierarchical": "hierarchical",
    "document_utility_depth3": "hierarchical",
    "document_adaptive_d0": "direct",
    "document_adaptive_d1": "flat",
    "document_adaptive_d2": "hierarchical",
    "document_adaptive_d3": "hierarchical",
}
ADAPTIVE_DOCUMENT_DEPTHS: dict[Family, int] = {
    "document_adaptive_d0": 0,
    "document_adaptive_d1": 1,
    "document_adaptive_d2": 2,
    "document_adaptive_d3": 3,
}
SPECIALIST_FAMILIES: tuple[Family, ...] = (
    "specialist_local",
    "specialist_generic",
    "specialist_table_join",
    "specialist_table_reconcile",
    "specialist_source_ast",
    "specialist_source_config",
    "specialist_recursive_table",
    "specialist_recursive_source",
)
SPECIALIST_TERMINAL_FAMILIES: tuple[Family, ...] = (
    "specialist_generic",
    "specialist_table_join",
    "specialist_table_reconcile",
    "specialist_source_ast",
    "specialist_source_config",
)
SPECIALIST_RECURSIVE_FAMILIES: tuple[Family, ...] = (
    "specialist_recursive_table",
    "specialist_recursive_source",
)
SPECIALIST_EXPERTS = {
    "generic_worker": {
        "capability": "General terminal file reading and straightforward Python calculations.",
        "affordances": ["single_json_arithmetic"],
        "limitations": "Single JSON artifact only; no multi-artifact reconciliation or source/config structure.",
    },
    "table_analyst": {
        "capability": "CSV and JSON joins, filters, grouping, reconciliation, and exact integer arithmetic.",
        "affordances": ["single_json_arithmetic", "multi_artifact_table"],
        "limitations": "No Python AST or source-configuration inspection.",
    },
    "source_inspector": {
        "capability": "Python AST and source/configuration inspection with exact structural calculations.",
        "affordances": ["source_config_inspection"],
        "limitations": "No tabular joins, ledger reconciliation, or generic JSON-list arithmetic.",
    },
}
TRAIN_VARIANTS = (0, 1, 2, 3)
EVAL_VARIANTS = (4, 5)
COMPLETION_GATE_PATH = "/workspace/.subagent-communication/completion_gate.py"
TEACHER_CONDITIONING_TEMPLATE = (
    "<Question>\n{question}\n"
    "This is an example for a response to the question:\n"
    "<Demonstration>\n{demonstration}\n"
    "Now answer with a response of your own. Keep the thinking process in the model's "
    "designated reasoning channel. In assistant content, obey the question's exact final-answer "
    "format with no extra prose:"
)

SYSTEM_PROMPT = (
    "Coordinate work through Prime Agent's persistent IPython kernel. Delegate only the "
    "shards the request assigns to children. The callable rlm returns an admission handle, "
    "not a result. Spawn independent children before doing coordinator-local work, retain their "
    "handles, and use that local work to let children run concurrently. A child must send "
    "requested results with agent_message to its parent. Never guess a missing result or finalize "
    "before every required reply arrives. After spawning children and finishing available local "
    "work, end the current turn without polling; explicit child messages resume the active run. "
    "Use agent_observe only to diagnose a bounded failure, never as the completion channel. "
    "The direct result of await rlm(...) is an RLMSpawnHandle with .name. If that handle was "
    "lost, rlm.list_subagents() returns RLMSubagent entries with .session_name instead; do not "
    "confuse the two APIs. "
    "Preserve successful state across turns, do not repeat unchanged cells, and return the "
    "requested JSON object only."
)
MESSAGE_DELIVERY_GUIDANCE = (
    "After yielding, Prime Agent resumes this same conversation with incoming messages visibly "
    "labeled `[from child:<name>]` or `[from parent]`. There is no "
    "`agent_message.list_messages` API; do not call it, inspect a roster, or poll."
)

OWNERSHIP_GUIDANCE = (
    "For this ownership-transition collection, make the first coordinator action one "
    "IPython cell. Assign each coordinator-owned value named by the task to a descriptive "
    "variable (for a follow-up task, use multiplier), and assign await rlm(...) to a child "
    "handle. Put the delegated path and child work only inside the rlm prompt; do not read, "
    "open, parse, or inspect that path in coordinator code. After a successful spawn, end "
    "the turn without polling."
)


class SubagentCommunicationData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: InstructionLevel = "standard"
    utility_policy_profile: UtilityPolicyProfile = "historical_v1"
    teacher_conditioned: bool = False
    answer: dict[str, int]
    expected_children: tuple[str, ...] = ()
    child_paths: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    followup_secret: int | None = None
    demonstration: str | None = None
    demonstrations: dict[str, str] | None = None
    turn_demonstrations: dict[str, str | list[str | None] | None] | None = None
    child_request_demonstrations: dict[str, list[str | None] | None] | None = None
    coordinator_demonstrations: dict[str, str | None] | None = None
    reward_post_fan_in_control: bool = False
    reward_bidirectional_control: bool = False
    preferred_expert: str | None = None


def _weighted(values: list[int]) -> int:
    return sum((index + 1) * value for index, value in enumerate(values))


def _values(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(-19, 29) for _ in range(count)]


def _json(values: list[int]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _document_fixture(
    variant: int,
    instance: int,
    seed: int,
) -> tuple[str, dict[str, str], dict[str, int]]:
    rng = random.Random(seed * 1_000_003 + variant * 10_007 + instance * 101)
    root = f"/workspace/document-recursion/v{variant}-i{instance}"
    files: dict[str, str] = {}
    answer: dict[str, int] = {}
    for index, stem in enumerate(("alpha", "beta", "gamma")):
        section_count = 2 + (variant + index) % 3
        sections = []
        for section in range(section_count):
            values = [rng.randint(10, 99) for _ in range(3 + (section + index) % 3)]
            sections.append(
                f"## Section {section + 1}\n\n"
                f"{stem} evidence " + " ".join(f"value-{value}" for value in values)
            )
        content = f"# {stem.title()} document\n\n" + "\n\n".join(sections) + "\n"
        path = f"{root}/{stem}.md"
        files[path] = content
        answer[f"{stem}_words"] = len(content.split())
        answer[f"{stem}_h2"] = sum(line.startswith("## ") for line in content.splitlines())
    answer["total_words"] = sum(answer[f"{stem}_words"] for stem in ("alpha", "beta", "gamma"))
    answer["total_h2"] = sum(answer[f"{stem}_h2"] for stem in ("alpha", "beta", "gamma"))
    return root, files, answer


def _document_worker_instruction(path: str) -> str:
    return (
        f"Read {path} using the CLI or IPython. Count words with Python str.split() over the "
        "complete file contents and count lines beginning exactly `## `. Send one JSON object "
        "with integer keys `words` and `h2` to your parent using "
        "await agent_message.send(json.dumps(result), receiver_role='parent'). After the "
        "delivery receipt succeeds, stop."
    )


def _document_manager_instruction(root: str) -> str:
    assignments = "\n".join(
        f"- {stem}-document-worker owns {root}/{stem}.md"
        for stem in ("alpha", "beta", "gamma")
    )
    final_keys = ", ".join(
        [
            *(f"{stem}_words, {stem}_h2" for stem in ("alpha", "beta", "gamma")),
            "total_words, total_h2",
        ]
    )
    return (
        "[recursive document coordinator session contract]\n"
        "session_role=document_coordinator\n"
        "is_root=false\n"
        "has_parent=true\n"
        "can_delegate=true\n"
        "can_finalize_user=false\n"
        "maximum_descendant_depth=1\n"
        "return_contract=exactly_one_parent_report\n"
        "[local cognition facts]\n"
        "owns_required_evidence=false\n"
        "remaining_work_requires_decomposition=false\n"
        "terminal_shards_ready=true\n"
        f"You own the document directory {root}. Do not compute the three file statistics "
        "yourself. Delegate all three assignments below to three independent terminal children, "
        "retaining their handles and spawning them before waiting:\n"
        f"{assignments}\n"
        "Each child must read only its assigned file, count all words with Python str.split(), "
        "count lines beginning exactly `## `, and send one JSON object with integer keys "
        "`words` and `h2` to you through agent_message.send. After all three explicit child "
        "reports arrive, assemble one JSON object with the per-file values and totals. Its exact "
        f"keys are: {final_keys}. Send that object exactly once to receiver_role='parent', then stop."
    )


def _document_subgroup_manager_instruction(
    root: str, group: str, stems: tuple[str, ...]
) -> str:
    assignments = "\n".join(
        f"- {stem}-document-worker owns {root}/{stem}.md" for stem in stems
    )
    partial_keys = ", ".join(
        key for stem in stems for key in (f"{stem}_words", f"{stem}_h2")
    )
    return (
        "[recursive document coordinator session contract]\n"
        "session_role=document_coordinator\n"
        "document_coordinator_level=subgroup\n"
        f"document_group={group}\n"
        "is_root=false\n"
        "has_parent=true\n"
        "can_delegate=true\n"
        "can_finalize_user=false\n"
        "maximum_descendant_depth=1\n"
        "return_contract=exactly_one_parent_report\n"
        "[local cognition facts]\n"
        "owns_required_evidence=false\n"
        "remaining_work_requires_decomposition=false\n"
        "terminal_shards_ready=true\n"
        f"You own only document group {group} under {root}. Do not read or inspect its files. "
        "Delegate every assignment below to the named terminal children, retaining all handles "
        "and spawning them before waiting:\n"
        f"{assignments}\n"
        "Each child must read only its assigned file, count all words with Python str.split(), "
        "count lines beginning exactly `## `, and send one JSON object with integer keys "
        "`words` and `h2` to you through agent_message.send. After every explicit child report "
        f"arrives, send one JSON object with exactly these integer keys to your parent: {partial_keys}. "
        "Send it exactly once to receiver_role='parent', then stop.\n"
        "depth3_contract_end=subgroup"
    )


def _document_depth3_manager_instruction(root: str) -> str:
    ab_contract = _document_subgroup_manager_instruction(
        root, "alpha,beta", ("alpha", "beta")
    )
    gamma_contract = _document_subgroup_manager_instruction(
        root, "gamma", ("gamma",)
    )
    final_keys = ", ".join(
        [
            *(f"{stem}_words, {stem}_h2" for stem in ("alpha", "beta", "gamma")),
            "total_words, total_h2",
        ]
    )
    return (
        "[recursive document coordinator session contract]\n"
        "session_role=document_coordinator\n"
        "document_coordinator_level=top\n"
        "is_root=false\n"
        "has_parent=true\n"
        "can_delegate=true\n"
        "can_finalize_user=false\n"
        "maximum_descendant_depth=2\n"
        "return_contract=exactly_one_parent_report\n"
        "[local cognition facts]\n"
        "owns_required_evidence=false\n"
        "remaining_work_requires_decomposition=true\n"
        "terminal_shards_ready=false\n"
        f"You own the decomposition of document directory {root}, but may not read or inspect "
        "its files. Delegate its two disjoint groups to exactly these non-root coordinators, "
        "retaining both handles and spawning both before waiting. Preserve each complete contract.\n\n"
        "Coordinator name: ab-document-manager\n"
        f"{ab_contract}\n\n"
        "Coordinator name: gamma-document-manager\n"
        f"{gamma_contract}\n\n"
        "After both explicit subgroup reports arrive, combine their per-file values and compute "
        "the two totals. Assemble one JSON object whose exact keys are: "
        f"{final_keys}. Send that object exactly once to receiver_role='parent', then stop.\n"
        "depth3_contract_end=top"
    )


def _without_adaptive_topology_labels(contract: str) -> str:
    """Remove scorer-only topology labels from an adaptive live contract."""

    replacements = {
        "document_coordinator_level=top\n": "",
        "document_coordinator_level=subgroup\n": "",
        "maximum_descendant_depth=2\n": "",
        "maximum_descendant_depth=1\n": "",
        "depth3_contract_end=top": "adaptive_recursive_contract_end=coordinator",
        "depth3_contract_end=subgroup": "adaptive_recursive_contract_end=group",
    }
    for old, new in replacements.items():
        contract = contract.replace(old, new)
    return contract


def _adaptive_document_request(
    family: Family,
    root: str,
    schema: str,
) -> str:
    required_depth = ADAPTIVE_DOCUMENT_DEPTHS[family]
    flat_paths = {
        f"{stem}-document-worker": f"{root}/{stem}.md"
        for stem in ("alpha", "beta", "gamma")
    }
    assignments = "\n".join(
        f"- {child}: {_document_worker_instruction(path)}"
        for child, path in flat_paths.items()
    )
    manager_prompt = _without_adaptive_topology_labels(
        _document_depth3_manager_instruction(root)
        if required_depth == 3
        else _document_manager_instruction(root)
    )
    owns_required_evidence = required_depth == 0
    remaining_work_requires_decomposition = required_depth in {2, 3}
    terminal_shards_ready = required_depth == 1
    inspection_policy = (
        "The current coordinator owns and may inspect the complete document directory."
        if owns_required_evidence
        else "The current coordinator may not inspect the document files."
    )
    admission_policy = {
        0: "No additional cognition is needed when owned evidence is sufficient.",
        1: "The current coordinator may admit the three terminal specialists.",
        2: "The current coordinator may admit one coordinator that can organize terminal specialists.",
        3: "The current coordinator may admit one coordinator because the remaining objective itself requires further decomposition.",
    }[required_depth]
    return (
        "[adaptive document cognition contract]\n"
        "Complete the document objective using the cheapest sufficient local cognition. "
        "At every coordinator node, choose only the next action justified by that node's "
        "owned evidence and scoped remaining work. Return only a generic action name. The "
        "harness executes the selected action without changing it.\n\n"
        "[local cognition facts]\n"
        f"owns_required_evidence={str(owns_required_evidence).lower()}\n"
        "remaining_work_requires_decomposition="
        f"{str(remaining_work_requires_decomposition).lower()}\n"
        f"terminal_shards_ready={str(terminal_shards_ready).lower()}\n"
        f"{inspection_policy} {admission_policy}\n\n"
        "Candidate action for owned work: Inspect every Markdown file in "
        f"{root} yourself using the CLI or IPython; do not create a subagent.\n\n"
        "Candidate action for terminal shards: admit the named terminal specialists, retain "
        "all handles, and aggregate only their explicit reports:\n"
        f"{assignments}\n\n"
        "Candidate action for work that still needs decomposition: admit exactly one "
        "coordinator named document-manager and preserve this complete scoped contract:\n\n"
        f"{manager_prompt}\n\n"
        "Choose exactly one generic next action. When the selected work completes, return "
        f"{schema}."
    )


def _specialist_registry(
    expert_ids: tuple[str, ...],
    relative_costs: dict[str, float] | None = None,
) -> str:
    unknown = set(expert_ids) - set(SPECIALIST_EXPERTS)
    if unknown:
        raise ValueError(f"unknown specialist registry entries: {sorted(unknown)}")
    costs = relative_costs or {expert_id: 1.0 for expert_id in expert_ids}
    if set(costs) != set(expert_ids) or not all(
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and cost > 0
        for cost in costs.values()
    ):
        raise ValueError("specialist relative costs must cover the visible registry")
    entries = []
    for expert_id in expert_ids:
        expert = SPECIALIST_EXPERTS[expert_id]
        entries.append(
            json.dumps(
                {
                    "expert_id": expert_id,
                    "role": "terminal_worker",
                    "capability": expert["capability"],
                    "affordances": expert["affordances"],
                    "tools": ["ipython", "filesystem"],
                    "limitations": expert["limitations"],
                    "relative_cost": costs[expert_id],
                },
                separators=(",", ":"),
            )
        )
    return "[capability registry]\n" + "\n".join(entries)


def _specialist_fixture(
    family: Family,
    variant: int,
    instance: int,
    seed: int,
) -> tuple[dict[str, str], int, str, str | None, tuple[str, ...]]:
    rng = random.Random(seed * 1_000_003 + variant * 10_007 + instance * 101)
    base_family = {
        "specialist_recursive_table": "specialist_table_join",
        "specialist_recursive_source": "specialist_source_ast",
    }.get(family, family)
    root = f"/workspace/specialist-worker/v{variant}-i{instance}"
    if base_family == "specialist_local":
        values = [rng.randint(-20, 30) for _ in range(8 + variant % 3)]
        return {}, _weighted(values), json.dumps(values), None, ()
    if base_family == "specialist_generic":
        values = [rng.randint(-30, 40) for _ in range(12 + variant % 3)]
        path = f"{root}/values.json"
        answer = _weighted(values)
        files = {path: json.dumps(values)}
        objective = (
            f"Read {path} as a top-level JSON integer list. Compute "
            f"{WEIGHTED_CHECKSUM_FORMULA}. Send exactly one compact JSON object "
            "with integer key `value` to receiver_role='parent', then stop."
        )
        return files, answer, objective, "generic_worker", (path,)
    if base_family == "specialist_table_join":
        rates = {f"c{index}": rng.randint(1, 4) for index in range(4)}
        rows = []
        total = 0
        for index in range(10 + variant):
            customer = f"c{rng.randrange(4)}"
            units = rng.randint(1, 8)
            price = rng.randint(3, 17)
            status = "posted" if (index + variant) % 3 else "void"
            rows.append(f"t{index},{customer},{units},{price},{status}")
            if status == "posted":
                total += units * price * rates[customer]
        csv_path = f"{root}/transactions.csv"
        rate_path = f"{root}/rates.json"
        files = {
            csv_path: "transaction_id,customer,units,unit_price,status\n"
            + "\n".join(rows)
            + "\n",
            rate_path: json.dumps(rates, sort_keys=True),
        }
        objective = (
            f"Read {csv_path} and {rate_path}. Join each transaction to its customer multiplier, "
            "keep only rows whose status is exactly `posted`, and sum "
            "units * unit_price * multiplier. Send exactly one compact JSON object with integer "
            "key `value` to receiver_role='parent', then stop."
        )
        return files, total, objective, "table_analyst", (csv_path, rate_path)
    if base_family == "specialist_table_reconcile":
        corrections: dict[str, int] = {}
        rows = []
        total = 0
        for index in range(7 + variant % 2):
            sku = f"sku-{index}"
            opening = rng.randint(20, 80)
            received = rng.randint(0, 25)
            shipped = rng.randint(0, 30)
            correction = rng.randint(-4, 6)
            corrections[sku] = correction
            rows.append(f"{sku},{opening},{received},{shipped}")
            total += opening + received - shipped + correction
        csv_path = f"{root}/inventory.csv"
        correction_path = f"{root}/corrections.json"
        files = {
            csv_path: "sku,opening,received,shipped\n" + "\n".join(rows) + "\n",
            correction_path: json.dumps(corrections, sort_keys=True),
        }
        objective = (
            f"Read {csv_path} and {correction_path}. For every SKU compute opening + received "
            "- shipped + its JSON correction, then sum the reconciled quantities across all SKUs. "
            "Send exactly one compact JSON object with integer key `value` to "
            "receiver_role='parent', then stop."
        )
        return files, total, objective, "table_analyst", (csv_path, correction_path)
    if base_family == "specialist_source_ast":
        function_counts = []
        files = {}
        for module_index, module in enumerate(("alpha", "beta")):
            sync_count = 2 + (variant + module_index) % 3
            async_count = 1 + (instance + module_index) % 2
            decorated_count = 1 + module_index
            lines = ["def trace(fn):\n    return fn\n"]
            for index in range(sync_count):
                decorator = "@trace\n" if index < decorated_count else ""
                lines.append(
                    f"{decorator}def sync_{index}(x):\n    return x + {index}\n"
                )
            for index in range(async_count):
                lines.append(
                    f"async def async_{index}(x):\n    return x * {index + 1}\n"
                )
            path = f"{root}/{module}.py"
            files[path] = "\n".join(lines)
            function_counts.append((sync_count + 1, async_count, decorated_count))
        total = sum(
            sync * 2 + async_count * 3 + decorated
            for sync, async_count, decorated in function_counts
        )
        paths = tuple(sorted(files))
        objective = (
            f"Parse the complete Python files {paths[0]} and {paths[1]} with ast. Across both "
            "files, count every FunctionDef (including helper functions), every AsyncFunctionDef, "
            "and every function node with at least one decorator. Compute "
            "2 * FunctionDef + 3 * AsyncFunctionDef + decorated_function_nodes. Send exactly one "
            "compact JSON object with integer key `value` to receiver_role='parent', then stop."
        )
        return files, total, objective, "source_inspector", paths
    if base_family == "specialist_source_config":
        worker_count = rng.randint(2, 8)
        timeout = rng.randint(10, 40)
        enabled = [
            name
            for index, name in enumerate(("cache", "audit", "retry", "trace"))
            if (variant + index) % 2
        ]
        toml_path = f"{root}/service.toml"
        env_path = f"{root}/features.env"
        files = {
            toml_path: f"[runtime]\nworkers = {worker_count}\ntimeout_seconds = {timeout}\n",
            env_path: "\n".join(
                f"FEATURE_{name.upper()}={'true' if name in enabled else 'false'}"
                for name in ("cache", "audit", "retry", "trace")
            )
            + "\n",
        }
        total = worker_count * timeout + len(enabled)
        objective = (
            f"Read {toml_path} with tomllib and {env_path} as KEY=value lines. Compute runtime "
            "workers * timeout_seconds + the number of feature values exactly equal to `true`. "
            "Send exactly one compact JSON object with integer key `value` to "
            "receiver_role='parent', then stop."
        )
        return files, total, objective, "source_inspector", (toml_path, env_path)
    raise ValueError(f"unsupported specialist family: {family}")


def _specialist_assignment(objective: str, paths: tuple[str, ...]) -> str:
    return "[terminal specialist assignment]\n" + json.dumps(
        {"worker_name": "task-worker", "objective": objective, "paths": paths},
        separators=(",", ":"),
    )


def _specialist_request(
    family: Family,
    objective: str,
    paths: tuple[str, ...],
    expert_ids: tuple[str, ...],
    relative_costs: dict[str, float] | None = None,
) -> str:
    registry = _specialist_registry(expert_ids, relative_costs)
    assignment = _specialist_assignment(objective, paths)
    recursive = family in SPECIALIST_RECURSIVE_FAMILIES
    owns = family == "specialist_local"
    manager_contract = (
        "[recursive document coordinator session contract]\n"
        "[recursive specialist coordinator session contract]\n"
        "session_role=document_coordinator\n"
        "is_root=false\n"
        "has_parent=true\n"
        "can_delegate=true\n"
        "can_finalize_user=false\n"
        "return_contract=exactly_one_parent_report\n"
        "[local cognition facts]\n"
        "owns_required_evidence=false\n"
        "remaining_work_requires_decomposition=false\n"
        "terminal_shards_ready=true\n"
        f"{registry}\n"
        f"{assignment}\n"
        "Choose the terminal worker by public capability metadata. After its explicit report, "
        "relay one JSON object with integer key `result` to receiver_role='parent', then stop.\n"
        "specialist_recursive_contract_end=manager"
    )
    facts = (
        "owns_required_evidence=true\n"
        "remaining_work_requires_decomposition=false\n"
        "terminal_shards_ready=false"
        if owns
        else (
            "owns_required_evidence=false\n"
            "remaining_work_requires_decomposition=true\n"
            "terminal_shards_ready=false"
            if recursive
            else (
                "owns_required_evidence=false\n"
                "remaining_work_requires_decomposition=false\n"
                "terminal_shards_ready=true"
            )
        )
    )
    if owns:
        local_values = json.loads(objective)
        if not isinstance(local_values, list) or not all(
            isinstance(value, int) for value in local_values
        ):
            raise ValueError("specialist-local objective must be an integer list")
        candidates = (
            "[owned specialist-local values]\n"
            f"{json.dumps(local_values, separators=(',', ':'))}\n"
            "Candidate owned action: compute sum((index + 1) * value for index, value in "
            "enumerate(values)) over the displayed integers."
        )
    elif recursive:
        candidates = (
            "Candidate decomposition action: admit exactly one coordinator named "
            "specialist-manager and preserve this complete scoped contract:\n\n"
            f"{manager_contract}"
        )
    else:
        candidates = (
            "Candidate terminal action: choose one registered terminal worker and delegate this "
            "complete assignment without changing it:\n"
            f"{assignment}"
        )
    return (
        "[specialist worker routing contract]\n"
        "Choose the cheapest sufficient local cognition from the public state and capability "
        "registry. Author the generic local action and, only for terminal delegation, the "
        "registered expert_id. The harness executes that exact choice and never replaces it.\n\n"
        "[local cognition facts]\n"
        f"{facts}\n"
        f"{registry}\n\n"
        f"{candidates}\n\n"
        "When the selected work completes, return one JSON object with exactly integer key `result`."
    )


def _child_instruction(path: str) -> str:
    return (
        f"Read {path}. Its top-level JSON value is the integer list itself, not an object: bind "
        f"values = json.loads(Path({path!r}).read_text()) without indexing a dictionary key. "
        f"Compute its weighted checksum using exactly {WEIGHTED_CHECKSUM_FORMULA}. Then execute exactly once: "
        "await agent_message.send(str(checksum), receiver_role='parent'). After the delivery "
        "receipt succeeds, answer with a concise confirmation and stop."
    )


def _followup_child_instruction(path: str) -> str:
    return (
        f"You are key-worker, my child. Read {path}. Its top-level JSON value is the integer list "
        f"itself, not an object: bind values = json.loads(Path({path!r}).read_text()), then retain "
        "subtotal = sum(values). "
        "Do not call rlm or message a child. In a separate IPython call execute exactly "
        "await agent_message.send('need multiplier', receiver_role='parent'). End that turn and "
        "resume only when the visible [from parent] follow-up arrives. Then make exactly one "
        "IPython call. Bind its integer body with int(...) rather than guessing or hardcoding it "
        "while preserving subtotal, compute result = subtotal * multiplier, and execute await "
        "agent_message.send(json.dumps({'subtotal': subtotal, 'result': result}), "
        "receiver_role='parent'). Do not print, inspect, or split that computation and send across "
        "cells; then stop."
    )


def _handshake_child_instruction() -> str:
    return (
        "You are relay-worker, my child. Do not call rlm or message a child. "
        "agent_message is a Python module, not a direct model tool. In one IPython call execute "
        "exactly await agent_message.send('need nonce', receiver_role='parent'), then end the turn "
        "without another tool call. Resume only after a visible [from parent] reply arrives. In one "
        "IPython call bind its integer body with int(...) and execute await "
        "agent_message.send(str(nonce), receiver_role='parent'), then stop."
    )


def _child_demonstration(path: str, checksum: int) -> str:
    return (
        "A successful Prime Agent child trajectory is:\n\n"
        "First execute this IPython cell, preserving the exact assigned path:\n"
        "import json\n"
        "from pathlib import Path\n"
        f"values = json.loads(Path({path!r}).read_text())\n"
        "checksum = sum((index + 1) * value for index, value in enumerate(values))\n"
        "checksum\n\n"
        f"The observed checksum is {checksum}. Then execute exactly once:\n"
        "await agent_message.send(str(checksum), receiver_role='parent')\n\n"
        "After the send receipt succeeds, answer with a concise confirmation and stop."
    )


def _expert_demonstration(
    family: Family,
    prompt: str,
    answer: dict[str, int],
    child_paths: dict[str, str],
    followup_secret: int | None,
) -> str | None:
    if family == "direct" or family in (*DOCUMENT_FAMILIES, *SPECIALIST_FAMILIES):
        return None
    if family == "followup":
        if followup_secret is None:
            raise ValueError("followup demonstration requires a multiplier")
        path = child_paths["key-worker"]
        child_prompt = _followup_child_instruction(path)
        return (
            "A successful Prime Agent coordinator trajectory is:\n\n"
            f"First retain multiplier = {followup_secret} in IPython state. Then execute exactly:\n"
            f"child = await rlm({child_prompt!r}, name='key-worker')\n\n"
            "Retain child and end the turn without polling. When key-worker sends exactly "
            "`need multiplier`, execute once:\n"
            "await agent_message.send(str(multiplier), receiver_role='child', "
            "receiver_name=child.name)\n\n"
            "End the turn again. After key-worker's second explicit message arrives, bind its "
            "subtotal and result without overwriting child or multiplier, then return exactly:\n"
            f"{json.dumps(answer)}"
        )
    if family == "handshake":
        if followup_secret is None:
            raise ValueError("handshake demonstration requires a nonce")
        child_prompt = _handshake_child_instruction()
        return (
            "A successful Prime Agent coordinator trajectory is:\n\n"
            f"First retain nonce = {followup_secret} in IPython state. Then execute exactly:\n"
            f"child = await rlm({child_prompt!r}, name='relay-worker')\n\n"
            "Retain child and end the turn without polling. When relay-worker sends exactly "
            "`need nonce`, execute once:\n"
            "await agent_message.send(str(nonce), receiver_role='child', "
            "receiver_name=child.name)\n\n"
            "End the turn again. After relay-worker echoes the nonce in its second explicit "
            "message, return exactly:\n"
            f"{json.dumps(answer)}"
        )

    matched = re.search(r"local checksum for (\[[^\]]+\])", prompt)
    if matched is None:
        raise ValueError(f"{family} prompt is missing coordinator-local values")
    local_values = json.loads(matched.group(1))
    local_cell = (
        f"local_values = {local_values!r}\n"
        "local = sum((index + 1) * value for index, value in enumerate(local_values))\n"
        "local"
    )
    if family == "single":
        path = child_paths["shard-worker"]
        child_prompt = _child_instruction(path)
        return (
            "A successful Prime Agent coordinator trajectory is:\n\n"
            "First execute this exact IPython cell, preserving the complete shard path:\n"
            f"handle = await rlm({child_prompt!r}, name='shard-worker')\n\n"
            "The silent result means the child was admitted and handle remains available. Next execute:\n"
            f"{local_cell}\n\n"
            "Make no tool call, including no IPython comment, pass, string literal, or print. "
            "The next assistant response itself must be exactly `Waiting for "
            "shard-worker's explicit reply.` as plain assistant text. After shard-worker's explicit "
            "message arrives, bind its "
            "body as remote, compute total = local + remote mentally, and make no tool call. Your "
            "first response after that message must be exactly this bare JSON, with no prose or Markdown:\n"
            f"{json.dumps(answer)}"
        )

    alpha_path = child_paths["alpha-worker"]
    beta_path = child_paths["beta-worker"]
    alpha_prompt = _child_instruction(alpha_path)
    beta_prompt = _child_instruction(beta_path)
    return (
        "A successful Prime Agent coordinator trajectory is:\n\n"
        "First execute these separate IPython cells, preserving both complete paths and handles:\n"
        f"alpha_handle = await rlm({alpha_prompt!r}, name='alpha-worker')\n\n"
        f"beta_handle = await rlm({beta_prompt!r}, name='beta-worker')\n\n"
        f"{local_cell}\n\n"
        "End the turn without polling or sending READY messages. When explicit child messages "
        "arrive, use each `[from child:<name>]` source label as authoritative provenance: bind "
        f"the alpha-worker body to alpha_value = {answer['alpha']} and the beta-worker body to "
        f"beta_value = {answer['beta']}. Do not bind an agent_message.send receipt, call "
        "agent_observe, or trigger refinement. Compute total from local, alpha_value, and "
        "beta_value, then return exactly:\n"
        f"{json.dumps(answer)}"
    )


def _bidirectional_child_demonstration(
    family: Family,
    child_paths: dict[str, str],
    answer: dict[str, int],
) -> tuple[str, str]:
    if family == "followup":
        path = child_paths["key-worker"]
        child_prompt = _followup_child_instruction(path)
        child_demo = (
            "A successful Prime Agent key-worker trajectory is:\n\n"
            "First execute:\n"
            "import json\n"
            "from pathlib import Path\n"
            f"values = json.loads(Path({path!r}).read_text())\n"
            "subtotal = sum(values)\n"
            "await agent_message.send('need multiplier', receiver_role='parent')\n\n"
            "End the turn without polling. After the parent's explicit reply arrives, bind only "
            "its integer body, preserving subtotal:\n"
            f"multiplier = {answer['multiplier']}\n"
            "result = subtotal * multiplier\n"
            "await agent_message.send(json.dumps({'subtotal': subtotal, 'result': result}), "
            "receiver_role='parent')\n\n"
            "After the send receipt succeeds, answer with a concise confirmation and stop."
        )
    elif family == "handshake":
        child_prompt = _handshake_child_instruction()
        child_demo = (
            "A successful Prime Agent relay-worker trajectory is:\n\n"
            "First execute this IPython cell exactly once:\n"
            "await agent_message.send('need nonce', receiver_role='parent')\n\n"
            "End the turn without polling. After the parent's explicit reply arrives, bind only "
            "its integer body:\n"
            f"nonce = {answer['nonce']}\n"
            "await agent_message.send(str(nonce), receiver_role='parent')\n\n"
            "After the second send receipt succeeds, answer with a concise confirmation and stop."
        )
    else:
        raise ValueError(f"{family} is not a bidirectional family")
    return f"[task from parent]\n\n{child_prompt}", child_demo


def _bidirectional_turn_demonstrations(
    family: Family,
    prompt: str,
    child_paths: dict[str, str],
    answer: dict[str, int],
) -> dict[str, str | list[str | None] | None] | None:
    if family == "followup":
        path = child_paths["key-worker"]
        child_prompt = _followup_child_instruction(path)
        coordinator_steps = [
            (
                "Execute one IPython cell that preserves both values:\n"
                f"multiplier = {answer['multiplier']}\n"
                f"child = await rlm({child_prompt!r}, name='key-worker')\n"
                "Then end the turn without polling or finalizing."
            ),
            (
                "The visible key-worker message requests the multiplier. Preserve child and execute once:\n"
                "await agent_message.send(str(multiplier), receiver_role='child', "
                "receiver_name=child.name)\n"
                "Then end the turn without polling."
            ),
            (
                "Use the visible key-worker result and retained multiplier. Do not call a tool. "
                f"Return exactly {json.dumps(answer)}"
            ),
        ]
        child_steps = [
            (
                "Execute one IPython cell:\n"
                "import json\n"
                "from pathlib import Path\n"
                f"values = json.loads(Path({path!r}).read_text())\n"
                "subtotal = sum(values)\n"
                "await agent_message.send('need multiplier', receiver_role='parent')\n"
                "Then end the turn without polling or answering the task."
            ),
            (
                "Bind only the integer body of the visible parent message while preserving subtotal, then execute:\n"
                f"multiplier = {answer['multiplier']}\n"
                "result = subtotal * multiplier\n"
                "await agent_message.send(json.dumps({'subtotal': subtotal, 'result': result}), "
                "receiver_role='parent')\n"
                "This must be one IPython cell with no print or inspection; then stop."
            ),
        ]
    elif family == "handshake":
        child_prompt = _handshake_child_instruction()
        coordinator_steps = [
            (
                "Execute one IPython cell that preserves both values:\n"
                f"nonce = {answer['nonce']}\n"
                f"child = await rlm({child_prompt!r}, name='relay-worker')\n"
                "Then end the turn without polling or finalizing."
            ),
            (
                "The visible relay-worker message requests the nonce. Preserve child and execute once:\n"
                "await agent_message.send(str(nonce), receiver_role='child', "
                "receiver_name=child.name)\n"
                "Then end the turn without polling."
            ),
            (
                "Use the nonce echoed in the visible relay-worker message. Do not call a tool. "
                f"Return exactly {json.dumps(answer)}"
            ),
        ]
        child_steps = [
            (
                "Execute one IPython cell exactly once:\n"
                "await agent_message.send('need nonce', receiver_role='parent')\n"
                "Then end the turn without polling or answering the task."
            ),
            (
                "Bind only the integer body of the visible parent message, then execute one IPython cell:\n"
                f"nonce = {answer['nonce']}\n"
                "await agent_message.send(str(nonce), receiver_role='parent')\n"
                "Then stop."
            ),
        ]
    else:
        return None

    child_question = f"[task from parent]\n\n{child_prompt}"
    return {
        prompt: coordinator_steps,
        child_question: child_steps,
        # A depth-one bidirectional task has exactly one child role.
        "*": child_steps,
    }


def _bidirectional_child_request_demonstrations(
    family: Family,
    prompt: str,
    child_paths: dict[str, str],
    answer: dict[str, int],
) -> dict[str, list[str | None] | None] | None:
    turn_demonstrations = _bidirectional_turn_demonstrations(family, prompt, child_paths, answer)
    if turn_demonstrations is None:
        return None

    child_question = next(question for question in turn_demonstrations if question.startswith("[task from parent]"))
    child_steps = turn_demonstrations[child_question]
    if not isinstance(child_steps, list):
        raise TypeError("bidirectional child turn demonstrations must be a sequence")
    request_only = [child_steps[0], None]
    return {
        prompt: None,
        child_question: request_only,
        # A depth-one bidirectional task has exactly one child role.
        "*": request_only,
    }


def _branch_demonstrations(
    family: Family,
    prompt: str,
    answer: dict[str, int],
    child_paths: dict[str, str],
    coordinator_demonstration: str | None,
) -> dict[str, str] | None:
    if coordinator_demonstration is None:
        return None
    demonstrations = {prompt: coordinator_demonstration}
    if family == "single":
        children = (("shard-worker", "remote"),)
    elif family == "parallel":
        children = (("alpha-worker", "alpha"), ("beta-worker", "beta"))
    elif family in {"followup", "handshake"}:
        child_question, child_demo = _bidirectional_child_demonstration(
            family,
            child_paths,
            answer,
        )
        demonstrations[child_question] = child_demo
        # These families have exactly one child role, so a paraphrased child
        # question can be mapped without guessing between roles.
        demonstrations["*"] = child_demo
        return demonstrations
    else:
        return demonstrations
    for child_name, answer_key in children:
        path = child_paths[child_name]
        child_question = f"[task from parent]\n\n{_child_instruction(path)}"
        demonstrations[child_question] = _child_demonstration(path, answer[answer_key])
    return demonstrations


def keep_post_child_message_responses(trace: vf.Trace) -> list[list[bool]]:
    """Select the first trainable response after each explicit child message."""
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        select_next_response = False
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = node.sampled and any(node.mask) and id(node) not in trained_nodes
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            if node.sampled:
                branch_mask.extend([select_next_response and is_new_trainable] * span)
                select_next_response = False
                continue
            branch_mask.extend([False] * span)
            select_next_response = node.message.role == "user" and content_text(
                node.message.content
            ).lstrip().startswith("[from child:")
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def keep_coordinator_pre_child_responses(trace: vf.Trace) -> list[list[bool]]:
    """Select coordinator responses before the first explicit child reply."""
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        is_child_branch = any(
            isinstance(node.message, UserMessage)
            and content_text(node.message.content).lstrip().startswith("[task from parent]")
            for node in branch.nodes
        )
        child_replied = False
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = node.sampled and any(node.mask) and id(node) not in trained_nodes
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            if node.sampled:
                branch_mask.extend([not is_child_branch and not child_replied and is_new_trainable] * span)
                continue

            branch_mask.extend([False] * span)
            if isinstance(node.message, UserMessage) and content_text(node.message.content).lstrip().startswith(
                "[from child:"
            ):
                child_replied = True
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def keep_complete_fan_in_response(trace: vf.Trace) -> list[list[bool]]:
    """Select the first trainable response after all expected children have replied."""
    expected_children = set(trace.task.data.expected_children)
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        seen_children: set[str] = set()
        select_next_response = False
        selected_response = False
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = node.sampled and any(node.mask) and id(node) not in trained_nodes
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            if node.sampled:
                keep = select_next_response and is_new_trainable and not selected_response
                branch_mask.extend([keep] * span)
                if keep:
                    selected_response = True
                select_next_response = False
                continue

            branch_mask.extend([False] * span)
            if node.message.role != "user" or selected_response:
                continue
            match = re.match(
                r"\[from child:([^\]]+)\]",
                content_text(node.message.content).lstrip(),
            )
            if match is not None:
                seen_children.add(match.group(1))
                select_next_response = bool(expected_children) and expected_children <= seen_children
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def keep_bidirectional_state_transitions(trace: vf.Trace) -> list[list[bool]]:
    """Select the first response at each root or child message transition."""
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        select_next_response = False
        saw_initial_user = False
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = node.sampled and any(node.mask) and id(node) not in trained_nodes
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            if node.sampled:
                branch_mask.extend([select_next_response and is_new_trainable] * span)
                select_next_response = False
                continue

            branch_mask.extend([False] * span)
            if node.message.role != "user":
                continue
            text = content_text(node.message.content).lstrip()
            if not saw_initial_user:
                saw_initial_user = True
                select_next_response = True
            elif text.startswith(("[from child:", "[from parent")):
                select_next_response = True
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def keep_first_coordinator_response(trace: vf.Trace) -> list[list[bool]]:
    """Select only the coordinator's first trainable response."""
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    selected = False
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        is_child_branch = any(
            isinstance(node.message, UserMessage)
            and content_text(node.message.content).lstrip().startswith("[task from parent]")
            for node in branch.nodes
        )
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = node.sampled and any(node.mask) and id(node) not in trained_nodes
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            keep = is_new_trainable and not is_child_branch and not selected
            branch_mask.extend([keep] * span)
            if keep:
                selected = True
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def keep_first_coordinator_tool_call(
    trace: vf.Trace,
    *,
    start_token_id: int = 248058,
    end_token_id: int = 248059,
) -> list[list[bool]]:
    """Select only the first coordinator response's serialized tool call."""
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    selected = False
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        is_child_branch = any(
            isinstance(node.message, UserMessage)
            and content_text(node.message.content).lstrip().startswith("[task from parent]")
            for node in branch.nodes
        )
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = node.sampled and any(node.mask) and id(node) not in trained_nodes
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            node_mask = [False] * span
            if is_new_trainable and not is_child_branch and not selected:
                try:
                    start = node.token_ids.index(start_token_id)
                    end = node.token_ids.index(end_token_id, start + 1)
                except ValueError:
                    pass
                else:
                    node_mask = [
                        start <= index <= end and sampled
                        for index, sampled in enumerate(node.mask)
                    ]
                selected = True
            branch_mask.extend(node_mask)
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def keep_child_request_phase_responses(trace: vf.Trace) -> list[list[bool]]:
    """Select child responses through the first successful message to its parent."""
    masks: list[list[bool]] = []
    trained_nodes: set[int] = set()
    for branch in trace.branches:
        branch_mask: list[bool] = []
        has_trainable_tokens = False
        is_child_branch = any(
            isinstance(node.message, UserMessage)
            and content_text(node.message.content).lstrip().startswith("[task from parent]")
            for node in branch.nodes
        )
        request_complete = False
        for node in branch.nodes:
            span = len(node.token_ids)
            is_new_trainable = node.sampled and any(node.mask) and id(node) not in trained_nodes
            if is_new_trainable:
                trained_nodes.add(id(node))
                has_trainable_tokens = True
            if node.sampled:
                branch_mask.extend([is_child_branch and not request_complete and is_new_trainable] * span)
                continue

            branch_mask.extend([False] * span)
            if not is_child_branch or request_complete:
                continue
            if isinstance(node.message, ToolMessage) and _message_sent(content_text(node.message.content)) or isinstance(node.message, UserMessage) and content_text(node.message.content).lstrip().startswith(
                "[from parent"
            ):
                request_complete = True
        if has_trainable_tokens:
            masks.append(branch_mask)
    return masks


def _completion_gate_source(
    expected_keys: tuple[str, ...],
    family: Family,
    *,
    expected_types: dict[str, str] | None = None,
    required_child_messages: dict[str, int] | None = None,
    feedback: str | None = None,
) -> str:
    default_child_messages = {
        "single": {"shard-worker": 1},
        "parallel": {"alpha-worker": 1, "beta-worker": 1},
        "followup": {"key-worker": 2},
        "handshake": {"relay-worker": 2},
        "document_flat": {
            "alpha-document-worker": 1,
            "beta-document-worker": 1,
            "gamma-document-worker": 1,
        },
        "document_hierarchical": {"document-manager": 1},
        **{family: {"task-worker": 1} for family in SPECIALIST_TERMINAL_FAMILIES},
        **{
            family: {"specialist-manager": 1}
            for family in SPECIALIST_RECURSIVE_FAMILIES
        },
    }.get(family, {})
    if required_child_messages is None:
        required_child_messages = default_child_messages
    if expected_types is None:
        expected_types = dict.fromkeys(expected_keys, "int")
    if set(expected_types) != set(expected_keys):
        raise ValueError("completion gate types must match the expected keys")
    supported_types = {"bool", "dict", "float", "int", "list", "null", "str"}
    unsupported_types = set(expected_types.values()) - supported_types
    if unsupported_types:
        raise ValueError(f"unsupported completion gate types: {sorted(unsupported_types)}")
    if feedback is not None:
        gate_feedback = feedback
    elif family == "followup":
        gate_feedback = (
            "completion gate: final JSON is not ready. Preserve the existing delegation: "
            "do not inspect the delegated shard, spawn another child, or redo the child's work. "
            "A request exists only when this conversation visibly contains a new user message "
            "beginning `[from child:key-worker]`; never infer it from the task, demonstration, "
            "child status, or expected protocol. If that visible message requests the multiplier, "
            "send it exactly once to the existing child. Otherwise call no tool and respond exactly "
            "`Waiting for key-worker's request.` This active autonomous run will accept future child "
            "messages. Return strict JSON only after a later visible key-worker message contains its "
            "final subtotal and result."
        )
    elif family == "handshake":
        gate_feedback = (
            "completion gate: final JSON is not ready. Preserve the existing delegation: do not "
            "spawn another child. A request exists only when this conversation visibly contains a "
            "new user message beginning `[from child:relay-worker]`; never infer it from the task, "
            "demonstration, child status, or expected protocol. If that visible message requests the "
            "nonce, send it exactly once to the existing child. Otherwise call no tool and respond "
            "exactly `Waiting for relay-worker's request.` This active autonomous run will accept "
            "future child messages. Return strict JSON only after a later visible relay-worker "
            "message echoes the nonce."
        )
    elif family in {
        "single",
        "parallel",
        "document_flat",
        "document_hierarchical",
        *SPECIALIST_TERMINAL_FAMILIES,
        *SPECIALIST_RECURSIVE_FAMILIES,
    }:
        gate_feedback = (
            "completion gate: final JSON is not ready. Preserve every existing delegation: do not "
            "inspect a delegated shard, spawn replacement children, or redo their work. If no new "
            "child message is present, your next assistant response itself must be a brief waiting "
            "status. Do not call IPython with a comment, pass, string literal, print, or any other "
            "placeholder; this active autonomous run will accept future child messages. After every "
            "existing child sends its result, make no tool call and return bare strict JSON immediately, "
            "with no prose or Markdown."
        )
    elif family in FREE_DOCUMENT_TOPOLOGY_FAMILIES:
        gate_feedback = (
            "completion gate: final JSON is not ready. Preserve the one topology you already "
            "selected and do not switch or mix graphs. If you selected direct work, finish the "
            "local document computation. If you delegated, do not inspect delegated files, "
            "spawn replacements, poll, or redo child work; respond with a brief waiting status "
            "until every child in the selected graph reports. Then return bare strict JSON with "
            "no prose or Markdown."
        )
    else:
        gate_feedback = (
            "completion gate: final JSON is not ready. Complete the coordinator-local computation "
            "and return one JSON object with exactly the requested keys and integer values."
        )
    format_feedback = (
        "completion gate: all required child evidence is already present. Do not call a tool, "
        "inspect state, repeat work, or wait. Return one bare JSON object now with exactly the "
        f"keys {expected_keys!r} and value types {expected_types!r}, with no prose or Markdown fence."
    )
    return f"""import ast
import json
import os
import sys
import time
from pathlib import Path

EXPECTED_KEYS = {expected_keys!r}
EXPECTED_TYPES = {expected_types!r}
REQUIRED_CHILD_MESSAGES = {required_child_messages!r}
FREE_DOCUMENT_TOPOLOGY = {family in FREE_DOCUMENT_TOPOLOGY_FAMILIES!r}


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def session_message(entry):
    if entry.get("type") == "custom_message":
        return {{
            "role": "custom",
            "customType": entry.get("customType"),
            "content": entry.get("content"),
            "details": entry.get("details"),
        }}
    message = entry.get("message")
    return message if isinstance(message, dict) else {{}}


def child_message_sender(message):
    if (
        message.get("role") == "custom"
        and message.get("customType") == "agent_message"
    ):
        details = message.get("details")
        if not isinstance(details, dict) or details.get("fromRelationship") != "child":
            return None
        sender = details.get("from")
        if not isinstance(sender, dict):
            return None
        for key in ("sessionName", "sessionId", "activeSessionId"):
            value = sender.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    # Older harnesses rendered received agent messages as user prompts.
    if message.get("role") != "user":
        return None
    text = content_text(message.get("content"))
    prefix = "[from child:"
    if not text.startswith(prefix) or "]" not in text[len(prefix):]:
        return None
    return text[len(prefix):].split("]", 1)[0]


def child_message_payload(message):
    details = message.get("details")
    raw = details.get("message") if isinstance(details, dict) else None
    if not isinstance(raw, str):
        text = content_text(message.get("content"))
        raw = text.rsplit("\\n\\n", 1)[-1].strip()
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    if not all(isinstance(key, str) and type(value) is int for key, value in payload.items()):
        return None
    return payload


def value_matches_type(value, expected_type):
    if expected_type == "bool":
        return isinstance(value, bool)
    if expected_type == "dict":
        return isinstance(value, dict)
    if expected_type == "float":
        return isinstance(value, float)
    if expected_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "list":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    if expected_type == "str":
        return isinstance(value, str)
    return False


def selected_required_child_messages(entries):
    if not FREE_DOCUMENT_TOPOLOGY:
        return REQUIRED_CHILD_MESSAGES

    def ipython_codes(value):
        codes = []

        def visit(item):
            if isinstance(item, list):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, dict):
                return
            function = item.get("function")
            candidate = function if isinstance(function, dict) else item
            name = candidate.get("name") or item.get("toolName")
            arguments = candidate.get("arguments")
            if arguments is None:
                arguments = item.get("input")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                arguments = None
            code = arguments.get("code") if isinstance(arguments, dict) else None
            if name == "ipython" and isinstance(code, str) and code not in codes:
                codes.append(code)
            for child in item.values():
                visit(child)

        visit(value)
        return codes

    flat_names = {{
        "alpha-document-worker",
        "beta-document-worker",
        "gamma-document-worker",
    }}
    for entry in entries:
        message = session_message(entry)
        if message.get("role") != "assistant":
            continue
        for code in ipython_codes(message):
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            names = set()
            spawn_calls = 0
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "rlm"
                ):
                    continue
                spawn_calls += 1
                for keyword in node.keywords:
                    if (
                        keyword.arg == "name"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        names.add(keyword.value.value)
            if spawn_calls == 3 and names == flat_names:
                return {{name: 1 for name in flat_names}}
            if spawn_calls == 1 and names == {{"document-manager"}}:
                return {{"document-manager": 1}}
            if spawn_calls == 0 and "/workspace/document-recursion/" in code:
                return {{}}
            if spawn_calls:
                return None
    return None


def inspect_sessions():
    agent_dir = Path(os.environ.get("PRIME_AGENT_CODING_AGENT_DIR", ""))
    session_files = sorted(
        (agent_dir / "sessions").rglob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    observed_counts = {{}}
    wait_required = {{}}
    observed_reports = {{}}
    valid = False
    for path in session_files:
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not entries:
            continue
        header = entries[0]
        if header.get("rlmDepth") not in (None, 0):
            continue
        if header.get("parentSession") or header.get("parentSessionId"):
            continue
        required_child_messages = selected_required_child_messages(entries)
        if required_child_messages is None:
            continue
        if not wait_required:
            wait_required = required_child_messages
        for name in required_child_messages:
            observed_counts.setdefault(name, 0)
        final_index = None
        final_payload = None
        for index in range(len(entries) - 1, -1, -1):
            message = session_message(entries[index])
            if message.get("role") != "assistant":
                continue
            final_index = index
            try:
                final_payload = json.loads(content_text(message.get("content")).strip())
            except (TypeError, json.JSONDecodeError):
                final_payload = None
            break
        if final_index is None:
            continue
        child_message_counts = {{name: 0 for name in required_child_messages}}
        all_child_message_counts = {{name: 0 for name in required_child_messages}}
        seen_child_message_ids = set()
        for index, entry in enumerate(entries):
            message = session_message(entry)
            child_name = child_message_sender(message)
            if child_name not in child_message_counts:
                continue
            details = message.get("details")
            message_id = details.get("id") if isinstance(details, dict) else None
            if isinstance(message_id, str) and message_id in seen_child_message_ids:
                continue
            if isinstance(message_id, str):
                seen_child_message_ids.add(message_id)
            all_child_message_counts[child_name] += 1
            payload = child_message_payload(message)
            if payload is not None:
                observed_reports[child_name] = payload
            if index < final_index:
                child_message_counts[child_name] += 1
        for child_name, count in all_child_message_counts.items():
            observed_counts[child_name] = max(observed_counts[child_name], count)
        child_evidence_ready = all(
            child_message_counts.get(name, 0) >= count
            for name, count in required_child_messages.items()
        )
        valid = valid or (
            child_evidence_ready
            and isinstance(final_payload, dict)
            and set(final_payload) == set(EXPECTED_KEYS)
            and all(
                value_matches_type(final_payload[key], EXPECTED_TYPES[key])
                for key in EXPECTED_KEYS
            )
        )
    observed = tuple(observed_counts.get(name, 0) for name in wait_required)
    ready = all(
        observed_counts.get(name, 0) >= count
        for name, count in wait_required.items()
    )
    return valid, ready, observed, bool(wait_required), observed_reports


valid, child_evidence_observed, observed, should_wait, observed_reports = inspect_sessions()
if valid:
    raise SystemExit(0)

# Child messages are asynchronous session state, not worktree changes. Wait for
# one bounded state transition so Prime Agent's unchanged-worktree retry guard
# cannot exhaust the autonomous loop before a live child reply is delivered.
try:
    grace_seconds = float(
        os.environ.get("VF_PRIME_AGENT_CHILD_EVIDENCE_GRACE_SECONDS", "30")
    )
except ValueError:
    grace_seconds = 30.0
grace_seconds = max(0.0, min(grace_seconds, 60.0))
deadline = time.monotonic() + grace_seconds
while should_wait and time.monotonic() < deadline:
    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    valid, child_evidence_observed, current, should_wait, observed_reports = inspect_sessions()
    if valid:
        raise SystemExit(0)
    if current != observed:
        break

feedback = {format_feedback!r} if child_evidence_observed else {gate_feedback!r}
if FREE_DOCUMENT_TOPOLOGY and child_evidence_observed and observed_reports:
    feedback += "\\nObserved child report map: " + json.dumps(
        observed_reports, sort_keys=True, separators=(",", ":")
    )
print(feedback, file=sys.stderr)
raise SystemExit(1)
"""


def _task_prompt(
    family: Family,
    variant: int,
    instance: int,
    seed: int,
    instruction_level: InstructionLevel,
    prompt_contract: PromptContract = "historical_v1",
    utility_policy_profile: UtilityPolicyProfile = "historical_v1",
    available_experts: tuple[str, ...] = tuple(SPECIALIST_EXPERTS),
    specialist_relative_costs: dict[str, float] | None = None,
) -> tuple[
    str,
    dict[str, int],
    tuple[str, ...],
    dict[str, str],
    dict[str, str],
    int | None,
    str | None,
]:
    rng = random.Random(seed * 1_000_003 + variant * 10_007 + instance * 101)
    prefix = (
        "Return one JSON object with exactly the requested keys and integer values."
    )
    if family not in (*DOCUMENT_FAMILIES, *SPECIALIST_FAMILIES) and (
        prompt_contract == "historical_v1" or family in {"direct", "single", "parallel"}
    ):
        prefix = f"{prefix} A shard checksum is sum((index + 1) * value)."
    preferred_expert = None

    if family == "direct":
        values = _values(rng, 8 + variant % 3)
        answer = {"checksum": _weighted(values)}
        request = (
            f"This small task must stay in the coordinator. Compute the checksum of {_json(values)} "
            'without creating a subagent. Return {"checksum": value}.'
        )
        children: tuple[str, ...] = ()
        child_paths: dict[str, str] = {}
        files: dict[str, str] = {}
        secret = None
        guidance = "Solve locally and do not spawn or message a child."
    elif family == "single":
        local = _values(rng, 7 + variant % 2)
        remote = _values(rng, 13 + variant % 3)
        remote_path = f"/workspace/subagent-shards/v{variant}-i{instance}-remote.json"
        local_value = _weighted(local)
        remote_value = _weighted(remote)
        answer = {
            "local": local_value,
            "remote": remote_value,
            "total": local_value + remote_value,
        }
        request = (
            f"Compute the local checksum for {_json(local)} yourself. Delegate the independent "
            f"remote checksum in {remote_path} to one child named shard-worker and obtain its "
            'explicit reply. Return {"local": value, "remote": value, "total": value}.'
        )
        children = ("shard-worker",)
        child_paths = {"shard-worker": remote_path}
        files = {remote_path: json.dumps(remote)}
        secret = None
        child_instruction = _child_instruction(remote_path)
        guidance = (
            f"Do not open {remote_path} in the coordinator. Spawn the child before computing the "
            f"local checksum, using handle = await rlm({child_instruction!r}, "
            "name='shard-worker'). Retain handle and compute the local checksum in "
            "separate state-preserving IPython calls while the child runs. Then stop calling tools "
            "and end the current turn so the explicit child message can resume the active run. "
            "Never finalize without that reply; the admission handle is not the answer."
        )
    elif family == "parallel":
        local = _values(rng, 6)
        alpha = _values(rng, 11 + variant % 2)
        beta = _values(rng, 12 + (variant + 1) % 2)
        alpha_path = f"/workspace/subagent-shards/v{variant}-i{instance}-alpha.json"
        beta_path = f"/workspace/subagent-shards/v{variant}-i{instance}-beta.json"
        local_value = _weighted(local)
        alpha_value = _weighted(alpha)
        beta_value = _weighted(beta)
        answer = {
            "local": local_value,
            "alpha": alpha_value,
            "beta": beta_value,
            "total": local_value + alpha_value + beta_value,
        }
        request = (
            f"Compute the local checksum for {_json(local)} yourself. In parallel, delegate "
            f"{alpha_path} to alpha-worker and {beta_path} to beta-worker. Each child must "
            'reply explicitly. Return {"local": value, "alpha": value, "beta": value, '
            '"total": value}.'
        )
        children = ("alpha-worker", "beta-worker")
        child_paths = {"alpha-worker": alpha_path, "beta-worker": beta_path}
        files = {alpha_path: json.dumps(alpha), beta_path: json.dumps(beta)}
        secret = None
        alpha_instruction = _child_instruction(alpha_path)
        beta_instruction = _child_instruction(beta_path)
        guidance = (
            f"Do not open {alpha_path} or {beta_path} in the coordinator. In separate IPython "
            f"calls, assign alpha = await rlm({alpha_instruction!r}, name='alpha-worker') and "
            f"beta = await rlm({beta_instruction!r}, name='beta-worker'). Retain both handles, "
            "then stop calling tools for this turn. "
            f"{MESSAGE_DELIVERY_GUIDANCE}"
        )
    elif family in SPECIALIST_FAMILIES:
        files, result, objective, preferred_expert, paths = _specialist_fixture(
            family, variant, instance, seed
        )
        answer = {"result": result}
        secret = None
        request = _specialist_request(
            family,
            objective,
            paths,
            available_experts,
            specialist_relative_costs,
        )
        if family == "specialist_local":
            children = ()
            child_paths = {}
            guidance = (
                "Keep the displayed computation local and do not activate a worker."
            )
        elif family in SPECIALIST_RECURSIVE_FAMILIES:
            children = ("specialist-manager",)
            child_paths = {"specialist-manager": paths[0].rsplit("/", 1)[0]}
            guidance = (
                "Delegate the still-decomposable objective to one coordinator. That coordinator "
                "must choose a terminal worker from the same public registry."
            )
        else:
            children = ("task-worker",)
            child_paths = {"task-worker": paths[0]}
            guidance = (
                "Choose one registered terminal worker by its public capabilities, preserve the "
                "complete assignment, and wait for its explicit report."
            )
    elif family in DOCUMENT_FAMILIES:
        root, files, answer = _document_fixture(variant, instance, seed)
        secret = None
        schema = "{" + ", ".join(f'"{key}": value' for key in answer) + "}"
        if family == "document_direct":
            request = (
                f"Inspect every Markdown file in {root} yourself using the CLI or IPython; do "
                "not create a subagent. For each file, count words with Python str.split() over "
                "the complete contents and count lines beginning exactly `## `. Return "
                f"{schema}."
            )
            children = ()
            child_paths = {}
            guidance = (
                "Keep the work local. First inventory the directory, then read every listed "
                "file, compute the requested counts, verify coverage, and return strict JSON."
            )
        elif family == "document_flat":
            children = tuple(f"{stem}-document-worker" for stem in ("alpha", "beta", "gamma"))
            child_paths = {
                child: f"{root}/{child.removesuffix('-document-worker')}.md"
                for child in children
            }
            assignments = "\n".join(
                f"- {child}: {_document_worker_instruction(path)}"
                for child, path in child_paths.items()
            )
            request = (
                "Delegate the following three document files to the three specifically named "
                "terminal children. Do not read the files in the root coordinator. Spawn all "
                "three children before waiting, retain every admission handle, and use only "
                "their explicit reports to assemble the final result:\n"
                f"{assignments}\n"
                f"Return {schema}."
            )
            guidance = (
                "Use one retained rlm handle per named worker, end the turn without polling, "
                "and aggregate only after all three explicit child reports arrive."
            )
        elif family == "document_hierarchical":
            manager_prompt = _document_manager_instruction(root)
            children = ("document-manager",)
            child_paths = {"document-manager": root}
            request = (
                f"Delegate the complete document directory {root} to exactly one non-root "
                "coordinator named document-manager. Do not inspect that directory in the root. "
                "The delegated prompt must begin with the recursive document coordinator header "
                "and preserve this complete contract:\n\n"
                f"{manager_prompt}\n\n"
                "Retain the document-manager admission handle and end the turn without polling. "
                "When its one explicit report arrives, return the same values as strict JSON with "
                f"this schema: {schema}."
            )
            guidance = (
                "Use one retained rlm handle for document-manager. The manager, not the root, "
                "owns file discovery, leaf delegation, aggregation, and the parent report."
            )
        elif family in ADAPTIVE_DOCUMENT_DEPTHS:
            flat_paths = {
                f"{stem}-document-worker": f"{root}/{stem}.md"
                for stem in ("alpha", "beta", "gamma")
            }
            request = _adaptive_document_request(family, root, schema)
            children = ()
            child_paths = {**flat_paths, "document-manager": root}
            guidance = (
                "Use only the public local cognition facts to choose the cheapest sufficient "
                "generic next action. Execute one graph, wait passively for delegated evidence, "
                "and return the exact JSON schema."
            )
        else:
            flat_children = tuple(
                f"{stem}-document-worker" for stem in ("alpha", "beta", "gamma")
            )
            flat_paths = {
                child: f"{root}/{child.removesuffix('-document-worker')}.md"
                for child in flat_children
            }
            assignments = "\n".join(
                f"- {child}: {_document_worker_instruction(path)}"
                for child, path in flat_paths.items()
            )
            manager_prompt = (
                _document_depth3_manager_instruction(root)
                if family == "document_utility_depth3"
                else _document_manager_instruction(root)
            )
            historical_utility_policy = {
                "document_utility_direct": (
                    "The root is permitted to inspect the directory. The total agent-admission "
                    "budget is zero. Select the reliable plan that satisfies that budget."
                ),
                "document_utility_flat": (
                    "The root is not permitted to inspect the directory. Descendant depth is "
                    "limited to one, and the root may admit at most three agents. Select the "
                    "reliable plan that respects ownership and depth."
                ),
                "document_utility_hierarchical": (
                    "The root is not permitted to inspect the directory and may admit at most "
                    "one agent. That admitted agent may delegate at one further depth. Select "
                    "the reliable plan that respects ownership and both budgets."
                ),
                "document_utility_depth3": (
                    "The root is not permitted to inspect the directory and may admit at most "
                    "one agent. That admitted coordinator may delegate at one further depth, "
                    "and each admitted subgroup coordinator may delegate terminal work at one "
                    "additional depth. Select the reliable plan that respects ownership and all budgets."
                ),
            }.get(family)
            causal_utility_policy = {
                "document_utility_direct": (
                    "The root is permitted to inspect the directory and may admit up to three "
                    "agents. An admitted coordinator may delegate at one further depth. All "
                    "legal plans are otherwise equally reliable. Select the plan with the "
                    "fewest total agent admissions."
                ),
                "document_utility_flat": (
                    "The root is not permitted to inspect the directory and may admit up to "
                    "three agents. An admitted coordinator may delegate at one further depth. "
                    "All legal plans are otherwise equally reliable. Select the plan with the "
                    "fewest total agent admissions."
                ),
                "document_utility_hierarchical": (
                    "The root is not permitted to inspect the directory and may admit at most "
                    "one agent. An admitted coordinator may delegate at one further depth. All "
                    "legal plans are otherwise equally reliable. Select the plan with the "
                    "fewest total agent admissions."
                ),
                "document_utility_depth3": (
                    "The root is not permitted to inspect the directory and may admit at most "
                    "one agent. An admitted coordinator may delegate at one further depth, and "
                    "each admitted subgroup coordinator may delegate terminal work at one "
                    "additional depth. All legal plans are otherwise equally reliable. Select "
                    "the plan with the fewest root-level agent admissions."
                ),
            }.get(family)
            utility_policy = (
                causal_utility_policy
                if utility_policy_profile
                in {
                    "causal_matched_v2",
                    "causal_action_boundary_v3",
                    "causal_decision_boundary_v4",
                }
                else historical_utility_policy
            )
            contract_intro = (
                "Choose exactly one legal execution topology for this document objective. "
                "Do not mix topologies. Correctness and ownership are mandatory; use fewer "
                "agent admissions when two plans are otherwise equally reliable."
                if utility_policy is None
                else (
                    "Choose exactly one candidate execution plan for this document objective. "
                    "Do not mix plans. Correctness, ownership, and the resource policy are "
                    f"mandatory. Resource policy: {utility_policy} First select the plan with "
                    "one exact IPython cell assigning `document_topology` to `direct`, `flat`, "
                    "or `hierarchical`. The harness will execute the selected public plan."
                )
            )
            if utility_policy_profile == "causal_action_boundary_v3":
                action_boundary = (
                    "\n\n[document topology decision facts at action boundary]\n"
                    f"Resource policy: {utility_policy} Apply the causal utility rubric now. "
                    "Emit exactly one `document_topology` assignment; do not treat available "
                    "recursion as required."
                )
            elif utility_policy_profile == "causal_decision_boundary_v4":
                action_boundary = (
                    "\n\n[document topology decision rule at action boundary]\n"
                    "Apply these cases in order: (1) when the root has inspection permission, select "
                    "`direct`; (2) otherwise, when the root can admit all three terminal workers, "
                    "select `flat`; (3) otherwise, when the root can admit at most one agent and "
                    "that manager can delegate one further depth, select `hierarchical`. "
                    "Availability of deeper recursion alone never changes an earlier decision. "
                    f"Current resource policy: {utility_policy} Emit exactly one "
                    "`document_topology` assignment."
                )
            else:
                action_boundary = ""
            request = (
                "[free document topology contract]\n"
                f"{contract_intro}\n\n"
                "Legal topology `direct`: Inspect every Markdown file in "
                f"{root} yourself using the CLI or IPython; do not create a subagent.\n\n"
                "Legal topology `flat`: delegate the three files to exactly these three "
                "terminal children, retain all handles, spawn all three before waiting, and "
                "aggregate only their explicit reports:\n"
                f"{assignments}\n\n"
                "Legal topology `hierarchical`: delegate the complete directory to exactly one "
                "non-root coordinator named document-manager, retain its handle, and preserve "
                "this complete recursive contract:\n\n"
                f"{manager_prompt}"
                f"{action_boundary}\n\n"
                f"Whichever topology you choose, return {schema}."
            )
            children = ()
            child_paths = {
                **flat_paths,
                "document-manager": root,
            }
            guidance = (
                "Compare direct, flat, and hierarchical against the public ownership and resource "
                "policy. Execute only one compliant graph, wait passively for any delegated "
                "reports, and return the exact JSON schema."
            )
    elif family == "handshake":
        secret = rng.randint(1_000, 9_999)
        answer = {"nonce": secret}
        request = (
            f"Delegate a bidirectional handshake to one child named relay-worker, but do not "
            f"include nonce {secret} in its spawn prompt. The child must first request the nonce "
            "from you through an agent message. Send it only in a direct follow-up, wait for the "
            'child to echo it in a final message, then return {"nonce": value}.'
        )
        children = ("relay-worker",)
        child_paths = {"relay-worker": "need nonce"}
        files = {}
        child_instruction = _handshake_child_instruction()
        guidance = (
            "Use three causally separate phases. Initial phase: make exactly one IPython call whose "
            f"cell is nonce = {secret} followed by child = await rlm({child_instruction!r}, "
            "name='relay-worker'). Do not put the nonce in the child string. End the cell immediately "
            "after the child assignment with no print or other statement, then end the assistant "
            "turn. After the spawn receipt, make no tool call; the next assistant response must be "
            "exactly `Waiting for relay-worker's request.` Request phase: do not message the child "
            "until a later resumed coordinator turn "
            "visibly contains `[from child:relay-worker]` with `need nonce`. Only then make exactly "
            "one IPython call: await agent_message.send(str(nonce), receiver_role='child', "
            "receiver_name=child.name). After the send receipt, make no tool call; the next assistant "
            "response must be exactly `Waiting for relay-worker's final echo.` Result phase: after the "
            "later visible child echo, call no tool and return the requested bare JSON. "
            f"{MESSAGE_DELIVERY_GUIDANCE}"
        )
    else:
        remote = _values(rng, 10 + variant % 3)
        remote_path = f"/workspace/subagent-shards/v{variant}-i{instance}-followup.json"
        secret = rng.randint(31, 47)
        subtotal = sum(remote)
        answer = {"subtotal": subtotal, "multiplier": secret, "result": subtotal * secret}
        request = (
            f"Delegate {remote_path} to key-worker. The child should sum the values, but do not "
            f"include the multiplier {secret} in its spawn prompt. It must first request the missing "
            "multiplier from you through an agent message. Send it only in a direct follow-up. "
            "The child must then multiply its retained subtotal by that multiplier, send both the "
            "subtotal and product as its final result, and you must wait for that message. Then "
            'return {"subtotal": value, '
            '"multiplier": value, "result": value}.'
        )
        children = ("key-worker",)
        child_paths = {"key-worker": remote_path}
        files = {remote_path: json.dumps(remote)}
        child_instruction = _followup_child_instruction(remote_path)
        guidance = (
            f"Do not open {remote_path} in the coordinator. Use three causally separate phases. "
            "Initial phase: make exactly one IPython call whose cell is "
            f"multiplier = {secret} followed by child = await rlm({child_instruction!r}, "
            "name='key-worker'). Do not put the multiplier in the child string. End the cell "
            "immediately after the child assignment with no print or other statement, then end the "
            "assistant turn. After the spawn receipt, make no tool call; the next assistant response "
            "must be exactly `Waiting for key-worker's request.` Request phase: do not message the "
            "child until a later resumed "
            "coordinator turn visibly contains `[from child:key-worker]` with `need multiplier`. "
            "Only then make exactly one IPython call: await agent_message.send(str(multiplier), "
            "receiver_role='child', receiver_name=child.name). After the send receipt, make no tool "
            "call; the next assistant response must be exactly `Waiting for key-worker's final "
            "result.` "
            "Result phase: after the later visible child result, call no tool and return the "
            "requested bare JSON. "
            f"{MESSAGE_DELIVERY_GUIDANCE}"
        )

    if instruction_level == "guided":
        request = f"{request}\n\nProtocol hint: {guidance}"
    return (
        f"{prefix}\n\n{request}",
        answer,
        children,
        child_paths,
        files,
        secret,
        preferred_expert,
    )


@dataclass
class IpythonEvent:
    code: str
    call_id: str
    node_index: int
    output: str = ""


@dataclass
class ModelToolEvent:
    name: str
    node_index: int


def _ipython_events(trace: vf.Trace) -> list[IpythonEvent]:
    events: list[IpythonEvent] = []
    by_node_call: dict[tuple[int, str], IpythonEvent] = {}
    for node_index, node in enumerate(trace.nodes):
        message = node.message
        if isinstance(message, AssistantMessage) and node.sampled:
            for call in message.tool_calls or []:
                if call.name != "ipython":
                    continue
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    continue
                source = arguments.get("code")
                if isinstance(source, str):
                    event = IpythonEvent(code=source, call_id=call.id, node_index=node_index)
                    events.append(event)
                    by_node_call[(node_index, call.id)] = event
        elif isinstance(message, ToolMessage) and (event := by_node_call.get((node.parent, message.tool_call_id))):
            event.output = content_text(message.content)
    return events


def _model_tool_events(trace: vf.Trace) -> list[ModelToolEvent]:
    return [
        ModelToolEvent(name=call.name, node_index=node_index)
        for node_index, node in enumerate(trace.nodes)
        if isinstance(node.message, AssistantMessage) and node.sampled
        for call in node.message.tool_calls or []
    ]


def _inert_cell(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    def inert_statement(statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Pass):
            return True
        if not isinstance(statement, ast.Expr):
            return False
        if isinstance(statement.value, ast.Constant):
            return True
        call = statement.value
        return bool(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "print"
            and not call.keywords
            and all(isinstance(argument, (ast.Constant, ast.JoinedStr)) for argument in call.args)
        )

    return not tree.body or all(inert_statement(statement) for statement in tree.body)


def _branch_root(trace: vf.Trace, node_index: int) -> int:
    visited: set[int] = set()
    while trace.nodes[node_index].parent is not None:
        if node_index in visited:
            break
        visited.add(node_index)
        node_index = trace.nodes[node_index].parent
    return node_index


def _duplicate_cells(trace: vf.Trace, events: list[IpythonEvent]) -> int:
    by_branch: dict[int, list[str]] = {}
    for event in events:
        branch_root = _branch_root(trace, event.node_index)
        by_branch.setdefault(branch_root, []).append(event.code.strip())
    return sum(count - 1 for branch in by_branch.values() for count in Counter(branch).values() if count > 1)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return f"{call.func.value.id}.{call.func.attr}"
    return None


def _keyword(call: ast.Call, name: str) -> Any:
    value = next((item.value for item in call.keywords if item.arg == name), None)
    return value.value if isinstance(value, ast.Constant) else None


def _assigned_call_names(tree: ast.AST) -> set[int]:
    assigned: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if isinstance(value, ast.Await):
            value = value.value
        if isinstance(value, ast.Call):
            assigned.add(id(value))
    return assigned


def _failed(output: str) -> bool:
    return any(marker in output for marker in ("Traceback", "Error:", "SyntaxError"))


def _message_sent(output: str) -> bool:
    return not _failed(output) and (
        "agentmsg_" in output or "Agent message sent" in output or "Agent message queued" in output
    )


@dataclass
class ChildMessage:
    name: str
    body: str
    message_id: str | None
    session_id: str | None
    node_index: int


def _incoming_child_messages(trace: vf.Trace) -> list[ChildMessage]:
    messages: list[ChildMessage] = []
    for node_index, node in enumerate(trace.nodes):
        message = node.message
        if not isinstance(message, UserMessage):
            continue
        text = content_text(message.content)
        matched = re.match(
            r"\[from child:([^\]]+)\]\s*\nAgent-to-agent message received\.",
            text,
        )
        if not matched:
            continue
        body = text.rsplit("\n\n", 1)[-1].strip()
        if "completed without sending a reply" in body or body.startswith("RLM child failure"):
            continue
        message_id = re.search(r"^Message id:\s*(\S+)", text, re.MULTILINE)
        session_id = re.search(r"^From:.*\bsession\s+([^,\s]+)", text, re.MULTILINE)
        messages.append(
            ChildMessage(
                name=matched.group(1),
                body=body,
                message_id=message_id.group(1) if message_id else None,
                session_id=session_id.group(1) if session_id else None,
                node_index=node_index,
            )
        )
    return messages


def _branch_session_id(trace: vf.Trace, node_index: int) -> str | None:
    root = _branch_root(trace, node_index)
    text = content_text(trace.nodes[root].message.content)
    matched = re.search(r"^Conversation log:\s*.*?/([^/\s]+)\.jsonl\s*$", text, re.MULTILINE)
    return matched.group(1) if matched else None


def _originating_parent_send_indices(
    trace: vf.Trace,
    messages: list[ChildMessage],
    sends: list[tuple[ast.Call, IpythonEvent]],
) -> dict[int, int]:
    """Link each received message to one sampled child send."""
    remaining = list(sends)
    linked: dict[int, int] = {}
    for message in messages:
        matched_index = next(
            (
                index
                for index, (_, event) in enumerate(remaining)
                if message.message_id is not None and message.message_id in event.output
            ),
            None,
        )
        if matched_index is None and message.session_id is not None:
            matched_index = next(
                (
                    index
                    for index, (_, event) in enumerate(remaining)
                    if event.node_index < message.node_index
                    and _branch_session_id(trace, event.node_index) == message.session_id
                ),
                None,
            )
        if matched_index is not None:
            _, event = remaining.pop(matched_index)
            linked[message.node_index] = event.node_index
    return linked


def _spawn_name(call: ast.Call, output: str) -> str | None:
    configured = _keyword(call, "name")
    if isinstance(configured, str):
        return configured
    matched = re.search(r"\bname='([^']+)'", output)
    return matched.group(1) if matched else None


def _spawn_prompt(call: ast.Call) -> str | None:
    configured = _keyword(call, "prompt")
    if isinstance(configured, str):
        return configured
    if call.args and isinstance(call.args[0], ast.Constant):
        value = call.args[0].value
        return value if isinstance(value, str) else None
    return None


def _contains_integer_literal(text: str, value: int) -> bool:
    return re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", text) is not None


def _delegated_path_used_outside_spawn(source: str, path: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or path not in node.value:
            continue
        ancestor: ast.AST | None = node
        while ancestor in parents:
            ancestor = parents[ancestor]
            if isinstance(ancestor, ast.Call) and _call_name(ancestor) == "rlm":
                break
        else:
            return True
    return False


def _complete_fan_in_message(messages: list[ChildMessage], expected_children: tuple[str, ...]) -> ChildMessage | None:
    expected = set(expected_children)
    seen: set[str] = set()
    for message in messages:
        if message.name in expected:
            seen.add(message.name)
        if expected and expected <= seen:
            return message
    return None


def _post_fan_in_behavior(
    trace: vf.Trace,
    expected_children: tuple[str, ...],
    messages: list[ChildMessage],
    events: list[IpythonEvent],
) -> dict[str, float]:
    complete_message = _complete_fan_in_message(messages, expected_children)
    if complete_message is None:
        return {
            "fan_in_complete": 0.0,
            "post_fan_in_cells": 0.0,
            "post_fan_in_failed_cells": 0.0,
            "post_fan_in_forbidden_calls": 0.0,
            "post_fan_in_duplicate_cells": 0.0,
            "post_fan_in_control": 0.0,
            "post_fan_in_control_aligned": 0.0,
        }

    coordinator_root = _branch_root(trace, complete_message.node_index)
    post_events = [
        event
        for event in events
        if event.node_index > complete_message.node_index and _branch_root(trace, event.node_index) == coordinator_root
    ]
    post_non_ipython_tools = [
        event
        for event in _model_tool_events(trace)
        if event.name != "ipython"
        and event.node_index > complete_message.node_index
        and _branch_root(trace, event.node_index) == coordinator_root
    ]
    failed_cells = sum(_failed(event.output) for event in post_events)
    forbidden_calls = len(post_non_ipython_tools) + sum(_inert_cell(event.code) for event in post_events)
    for event in post_events:
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = _call_name(call) or ""
            forbidden_calls += (
                name == "rlm"
                or name.startswith("agent_observe.")
                or name
                in {
                    "rlm.list_subagents",
                    "agent_message.list_agents",
                    "agent_message.recv",
                    "agent_message.send",
                }
            )

    code_counts = Counter(event.code.strip() for event in post_events)
    duplicate_cells = sum(count - 1 for count in code_counts.values() if count > 1)
    component_scores = [
        1 / (1 + failed_cells),
        1 / (1 + forbidden_calls),
        1 / (1 + duplicate_cells),
    ]
    aligned = failed_cells == forbidden_calls == duplicate_cells == 0
    return {
        "fan_in_complete": 1.0,
        "post_fan_in_cells": float(len(post_events)),
        "post_fan_in_failed_cells": float(failed_cells),
        "post_fan_in_forbidden_calls": float(forbidden_calls),
        "post_fan_in_duplicate_cells": float(duplicate_cells),
        "post_fan_in_control": sum(component_scores) / len(component_scores),
        "post_fan_in_control_aligned": float(aligned),
    }


def _protocol_behavior(
    trace: vf.Trace,
    family: Family,
    expected_children: tuple[str, ...],
    child_paths: dict[str, str],
    followup_secret: int | None,
) -> dict[str, float]:
    events = _ipython_events(trace)
    model_tools = _model_tool_events(trace)
    non_ipython_tools = [event for event in model_tools if event.name != "ipython"]
    inert_cells = sum(_inert_cell(event.code) for event in events)
    calls: list[tuple[ast.Call, bool, IpythonEvent]] = []
    for event in events:
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned = _assigned_call_names(tree)
        calls.extend((node, id(node) in assigned, event) for node in ast.walk(tree) if isinstance(node, ast.Call))

    coordinator_root = _branch_root(trace, 0)
    branch_aware = any(_branch_root(trace, event.node_index) != coordinator_root for _, _, event in calls)

    def is_coordinator_event(event: IpythonEvent) -> bool:
        # Legacy and synthetic traces may flatten all agents into one branch.
        return not branch_aware or _branch_root(trace, event.node_index) == coordinator_root

    def is_child_event(event: IpythonEvent) -> bool:
        return not branch_aware or _branch_root(trace, event.node_index) != coordinator_root

    coordinator_calls = [(call, retained, event) for call, retained, event in calls if is_coordinator_event(event)]
    child_calls = [(call, retained, event) for call, retained, event in calls if is_child_event(event)]
    attempted_spawns = [
        (call, retained, event) for call, retained, event in coordinator_calls if _call_name(call) == "rlm"
    ]
    spawns = [item for item in attempted_spawns if not _failed(item[2].output)]
    names = {_spawn_name(call, event.output) for call, _, event in spawns}
    all_attempted_spawns = [item for item in calls if _call_name(item[0]) == "rlm"]
    all_spawns = [item for item in all_attempted_spawns if not _failed(item[2].output)]
    all_spawn_names = [
        _spawn_name(call, event.output) for call, _, event in all_spawns
    ]
    depth3_expected_names = Counter(
        {
            "document-manager": 1,
            "ab-document-manager": 1,
            "gamma-document-manager": 1,
            "alpha-document-worker": 1,
            "beta-document-worker": 1,
            "gamma-document-worker": 1,
        }
    )
    depth3_name_counts = Counter(all_spawn_names)
    depth3_graph_complete = (
        family in {"document_utility_depth3", "document_adaptive_d3"}
        and depth3_name_counts == depth3_expected_names
        and len(all_attempted_spawns) == len(all_spawns)
        and all(retained for _, retained, _ in all_spawns)
    )
    selected_free_topology: str | None = None
    utility_topology = UTILITY_DOCUMENT_TOPOLOGIES.get(family)
    if family in FREE_DOCUMENT_TOPOLOGY_FAMILIES:
        flat_names = {
            "alpha-document-worker",
            "beta-document-worker",
            "gamma-document-worker",
        }
        if not spawns:
            selected_free_topology = "direct"
            expected_children = ()
            child_paths = {}
        elif len(spawns) == 3 and names == flat_names:
            selected_free_topology = "flat"
            expected_children = tuple(sorted(flat_names))
            child_paths = {
                name: child_paths[name]
                for name in expected_children
            }
        elif len(spawns) == 1 and names == {"document-manager"}:
            selected_free_topology = "hierarchical"
            expected_children = ("document-manager",)
            child_paths = {"document-manager": child_paths["document-manager"]}
        else:
            selected_free_topology = "invalid"
            expected_children = ()
            child_paths = {}
    delegated_paths = {path for path in child_paths.values() if path.startswith("/")}
    coordinator_delegated_path_accesses = sum(
        any(_delegated_path_used_outside_spawn(event.code, path) for path in delegated_paths)
        for event in events
        if is_coordinator_event(event)
    )
    parent_messages = _incoming_child_messages(trace)
    parent_message_names = {message.name for message in parent_messages}
    child_messages = [
        (call, event)
        for call, _, event in coordinator_calls
        if _call_name(call) == "agent_message.send"
        and _keyword(call, "receiver_role") == "child"
        and _message_sent(event.output)
    ]
    parent_sends = [
        (call, event)
        for call, _, event in child_calls
        if _call_name(call) == "agent_message.send"
        and _keyword(call, "receiver_role") == "parent"
        and not _failed(event.output)
    ]
    list_calls = sum(
        _call_name(call) in {"rlm.list_subagents", "agent_message.list_agents"} and not _failed(event.output)
        for call, _, event in coordinator_calls
    )
    observation_calls = sum(
        (_call_name(call) or "").startswith("agent_observe.") and not _failed(event.output)
        for call, _, event in coordinator_calls
    ) + sum(
        event.name == "agent_observe" or event.name.startswith("agent_observe.")
        for event in non_ipython_tools
        if not branch_aware or _branch_root(trace, event.node_index) == coordinator_root
    )
    repeated = _duplicate_cells(trace, events)
    failed_cells = sum(_failed(event.output) for event in events)
    retained = sum(retained for _, retained, _ in spawns)
    def delegates_payload(name: str, path: str, call: ast.Call, event: IpythonEvent) -> bool:
        if _spawn_name(call, event.output) != name:
            return False
        prompt = _spawn_prompt(call) or ""
        if family == "handshake":
            lowered = prompt.lower()
            return "need nonce" in lowered or ("nonce" in lowered and "parent" in lowered)
        return path in prompt

    delegated = {
        name
        for name, path in child_paths.items()
        if any(delegates_payload(name, path, call, event) for call, _, event in spawns)
    }
    secret_withheld = True
    if followup_secret is not None and spawns:
        first_prompt = _spawn_prompt(spawns[0][0])
        secret_withheld = bool(
            first_prompt and not _contains_integer_literal(first_prompt, followup_secret)
        )

    expected_messages = [message for message in parent_messages if message.name in expected_children]
    parent_send_indices = _originating_parent_send_indices(trace, expected_messages, parent_sends)
    first_parent_send_by_branch: dict[int, int] = {}
    for _, event in parent_sends:
        if event.node_index not in parent_send_indices.values():
            continue
        branch_root = _branch_root(trace, event.node_index)
        first_parent_send_by_branch.setdefault(branch_root, event.node_index)
    post_request_child_tools = [
        event
        for event in model_tools
        for branch_root, send_index in first_parent_send_by_branch.items()
        if event.node_index > send_index and _branch_root(trace, event.node_index) == branch_root
    ]
    request_term = "nonce" if family == "handshake" else "multiplier"
    literal_request_phrase = f"need {request_term}"
    literal_request_message = next(
        (message for message in expected_messages if literal_request_phrase in message.body.lower()),
        None,
    )

    def originating_send_index(message: ChildMessage | None) -> int | None:
        if message is None:
            return None
        return parent_send_indices.get(message.node_index)

    explicit_parent_messages = [message for message in expected_messages if originating_send_index(message) is not None]
    explicit_parent_message_names = {message.name for message in explicit_parent_messages}

    literal_request_index = originating_send_index(literal_request_message)
    literal_result_messages = [
        (message, index)
        for message in expected_messages
        if message is not literal_request_message and (index := originating_send_index(message)) is not None
    ]
    followup_request_sent = literal_request_index is not None
    followup_after_request = any(
        literal_request_index is not None and literal_request_index < child_event.node_index
        for _, child_event in child_messages
    )
    result_after_followup = any(
        literal_request_index is not None and literal_request_index < child_event.node_index < result_index
        for _, child_event in child_messages
        for _, result_index in literal_result_messages
    )
    followup_causal = followup_request_sent and followup_after_request and result_after_followup

    # The natural task contract requires an explicit request for the missing
    # concept, not a memorized wire phrase. Preserve the literal metrics above
    # for diagnostics while scoring causal mastery from provenance and order.
    natural_request_message = next(
        (message for message in expected_messages if request_term in message.body.lower()),
        None,
    )
    natural_request_index = originating_send_index(natural_request_message)
    natural_result_messages = [
        (message, index)
        for message in expected_messages
        if message is not natural_request_message and (index := originating_send_index(message)) is not None
    ]
    natural_request_sent = natural_request_index is not None
    natural_followup_after_request = any(
        natural_request_index is not None and natural_request_index < child_event.node_index
        for _, child_event in child_messages
    )
    natural_result_after_followup = any(
        natural_request_index is not None and natural_request_index < child_event.node_index < result_index
        for _, child_event in child_messages
        for _, result_index in natural_result_messages
    )
    natural_followup_causal = (
        natural_request_sent and natural_followup_after_request and natural_result_after_followup
    )
    causal_natural_result_indices = [
        result_index
        for _, child_event in child_messages
        for _, result_index in natural_result_messages
        if natural_request_index is not None and natural_request_index < child_event.node_index < result_index
    ]
    allowed_result_send_index = min(causal_natural_result_indices, default=None)
    final_natural_result_message = next(
        (
            message
            for message, result_index in natural_result_messages
            if result_index == allowed_result_send_index
        ),
        None,
    )
    post_result_coordinator_cells = sum(
        final_natural_result_message is not None
        and event.node_index > final_natural_result_message.node_index
        and is_coordinator_event(event)
        for event in events
    )
    coordinator_followup_indices = {event.node_index for _, event in child_messages}
    post_parent_send_tool_calls = sum(
        event.node_index != allowed_result_send_index
        and event.node_index not in coordinator_followup_indices
        for event in post_request_child_tools
    )
    result_matches_secret = family != "handshake" or any(
        natural_request_index is not None
        and natural_request_index < child_event.node_index < result_index
        and followup_secret is not None
        and str(followup_secret) in message.body
        for _, child_event in child_messages
        for message, result_index in natural_result_messages
    )
    followup_phase_score = (
        float(followup_request_sent) + float(followup_after_request) + float(result_after_followup)
    ) / 3
    natural_followup_phase_score = (
        float(natural_request_sent)
        + float(natural_followup_after_request)
        + float(natural_result_after_followup)
    ) / 3
    retained_ready = retained == len(expected_children)
    stateful_control_progress = 0.0
    if (
        family in {"followup", "handshake"}
        and retained_ready
        and coordinator_delegated_path_accesses == 0
    ):
        stateful_control_progress = (
            1.0
            + float(natural_request_sent)
            + float(natural_followup_after_request)
            + float(natural_result_after_followup)
            + float(secret_withheld)
            + float(repeated == 0)
        ) / 6

    if (
        family in {"direct", "document_direct", "specialist_local"}
        or selected_free_topology == "direct"
    ):
        checks = [not spawns, not parent_messages, not child_messages, repeated == 0]
    elif (
        family
        in {
            "single",
            "document_hierarchical",
            *SPECIALIST_TERMINAL_FAMILIES,
            *SPECIALIST_RECURSIVE_FAMILIES,
        }
        or selected_free_topology == "hierarchical"
    ):
        checks = [
            len(spawns) == 1,
            retained == len(expected_children),
            set(expected_children) <= names,
            set(expected_children) <= delegated,
            set(expected_children) <= parent_message_names,
            repeated == 0,
        ]
    elif family in {"parallel", "document_flat"} or selected_free_topology == "flat":
        checks = [
            len(spawns) == len(expected_children),
            retained == len(expected_children),
            set(expected_children) <= names,
            set(expected_children) <= delegated,
            set(expected_children) <= parent_message_names,
            repeated == 0,
        ]
    elif family in FREE_DOCUMENT_TOPOLOGY_FAMILIES:
        checks = [False]
    else:
        checks = [
            len(spawns) == 1,
            retained == len(expected_children),
            set(expected_children) <= names,
            set(expected_children) <= delegated,
            secret_withheld,
            len(child_messages) >= 1,
            len(expected_messages) >= 2,
            natural_request_sent,
            natural_followup_after_request,
            natural_result_after_followup,
            *([result_matches_secret] if family == "handshake" else []),
            repeated == 0,
        ]
    if family not in {"direct", "document_direct"} and selected_free_topology != "direct":
        checks.append(coordinator_delegated_path_accesses == 0)
    if utility_topology is not None:
        checks.append(selected_free_topology == utility_topology)
    if family in {"document_utility_depth3", "document_adaptive_d3"}:
        checks.append(depth3_graph_complete)
    post_fan_in = _post_fan_in_behavior(
        trace,
        expected_children,
        parent_messages,
        events,
    )
    bidirectional_control_components = [
        failed_cells == 0,
        list_calls == 0,
        observation_calls == 0,
        not non_ipython_tools,
        inert_cells == 0,
        post_parent_send_tool_calls == 0,
        coordinator_delegated_path_accesses == 0,
        repeated == 0,
        len(child_messages) == 1,
        len(explicit_parent_messages) == 2,
        post_result_coordinator_cells == 0,
    ]
    bidirectional_control = (
        sum(bidirectional_control_components) / len(bidirectional_control_components)
        if (
            family in {"followup", "handshake"}
            and natural_followup_causal
            and coordinator_delegated_path_accesses == 0
        )
        else 0.0
    )
    clean_checks = [
        all(checks),
        list_calls == 0,
        observation_calls == 0,
        failed_cells == 0,
        not non_ipython_tools,
        inert_cells == 0,
        post_parent_send_tool_calls == 0,
    ]
    if family in {
        "single",
        "parallel",
        "document_flat",
        "document_hierarchical",
    } or selected_free_topology in {"flat", "hierarchical"}:
        clean_checks.append(set(expected_children) <= explicit_parent_message_names)
        clean_checks.append(post_fan_in["post_fan_in_cells"] == 0)
    elif family in {"followup", "handshake"}:
        clean_checks.append(len(child_messages) == 1)
        clean_checks.append(len(explicit_parent_messages) == len(expected_messages) == 2)
        clean_checks.append(post_result_coordinator_cells == 0)
    adaptive_required_depth = ADAPTIVE_DOCUMENT_DEPTHS.get(family)
    adaptive_exercised_depth = -1
    if adaptive_required_depth is not None:
        if selected_free_topology == "direct" and not all_spawns and events:
            adaptive_exercised_depth = 0
        elif selected_free_topology == "flat" and len(all_spawns) == 3:
            adaptive_exercised_depth = 1
        elif selected_free_topology == "hierarchical" and depth3_graph_complete:
            adaptive_exercised_depth = 3
        elif selected_free_topology == "hierarchical" and len(all_spawns) == 4:
            adaptive_exercised_depth = 2
    behavior = {
        "protocol_score": sum(checks) / len(checks),
        "protocol_aligned": float(all(checks)),
        "clean_protocol_aligned": float(all(clean_checks)),
        "topology_valid": float(selected_free_topology != "invalid") if family in FREE_DOCUMENT_TOPOLOGY_FAMILIES else 0.0,
        "topology_direct": float(selected_free_topology == "direct"),
        "topology_flat": float(selected_free_topology == "flat"),
        "topology_hierarchical": float(selected_free_topology == "hierarchical"),
        "topology_utility_aligned": (
            float(selected_free_topology == utility_topology)
            if utility_topology is not None
            else 0.0
        ),
        "coordination_spawn_calls": float(
            sum(
                _call_name(call) == "rlm" and not _failed(event.output)
                for call, _, event in calls
            )
        ),
        "coordination_failed_spawn_calls": float(
            len(all_attempted_spawns) - len(all_spawns)
        ),
        "depth3_top_manager_spawns": float(
            depth3_name_counts["document-manager"]
        ),
        "depth3_subgroup_manager_spawns": float(
            depth3_name_counts["ab-document-manager"]
            + depth3_name_counts["gamma-document-manager"]
        ),
        "depth3_leaf_spawns": float(
            sum(
                depth3_name_counts[f"{stem}-document-worker"]
                for stem in ("alpha", "beta", "gamma")
            )
        ),
        "depth3_graph_complete": float(depth3_graph_complete),
        "maximum_exercised_coordination_depth": float(
            3 if depth3_graph_complete else 0
        ),
        "adaptive_required_coordination_depth": float(
            adaptive_required_depth if adaptive_required_depth is not None else -1
        ),
        "adaptive_exercised_coordination_depth": float(adaptive_exercised_depth),
        "adaptive_depth_aligned": float(
            adaptive_required_depth is not None
            and adaptive_exercised_depth == adaptive_required_depth
        ),
        "spawn_calls": float(len(spawns)),
        "failed_spawn_calls": float(len(attempted_spawns) - len(spawns)),
        "retained_handles": float(retained),
        "named_children": float(len(set(expected_children) & names)),
        "delegated_payloads": float(len(delegated)),
        "coordinator_delegated_path_accesses": float(coordinator_delegated_path_accesses),
        "messages_to_parent": float(len(parent_messages)),
        "explicit_messages_to_parent": float(len(explicit_parent_messages)),
        "post_parent_send_tool_calls": float(post_parent_send_tool_calls),
        "post_result_coordinator_cells": float(post_result_coordinator_cells),
        "messages_to_child": float(len(child_messages)),
        "roster_calls": float(list_calls),
        "observation_calls": float(observation_calls),
        "non_ipython_tool_calls": float(len(non_ipython_tools)),
        "inert_cells": float(inert_cells),
        "secret_withheld": float(secret_withheld),
        "followup_request_sent": float(followup_request_sent),
        "followup_after_request": float(followup_after_request),
        "result_after_followup": float(result_after_followup),
        "followup_result_matches_secret": float(result_matches_secret),
        "followup_phase_score": followup_phase_score,
        "followup_causal": float(followup_causal),
        "natural_request_sent": float(natural_request_sent),
        "natural_followup_after_request": float(natural_followup_after_request),
        "natural_result_after_followup": float(natural_result_after_followup),
        "natural_followup_phase_score": natural_followup_phase_score,
        "natural_followup_causal": float(natural_followup_causal),
        "stateful_control_progress": stateful_control_progress,
        "bidirectional_control": bidirectional_control,
        "duplicate_cells": float(repeated),
        "failed_cells": float(failed_cells),
    }
    behavior.update(post_fan_in)
    return behavior


def _answer_score(reply: str, expected: dict[str, int]) -> float:
    try:
        actual = json.loads(reply.strip())
    except (AttributeError, json.JSONDecodeError):
        return 0.0
    if not isinstance(actual, dict):
        return 0.0
    return sum(actual.get(key) == value for key, value in expected.items()) / len(expected)


def _ownership_transition_behavior(
    trace: vf.Trace,
    family: str,
    expected_children: tuple[str, ...],
    child_paths: dict[str, str],
    followup_secret: int | None,
) -> dict[str, float]:
    events = _ipython_events(trace)
    coordinator_root = _branch_root(trace, 0)
    branch_aware = any(_branch_root(trace, event.node_index) != coordinator_root for event in events)
    coordinator_events = [
        event
        for event in events
        if not branch_aware or _branch_root(trace, event.node_index) == coordinator_root
    ]
    if not coordinator_events:
        return {
            "ownership_transition": 0.0,
            "ownership_transition_dense": 0.0,
            "ownership_one_spawn": 0.0,
            "ownership_retained_secret": 0.0,
            "ownership_retained_handle": 0.0,
            "ownership_named_child": 0.0,
            "ownership_delegated_payload": 0.0,
            "ownership_secret_withheld": 0.0,
            "ownership_path_owned": 0.0,
        }

    event = coordinator_events[0]
    try:
        tree = ast.parse(event.code)
    except SyntaxError:
        return {
            "ownership_transition": 0.0,
            "ownership_transition_dense": 0.0,
            "ownership_one_spawn": 0.0,
            "ownership_retained_secret": 0.0,
            "ownership_retained_handle": 0.0,
            "ownership_named_child": 0.0,
            "ownership_delegated_payload": 0.0,
            "ownership_secret_withheld": 0.0,
            "ownership_path_owned": 0.0,
        }

    assigned = _assigned_call_names(tree)
    spawns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "rlm"
    ]
    one_spawn = len(spawns) == 1 and not _failed(event.output)
    secret_name = "nonce" if family == "handshake" else "multiplier"
    retained_secret = followup_secret is None or any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == secret_name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value == followup_secret
        for node in ast.walk(tree)
    )
    retained = one_spawn and id(spawns[0]) in assigned
    named = one_spawn and _spawn_name(spawns[0], event.output) in expected_children
    prompt = (_spawn_prompt(spawns[0]) or "") if one_spawn else ""
    if family == "handshake":
        lowered = prompt.lower()
        delegated = one_spawn and (
            "need nonce" in lowered or ("nonce" in lowered and "parent" in lowered)
        )
    else:
        delegated = one_spawn and all(path in prompt for path in child_paths.values())
    secret_withheld = followup_secret is None or not _contains_integer_literal(
        prompt, followup_secret
    )
    path_owned = all(
        not _delegated_path_used_outside_spawn(event.code, path)
        for path in child_paths.values()
    )
    checks = (
        one_spawn,
        retained_secret,
        retained,
        named,
        delegated,
        secret_withheld,
        path_owned,
    )
    return {
        "ownership_transition": float(all(checks)),
        "ownership_transition_dense": sum(checks) / len(checks),
        "ownership_one_spawn": float(one_spawn),
        "ownership_retained_secret": float(retained_secret),
        "ownership_retained_handle": float(retained),
        "ownership_named_child": float(named),
        "ownership_delegated_payload": float(delegated),
        "ownership_secret_withheld": float(secret_withheld),
        "ownership_path_owned": float(path_owned),
    }


class SubagentCommunicationTaskConfig(vf.TaskConfig):
    reward_mode: Literal["standard", "ownership_transition"] = "standard"
    reward_shape: Literal["strict", "dense"] = "strict"


class SubagentCommunicationTask(
    vf.Task[SubagentCommunicationData, vf.State, SubagentCommunicationTaskConfig]
):
    def _include_standard_rewards(self) -> bool:
        return getattr(self.config, "reward_mode", "standard") != "ownership_transition"

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        directories = sorted(
            {
                COMPLETION_GATE_PATH.rsplit("/", 1)[0],
                *(path.rsplit("/", 1)[0] for path in self.data.files),
            }
        )
        created = await runtime.run(["mkdir", "-p", *directories], {})
        if created.exit_code != 0:
            raise RuntimeError(f"subagent shard setup failed: {created.stderr[-500:]}")
        for path, contents in self.data.files.items():
            await runtime.write(path, contents.encode())
        await runtime.write(
            COMPLETION_GATE_PATH,
            _completion_gate_source(tuple(self.data.answer), self.data.family).encode(),
        )

    @vf.reward(weight=1.0)
    async def protocol_gated_accuracy(self, trace: vf.Trace) -> float:
        if not self._include_standard_rewards():
            return 0.0
        accuracy = _answer_score(trace.last_reply, self.data.answer)
        if self.data.family in {"direct", "document_direct", "specialist_local"}:
            return accuracy
        behavior = _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )
        return accuracy * behavior["protocol_aligned"]

    @vf.reward(weight=1.0)
    async def delegation_protocol(self, trace: vf.Trace) -> float:
        if not self._include_standard_rewards():
            return 0.0
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["protocol_score"]

    @vf.reward(weight=1.0)
    async def stateful_control_progress(self, trace: vf.Trace) -> float:
        if not self._include_standard_rewards():
            return 0.0
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["stateful_control_progress"]

    @vf.reward(weight=1.0)
    async def post_fan_in_control_reward(self, trace: vf.Trace) -> float:
        if not self._include_standard_rewards():
            return 0.0
        if not self.data.reward_post_fan_in_control or self.data.family not in {
            "single",
            "parallel",
        }:
            return 0.0
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["post_fan_in_control"]

    @vf.reward(weight=1.0)
    async def bidirectional_control_reward(self, trace: vf.Trace) -> float:
        if not self._include_standard_rewards():
            return 0.0
        if not self.data.reward_bidirectional_control or self.data.family not in {
            "followup",
            "handshake",
        }:
            return 0.0
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["bidirectional_control"]

    @vf.reward(weight=1.0)
    async def ownership_transition_reward(self, trace: vf.Trace) -> float:
        if getattr(self.config, "reward_mode", "standard") != "ownership_transition":
            return 0.0
        behavior = _ownership_transition_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )
        key = (
            "ownership_transition_dense"
            if self.config.reward_shape == "dense"
            else "ownership_transition"
        )
        return behavior[key]

    @vf.metric
    async def answer_accuracy(self, trace: vf.Trace) -> float:
        return _answer_score(trace.last_reply, self.data.answer)

    @vf.metric
    async def ownership_transition(self, trace: vf.Trace) -> dict[str, float]:
        return _ownership_transition_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )

    @vf.metric
    async def delegation_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )

    @vf.metric
    async def post_fan_in_control(self, trace: vf.Trace) -> float:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["post_fan_in_control"]


class SubagentCommunicationConfig(vf.TasksetConfig):
    task: SubagentCommunicationTaskConfig = SubagentCommunicationTaskConfig()
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instruction_level: InstructionLevel = "standard"
    prompt_contract: PromptContract = "historical_v1"
    utility_policy_profile: UtilityPolicyProfile = "historical_v1"
    instances_per_template: int = Field(4, ge=1)
    instance_offset: int = Field(0, ge=0)
    seed: int = 20260809
    teacher_conditioned: bool = False
    ownership_guided: bool = False
    reward_post_fan_in_control: bool = False
    reward_bidirectional_control: bool = False
    available_experts: tuple[str, ...] = tuple(SPECIALIST_EXPERTS)
    specialist_relative_costs: dict[str, float] = Field(
        default_factory=lambda: {
            expert_id: 1.0 for expert_id in SPECIALIST_EXPERTS
        }
    )


class SubagentCommunicationTaskset(
    vf.Taskset[SubagentCommunicationTask, SubagentCommunicationConfig]
):
    def load(self) -> list[SubagentCommunicationTask]:
        if self.config.teacher_conditioned and self.config.ownership_guided:
            raise ValueError(
                "teacher_conditioned and ownership_guided are mutually exclusive"
            )
        if not self.config.available_experts:
            raise ValueError("available_experts must contain at least generic_worker")
        unknown_experts = set(self.config.available_experts) - set(SPECIALIST_EXPERTS)
        if unknown_experts:
            raise ValueError(f"unknown available_experts: {sorted(unknown_experts)}")
        if len(set(self.config.available_experts)) != len(
            self.config.available_experts
        ):
            raise ValueError("available_experts must be unique")
        if "generic_worker" not in self.config.available_experts:
            raise ValueError("available_experts must retain generic_worker")
        if (
            not set(self.config.available_experts).issubset(
                self.config.specialist_relative_costs
            )
            or not set(self.config.specialist_relative_costs).issubset(
                SPECIALIST_EXPERTS
            )
            or not all(
                isinstance(cost, (int, float))
                and not isinstance(cost, bool)
                and cost > 0
                for cost in self.config.specialist_relative_costs.values()
            )
        ):
            raise ValueError(
                "specialist_relative_costs must be positive and cover available_experts"
            )
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        instances = range(
            self.config.instance_offset,
            self.config.instance_offset + self.config.instances_per_template,
        )
        for instance in instances:
            for variant in variants:
                for family in self.config.families:
                    (
                        prompt,
                        answer,
                        children,
                        child_paths,
                        files,
                        secret,
                        preferred_expert,
                    ) = _task_prompt(
                        family,
                        variant,
                        instance,
                        self.config.seed,
                        self.config.instruction_level,
                        self.config.prompt_contract,
                        self.config.utility_policy_profile,
                        self.config.available_experts,
                        {
                            expert_id: self.config.specialist_relative_costs[expert_id]
                            for expert_id in self.config.available_experts
                        },
                    )
                    demonstration = _expert_demonstration(
                        family,
                        prompt,
                        answer,
                        child_paths,
                        secret,
                    )
                    demonstrations = _branch_demonstrations(
                        family,
                        prompt,
                        answer,
                        child_paths,
                        demonstration,
                    )
                    turn_demonstrations = _bidirectional_turn_demonstrations(
                        family,
                        prompt,
                        child_paths,
                        answer,
                    )
                    child_request_demonstrations = _bidirectional_child_request_demonstrations(
                        family,
                        prompt,
                        child_paths,
                        answer,
                    )
                    coordinator_demonstrations = (
                        {
                            **{
                                question: branch_demo if question == prompt else None
                                for question, branch_demo in demonstrations.items()
                            },
                            "*": None,
                        }
                        if demonstrations is not None
                        else None
                    )
                    if self.config.teacher_conditioned:
                        if demonstration is None:
                            raise ValueError("teacher_conditioned preflight requires a supported demonstration family")
                        prompt = TEACHER_CONDITIONING_TEMPLATE.format(
                            question=prompt,
                            demonstration=demonstration,
                        )
                    tasks.append(
                        SubagentCommunicationTask(
                            SubagentCommunicationData(
                                idx=len(tasks),
                                name=f"{family}-v{variant}-i{instance}",
                                prompt=prompt,
                                system_prompt=(
                                    f"{SYSTEM_PROMPT}\n\n{OWNERSHIP_GUIDANCE}"
                                    if self.config.ownership_guided
                                    else SYSTEM_PROMPT
                                ),
                                family=family,
                                template_variant=variant,
                                instruction_level=self.config.instruction_level,
                                utility_policy_profile=self.config.utility_policy_profile,
                                teacher_conditioned=self.config.teacher_conditioned,
                                answer=answer,
                                expected_children=children,
                                child_paths=child_paths,
                                files=files,
                                followup_secret=secret,
                                demonstration=demonstration,
                                demonstrations=demonstrations,
                                turn_demonstrations=turn_demonstrations,
                                child_request_demonstrations=child_request_demonstrations,
                                coordinator_demonstrations=coordinator_demonstrations,
                                reward_post_fan_in_control=self.config.reward_post_fan_in_control,
                                reward_bidirectional_control=self.config.reward_bidirectional_control,
                                preferred_expert=preferred_expert,
                            ),
                            self.config.task,
                        )
                    )
        return tasks
