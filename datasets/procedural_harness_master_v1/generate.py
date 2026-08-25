#!/usr/bin/env python3
"""Procedural Harness Master Benchmark V1 data generator."""

from __future__ import annotations

import argparse
import ast
import configparser
import csv
import hashlib
import io
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "procedural-harness-master-v1/episode/v1"
GENERATOR_VERSION = "2026-08-16.v2"
# Preserve the frozen task assignments while repairing public contract wording.
SEED_VERSION = "2026-08-16.v1"
Split = Literal["train_gen", "valid_gen", "ood_gen"]
CurriculumRung = Literal[
    "atomic_state",
    "atomic_send",
    "atomic_child_request",
    "atomic_followup",
    "atomic_parallel",
    "natural_n1",
    "natural_n1a",
    "natural_n1a_local",
    "natural_n1b",
    "natural_direct_control",
    "natural_n2",
]
CURRICULUM_VERSION = "2026-08-18.harness-actions-v7"
CAUSAL_N1_CURRICULUM_VERSION = "2026-08-21.causal-v3"
# Keep episode assignments fixed when public contract wording is clarified.
CURRICULUM_SEED_VERSION = "2026-08-16.harness-actions-v1"

SYSTEM_PROMPT = (
    "Coordinate through Prime Agent's persistent IPython kernel. Solve directly when "
    "the coordinator owns the work; delegate only explicitly child-owned resources. "
    "Preserve local state and child handles, spawn independent children before waiting, "
    "yield instead of polling, and treat visible child messages as the completion channel. "
    "Never inspect child-owned resources before an explicit failure and reclaim. Verify "
    "child results when coordinator-owned evidence exists. Return exactly the requested JSON."
)

TRAIN_RESOURCES = (
    "json_sum",
    "csv_total",
    "word_count",
    "md_h2",
    "log_error",
    "python_defs",
    "json_max",
    "sha_prefix",
)
OOD_RESOURCES = ("tsv_total", "xml_items", "jsonl_active_sum", "ini_quota_sum")
TRAIN_FAMILIES = ("direct", "single", "parallel", "mixed", "followup", "verify")
OOD_FAMILIES = TRAIN_FAMILIES + ("triple", "reclaim")
STYLES = {
    "train_gen": ("explicit", "natural_a", "natural_b"),
    "valid_gen": ("natural_c", "compact"),
    "ood_gen": ("terse", "narrative"),
}
PATH_STYLES = {
    "train_gen": ("nested", "flat"),
    "valid_gen": ("ticketed",),
    "ood_gen": ("opaque", "long"),
}
CHILD_NAMES = {
    "train_gen": (
        "alpha-worker",
        "beta-worker",
        "ledger-worker",
        "table-worker",
        "relay-worker",
        "audit-worker",
    ),
    "valid_gen": (
        "north-worker",
        "south-worker",
        "delta-worker",
        "proof-worker",
        "signal-worker",
    ),
    "ood_gen": ("kestrel", "mosaic", "quartz", "raven", "saffron", "vector"),
}
STATE_NAMES = {
    "train_gen": ("ticket", "nonce", "offset", "multiplier"),
    "valid_gen": ("marker", "token", "bias"),
    "ood_gen": ("anchor", "epoch", "carry"),
}
SCHEMAS = {
    "direct": '{"result": <value>}',
    "single": '{"child": <value>, "offset": <integer>, "result": <value>}',
    "parallel": '{"alpha": <value>, "beta": <value>, "offset": <integer>, "result": <value>}',
    "mixed": '{"local": <value>, "child": <value>, "offset": <integer>, "result": <value>}',
    "followup": '{"subtotal": <integer>, "multiplier": <integer>, "result": <integer>}',
    "verify": '{"child": <value>, "verified": true, "result": <value>}',
    "triple": '{"alpha": <value>, "beta": <value>, "gamma": <value>, "offset": <integer>, "result": <value>}',
    "reclaim": '{"reclaimed": true, "result": <value>}',
    "atomic_state": '{"marker": <integer>, "result": <integer>}',
    "atomic_send": '{"value": <integer>}',
    "atomic_child_request": '{"multiplier": <integer>, "request_received": true}',
    "atomic_followup": '{"multiplier": <integer>, "result": <integer>}',
    "atomic_parallel": '{"alpha": <integer>, "beta": <integer>, "result": <integer>}',
    "natural_n1": '{"finding": <integer>, "parameter": <integer>, "result": <integer>}',
    "natural_n1a": '{"finding": <integer>, "result": <integer>}',
    "natural_n1a_local": '{"finding": <integer>, "local": <integer>, "result": <integer>}',
    "natural_n1b": '{"finding": <integer>, "parameter": <integer>, "result": <integer>}',
    "natural_direct_control": '{"finding": <integer>, "parameter": <integer>, "result": <integer>}',
    "natural_n2": '{"finding": <integer>, "parameter": <integer>, "result": <integer>}',
}


@dataclass(frozen=True)
class Resource:
    family: str
    path: str
    content: str
    result: int | str
    operation: str


@dataclass(frozen=True)
class NaturalScenario:
    key: str
    context: str
    child_role: str
    finding_key: str
    parameter_label: str
    parameter_key: str
    result_key: str
    milestone: str


NATURAL_SCENARIOS: dict[Split, tuple[NaturalScenario, ...]] = {
    "train_gen": (
        NaturalScenario(
            "release_audit",
            "a release audit",
            "release reviewer",
            "finding",
            "release multiplier",
            "release_multiplier",
            "release_score",
            "the release evidence is counted",
        ),
        NaturalScenario(
            "sensor_calibration",
            "a sensor calibration",
            "calibration analyst",
            "baseline",
            "calibration factor",
            "calibration_factor",
            "calibrated_value",
            "the baseline is established",
        ),
        NaturalScenario(
            "ledger_reconciliation",
            "a ledger reconciliation",
            "ledger reviewer",
            "subtotal",
            "reconciliation factor",
            "reconciliation_factor",
            "reconciled_total",
            "the independent subtotal is ready",
        ),
        NaturalScenario(
            "incident_triage",
            "an incident triage",
            "incident analyst",
            "signal_count",
            "severity weight",
            "severity_weight",
            "risk_score",
            "the incident signals are classified",
        ),
        NaturalScenario(
            "deployment_review",
            "a deployment review",
            "deployment reviewer",
            "check_count",
            "rollout factor",
            "rollout_factor",
            "deployment_score",
            "the preflight checks are complete",
        ),
        NaturalScenario(
            "compliance_review",
            "a compliance review",
            "compliance analyst",
            "exception_count",
            "policy weight",
            "policy_weight",
            "compliance_score",
            "the exceptions are independently counted",
        ),
        NaturalScenario(
            "quality_assessment",
            "a quality assessment",
            "quality reviewer",
            "defect_count",
            "quality factor",
            "quality_factor",
            "quality_score",
            "the quality findings are complete",
        ),
        NaturalScenario(
            "capacity_review",
            "a capacity review",
            "capacity analyst",
            "usage_total",
            "capacity factor",
            "capacity_factor",
            "capacity_score",
            "the usage evidence is summarized",
        ),
    ),
    "valid_gen": (
        NaturalScenario(
            "inventory_settlement",
            "an inventory settlement",
            "inventory reviewer",
            "item_total",
            "settlement factor",
            "settlement_factor",
            "settled_total",
            "the inventory total is established",
        ),
        NaturalScenario(
            "research_screen",
            "a research evidence screen",
            "evidence reviewer",
            "evidence_count",
            "confidence weight",
            "confidence_weight",
            "confidence_score",
            "the evidence screen is complete",
        ),
        NaturalScenario(
            "security_attestation",
            "a security attestation",
            "security reviewer",
            "issue_count",
            "assurance factor",
            "assurance_factor",
            "assurance_score",
            "the security findings are complete",
        ),
    ),
    "ood_gen": (
        NaturalScenario(
            "procurement_clearance",
            "a procurement clearance",
            "procurement reviewer",
            "line_total",
            "clearance factor",
            "clearance_factor",
            "clearance_score",
            "the procurement review is complete",
        ),
        NaturalScenario(
            "reliability_forecast",
            "a reliability forecast",
            "reliability analyst",
            "event_total",
            "forecast factor",
            "forecast_factor",
            "forecast_score",
            "the event evidence is summarized",
        ),
        NaturalScenario(
            "archive_certification",
            "an archive certification",
            "archive reviewer",
            "record_count",
            "certification factor",
            "certification_factor",
            "certification_score",
            "the archive records are certified",
        ),
    ),
}

NATURAL_USER_PROMPT_FORBIDDEN = (
    "agent_message",
    "ipython",
    "retain handle",
    "rlm",
    "spawn",
    "yield",
    "poll",
)


def _seed(master: int, split: Split, index: int) -> int:
    raw = f"{SEED_VERSION}|{master}|{split}|{index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _root(split: Split, index: int, rng: random.Random) -> str:
    token = hashlib.sha256(f"{split}:{index}:{rng.random()}".encode()).hexdigest()[:10]
    style = rng.choice(PATH_STYLES[split])
    return {
        "nested": f"/workspace/harness-v1/{split}/batch-{index // 64}/ep-{token}",
        "flat": f"/workspace/hv1-{split}-{token}",
        "ticketed": f"/workspace/jobs/TKT-{1000 + index}/payload-{token}",
        "opaque": f"/workspace/.cache/{token[:2]}/{token[2:6]}/{token[6:]}",
        "long": f"/workspace/projects/procedural-harness-master-v1/{split}/{index:06d}/{token}",
    }[style]


def _resource(family: str, root: str, slot: str, rng: random.Random) -> Resource:
    stem = f"{root}/{slot}"
    if family == "json_sum":
        path = stem + ".json"
        values = [rng.randint(-20, 45) for _ in range(7)]
        content = json.dumps(values)
        result = sum(values)
        op = "sum the top-level JSON integer list"
    elif family == "csv_total":
        path = stem + ".csv"
        rows = [{"id": i, "amount": rng.randint(2, 95)} for i in range(6)]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["id", "amount"])
        writer.writeheader()
        writer.writerows(rows)
        content = buf.getvalue()
        result = sum(row["amount"] for row in rows)
        op = "sum the CSV amount column"
    elif family == "word_count":
        path = stem + ".txt"
        keyword = rng.choice(("retry", "stable", "green"))
        words = [
            rng.choice(("ready", keyword, "done", "wait", keyword)) for _ in range(18)
        ]
        content = " ".join(words)
        result = words.count(keyword)
        op = f"count exact {keyword!r} tokens"
    elif family == "md_h2":
        path = stem + ".md"
        result = rng.randint(3, 7)
        content = (
            "# Report\n\n" + "\n".join(f"## S{i}\nbody" for i in range(result)) + "\n"
        )
        op = "count level-2 Markdown headings"
    elif family == "log_error":
        path = stem + ".log"
        levels = [rng.choice(("INFO", "WARN", "ERROR")) for _ in range(16)]
        content = "\n".join(f"{x} event-{i}" for i, x in enumerate(levels)) + "\n"
        result = levels.count("ERROR")
        op = "count ERROR-level log lines"
    elif family == "python_defs":
        path = stem + ".py"
        n, m = rng.randint(2, 4), rng.randint(1, 3)
        content = "\n".join(
            [f"def f{i}():\n return {i}" for i in range(n)]
            + [f"async def a{i}():\n return {i}" for i in range(m)]
        )
        tree = ast.parse(content)
        result = sum(
            isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) for x in tree.body
        )
        op = "count top-level sync and async function definitions"
    elif family == "json_max":
        path = stem + ".json"
        values = {f"m{i}": rng.randint(-30, 120) for i in range(7)}
        content = json.dumps(values)
        result = max(values.values())
        op = "return the largest JSON integer value"
    elif family == "sha_prefix":
        path = stem + ".bin"
        content = f"payload:{rng.getrandbits(96):024x}:{slot}"
        result = hashlib.sha256(content.encode()).hexdigest()[:8]
        op = "return the first eight SHA-256 hex characters"
    elif family == "tsv_total":
        path = stem + ".tsv"
        scores = [rng.randint(3, 80) for _ in range(7)]
        content = (
            "name\tscore\n"
            + "\n".join(f"n{i}\t{x}" for i, x in enumerate(scores))
            + "\n"
        )
        result = sum(scores)
        op = "sum the TSV score column"
    elif family == "xml_items":
        path = stem + ".xml"
        n = rng.randint(4, 9)
        content = "<root>" + "".join(f"<item id='{i}'/>" for i in range(n)) + "</root>"
        result = len(ET.fromstring(content).findall("item"))
        op = "count XML item elements"
    elif family == "jsonl_active_sum":
        path = stem + ".jsonl"
        rows = [
            {"active": rng.choice((True, False)), "value": rng.randint(-9, 35)}
            for _ in range(10)
        ]
        content = "\n".join(json.dumps(x) for x in rows) + "\n"
        result = sum(x["value"] for x in rows if x["active"])
        op = "sum value for JSONL records with active=true"
    elif family == "ini_quota_sum":
        path = stem + ".ini"
        quotas = [rng.randint(1, 25) for _ in range(5)]
        content = "\n".join(f"[worker{i}]\nquota={q}\n" for i, q in enumerate(quotas))
        parser = configparser.ConfigParser()
        parser.read_string(content)
        result = sum(parser.getint(s, "quota") for s in parser.sections())
        op = "sum quota across INI sections"
    else:
        raise ValueError(f"unknown resource family {family}")
    return Resource(family, path, content, result, op)


def _pick_resource(
    split: Split, rng: random.Random, root: str, slot: str, *, integer: bool = False
) -> Resource:
    families = OOD_RESOURCES if split == "ood_gen" else TRAIN_RESOURCES
    if integer:
        families = tuple(x for x in families if x != "sha_prefix")
    return _resource(rng.choice(families), root, slot, rng)


def _combine(values: list[int | str], offset: int) -> int | str:
    if all(isinstance(x, int) for x in values):
        return sum(int(x) for x in values) + offset
    return "|".join(map(str, values + [offset]))


def _header(style: str, state: tuple[str, int] | None) -> str:
    if state is None:
        return {
            "explicit": "Execute this ownership-aware task.",
            "natural_a": "Handle this job under the stated responsibility split.",
            "natural_b": "Work this request without crossing resource boundaries.",
            "natural_c": "Coordinate according to the ownership assignments.",
            "compact": "Follow ownership.",
            "terse": "Respect ownership.",
            "narrative": "You are coordinating a job with explicit resource owners.",
        }[style]
    name, value = state
    return {
        "explicit": f"Retain {name}={value} in coordinator state.",
        "natural_a": f"Keep {name}={value} locally while you coordinate.",
        "natural_b": f"Remember {name}={value} for the final combination.",
        "natural_c": f"Preserve {name}={value} across the interaction.",
        "compact": f"Keep {name}={value}.",
        "terse": f"Carry {name}={value}.",
        "narrative": f"Your local context includes {name}={value}; retain it.",
    }[style]


def _contract(
    required: list[str],
    forbidden: list[str],
    ordering: list[tuple[str, str]],
    cardinality: dict[str, int],
) -> dict[str, Any]:
    return {
        "required_atoms": required,
        "forbidden_atoms": forbidden,
        "ordering": [{"before": a, "after": b} for a, b in ordering],
        "cardinality": cardinality,
        "hard_gate": {
            "final_answer_exact": True,
            "all_required_atoms": True,
            "all_forbidden_atoms_false": True,
            "ordering_satisfied": True,
            "cardinality_exact": True,
        },
    }


def _child(
    name: str, resource: Resource, message_contract: Any = "send result once to parent"
) -> dict[str, Any]:
    return {
        "name": name,
        "resource_path": resource.path,
        "operation": resource.operation,
        "expected_result": resource.result,
        "message_contract": message_contract,
    }


def _natural_schema(*keys: str) -> str:
    return "{" + ", ".join(f'"{key}": <integer>' for key in keys) + "}"


def _natural_intro(style: str, scenario: NaturalScenario) -> str:
    return {
        "explicit": f"Complete {scenario.context} under the stated separation of duties.",
        "natural_a": f"Prepare the final record for {scenario.context}.",
        "natural_b": f"Coordinate {scenario.context} without crossing responsibility boundaries.",
        "natural_c": f"Bring {scenario.context} to a verified conclusion.",
        "compact": f"Complete {scenario.context} with ownership intact.",
        "terse": f"Finish {scenario.context}; preserve separation of duties.",
        "narrative": f"A team is completing {scenario.context}, with evidence divided by role.",
    }[style]


def _natural_control_boundary(
    style: str,
    rung: Literal["natural_n1", "natural_n2"],
    *,
    has_local_work: bool,
) -> str:
    invalidity = {
        "explicit": "Coordinator inspection, duplicate review, or status queries invalidate the record.",
        "natural_a": "The record loses validity if the coordinator repeats the review or asks for status.",
        "natural_b": "A duplicate coordinator inspection or progress query makes the result unusable.",
        "natural_c": "Certification fails if the coordinator duplicates the evidence work or checks progress.",
        "compact": "Duplicate inspection or status checking voids the record.",
        "terse": "No duplicate inspection or progress checks; either voids the result.",
        "narrative": "The process is auditable only if the coordinator neither repeats the evidence review nor checks its progress.",
    }[style]
    if rung == "natural_n1":
        next_event = (
            "The coordinator may finish only its separately assigned local review while the reviewer works; "
            "after that, the next valid event is the delivered reviewer report."
            if has_local_work
            else "After assignment, the next valid coordinator event is the delivered reviewer report."
        )
    else:
        next_event = (
            "The coordinator may finish only its separately assigned local review before the reviewer asks for the private value. "
            "After answering that request, the next valid event is the completed reviewer report."
            if has_local_work
            else "After assignment, the next valid coordinator event is the reviewer's explicit request for the private value; "
            "after answering it, the next valid event is the completed reviewer report."
        )
    return f" {invalidity} {next_event}"


def _causal_n1_curriculum_episode(
    rung: Literal[
        "natural_n1a",
        "natural_n1a_local",
        "natural_n1b",
        "natural_direct_control",
    ],
    split: Split,
    index: int,
    seed: int,
    rng: random.Random,
    style: str,
    child_name: str,
    private_payload_mode: Literal["raw_resource", "finding_card"],
) -> dict[str, Any]:
    scenarios = NATURAL_SCENARIOS[split]
    scenario = scenarios[index % len(scenarios)]
    styles = STYLES[split]
    style = styles[(index + index // len(scenarios)) % len(styles)]
    root = _root(split, index, rng)
    remote = _pick_resource(split, rng, root, "review", integer=True)
    if private_payload_mode == "finding_card":
        remote = Resource(
            family=remote.family,
            path=remote.path,
            content=json.dumps({scenario.finding_key: int(remote.result)}),
            result=remote.result,
            operation=(
                f"report the integer stored under {scenario.finding_key} in the private "
                "evidence card"
            ),
        )

    if rung == "natural_direct_control":
        finding = int(remote.result)
        parameter = rng.randint(2, 19)
        answer = {
            scenario.finding_key: finding,
            scenario.parameter_key: parameter,
            scenario.result_key: finding + parameter,
        }
        prompt = (
            f"{_natural_intro(style, scenario)} The complete coordinator-owned record is "
            f"already present in this request: {scenario.finding_key}={finding} and "
            f"{scenario.parameter_key}={parameter}. No external evidence, separate reviewer, "
            f"or later interaction is needed. Publish {scenario.result_key} as their sum. "
            f"Return {_natural_schema(scenario.finding_key, scenario.parameter_key, scenario.result_key)}."
        )
        oracle = {
            "expected_route": "direct",
            "final_answer": answer,
            "coordinator_state": {},
            "resource_ownership": {},
            "private_resources": {},
            "children": [],
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                ["final_answer"],
                ["spawn_child", "poll", "discover_child"],
                [],
                {"spawn_child": 0, "parent_to_child_message": 0},
            ),
        }
        row = _row(
            split,
            index,
            seed,
            rung,
            style,
            prompt,
            {},
            oracle,
            ["coordinator_local_compute", "nondelegation"],
            1,
            "single_turn_no_child_delegation",
        )
        row["generator_version"] = CAUSAL_N1_CURRICULUM_VERSION
        row["metadata"].update(
            {
                "curriculum_rung": rung,
                "natural_stage": "paired_control",
                "semantic_family": scenario.key,
                "graph_variant": "direct_coordinator_compute_control",
                "control_contract_variant": style,
                "private_payload_mode": "none",
            }
        )
        row["metadata"]["axis_signature"] = hashlib.sha256(
            json.dumps(row["metadata"], sort_keys=True).encode()
        ).hexdigest()[:16]
        return row

    files: dict[str, str] = {}
    private_resources = {remote.path: remote.content}
    ownership = {
        remote.path: {
            "owner": f"child:{child_name}",
            "family": remote.family,
            "operation": remote.operation,
        }
    }
    child = _child(child_name, remote)
    intro = _natural_intro(style, scenario)
    local = None

    if rung in {"natural_n1a", "natural_n1a_local"}:
        if rung == "natural_n1a_local":
            local = _pick_resource(split, rng, root, "coordinator", integer=True)
            files[local.path] = local.content
            ownership[local.path] = {
                "owner": "coordinator",
                "family": local.family,
                "operation": local.operation,
            }
        completed = int(remote.result) * 2 + (int(local.result) if local else 0)
        answer = {
            scenario.finding_key: int(remote.result),
            **({"local": int(local.result)} if local else {}),
            scenario.result_key: completed,
        }
        local_clause = ""
        required = [
            f"spawn:{child_name}",
            f"retain_handle:{child_name}",
            "yield",
            f"receive:{child_name}",
            "final_answer",
        ]
        ordering = [
            (f"spawn:{child_name}", "yield"),
            (f"retain_handle:{child_name}", "yield"),
            ("yield", f"receive:{child_name}"),
            (f"receive:{child_name}", "final_answer"),
        ]
        if local is not None:
            local_clause = (
                f" The coordinator separately owns {local.path} and must {local.operation} "
                f"while the reviewer works. Add that local finding to {scenario.result_key}."
            )
            required.insert(2, "coordinator_read_local")
            ordering.extend(
                [
                    (f"spawn:{child_name}", "coordinator_read_local"),
                    ("coordinator_read_local", "yield"),
                ]
            )
        prompt = (
            f"{intro} {child_name} is the designated {scenario.child_role} and the only "
            f"party that receives the private evidence packet identified as {remote.path}; "
            f"the coordinator does not possess a copy. The reviewer must {remote.operation}."
            f"{local_clause}"
            f"{_natural_control_boundary(style, 'natural_n1', has_local_work=local is not None)} "
            f"Publish {scenario.result_key} as twice the delivered {scenario.finding_key}"
            f"{' plus the local finding' if local else ''}. Return "
            f"{_natural_schema(scenario.finding_key, *(['local'] if local else []), scenario.result_key)}."
        )
        forbidden = [
            "coordinator_read_child_owned",
            "poll",
            "discover_child",
            "delegate_coordinator_owned",
        ]
        if local is None:
            forbidden.append("post_spawn_tool_before_child")
        contract = _contract(
            required,
            forbidden,
            ordering,
            {
                "spawn_child": 1,
                "child_result_message": 1,
                "parent_to_child_message": 0,
            },
        )
        natural_stage = "N1a"
        graph_variant = (
            "pure_async_child_with_local_work"
            if local is not None
            else "pure_async_child"
        )
        atoms = ["semantic_ownership", "asynchronous_delegation"]
        if local is not None:
            atoms.append("independent_local_work")
        timing = "one_causal_resume_cycle"
        difficulty = 2 if local is None else 3
        persistence_lease = None
        state = {}
    else:
        parameter = rng.randint(101, 997)
        lease_path = f"{root}/coordinator-intake.json"
        files[lease_path] = json.dumps({scenario.parameter_key: parameter})
        ownership[lease_path] = {
            "owner": "coordinator",
            "family": "expiring_intake_register",
            "operation": (
                f"capture {scenario.parameter_key} into durable coordinator state before "
                "reviewer assignment"
            ),
        }
        answer = {
            scenario.finding_key: int(remote.result),
            scenario.parameter_key: parameter,
            scenario.result_key: int(remote.result) + parameter,
        }
        prompt = (
            f"{intro} {child_name} is the designated {scenario.child_role} and the only "
            f"party that receives the private evidence packet identified as {remote.path}; "
            f"the coordinator does not possess a copy. The reviewer must {remote.operation}. "
            f"A one-use coordinator intake register at {lease_path} contains the required "
            f"{scenario.parameter_label} under key {scenario.parameter_key}. Its access lease "
            "closes permanently as soon as reviewer assignment begins, and the earlier "
            "conversation transcript is discarded when the reviewer returns. Capture the "
            "intake value in durable coordinator state before making that assignment. After the "
            f"delivered report, recover and use the captured value to publish "
            f"{scenario.result_key} as the reviewer finding plus the intake value."
            f"{_natural_control_boundary(style, 'natural_n1', has_local_work=False)} "
            f"Return {_natural_schema(scenario.finding_key, scenario.parameter_key, scenario.result_key)}."
        )
        contract = _contract(
            [
                "capture_state",
                "persistence_lease_closed",
                f"spawn:{child_name}",
                f"retain_handle:{child_name}",
                "yield",
                f"receive:{child_name}",
                "context_boundary",
                "reuse_captured_state",
                "final_answer",
            ],
            [
                "coordinator_read_child_owned",
                "poll",
                "discover_child",
                "delegate_coordinator_owned",
                "delegate_private_value",
                "post_spawn_tool_before_child",
            ],
            [
                ("capture_state", f"spawn:{child_name}"),
                (f"spawn:{child_name}", "persistence_lease_closed"),
                ("persistence_lease_closed", "yield"),
                (f"spawn:{child_name}", "yield"),
                (f"retain_handle:{child_name}", "yield"),
                ("yield", f"receive:{child_name}"),
                (f"spawn:{child_name}", "context_boundary"),
                ("context_boundary", "reuse_captured_state"),
                (f"receive:{child_name}", "reuse_captured_state"),
                ("reuse_captured_state", "final_answer"),
            ],
            {
                "spawn_child": 1,
                "child_result_message": 1,
                "parent_to_child_message": 0,
            },
        )
        natural_stage = "N1b"
        graph_variant = "context_boundary_persistence_then_async_child"
        atoms = [
            "semantic_ownership",
            "causal_persistence",
            "asynchronous_delegation",
        ]
        timing = "capture_then_context_loss_then_one_causal_resume_cycle"
        difficulty = 3
        persistence_lease = {
            "path": lease_path,
            "key": scenario.parameter_key,
            "expected_value": parameter,
        }
        state = {scenario.parameter_key: parameter}

    oracle = {
        "expected_route": rung,
        "final_answer": answer,
        "coordinator_state": state,
        "resource_ownership": ownership,
        "private_resources": private_resources,
        "children": [child],
        "fault_plan": {"type": "none"},
        "trajectory_contract": contract,
    }
    if persistence_lease is not None:
        oracle["persistence_lease"] = persistence_lease
    row = _row(
        split,
        index,
        seed,
        rung,
        style,
        prompt,
        files,
        oracle,
        atoms,
        difficulty,
        timing,
    )
    row["generator_version"] = CAUSAL_N1_CURRICULUM_VERSION
    row["metadata"].update(
        {
            "curriculum_rung": rung,
            "natural_stage": natural_stage,
            "semantic_family": scenario.key,
            "graph_variant": graph_variant,
            "control_contract_variant": style,
            "private_payload_mode": private_payload_mode,
        }
    )
    row["metadata"]["axis_signature"] = hashlib.sha256(
        json.dumps(row["metadata"], sort_keys=True).encode()
    ).hexdigest()[:16]
    return row


def _natural_curriculum_episode(
    rung: Literal["natural_n1", "natural_n2"],
    split: Split,
    index: int,
    seed: int,
    rng: random.Random,
    style: str,
    child_name: str,
    private_payload_mode: Literal["raw_resource", "finding_card"],
) -> dict[str, Any]:
    scenarios = NATURAL_SCENARIOS[split]
    scenario = scenarios[index % len(scenarios)]
    styles = STYLES[split]
    style = styles[(index + index // len(scenarios)) % len(styles)]
    root = _root(split, index, rng)
    remote = _pick_resource(split, rng, root, "review", integer=True)
    if private_payload_mode == "finding_card":
        remote = Resource(
            family=remote.family,
            path=remote.path,
            content=json.dumps({scenario.finding_key: int(remote.result)}),
            result=remote.result,
            operation=(
                f"report the integer stored under {scenario.finding_key} in the private "
                "evidence card"
            ),
        )
    parameter = rng.randint(101, 997)
    files: dict[str, str] = {}
    private_resources = {remote.path: remote.content}
    ownership = {
        remote.path: {
            "owner": f"child:{child_name}",
            "family": remote.family,
            "operation": remote.operation,
        }
    }
    state = {scenario.parameter_key: parameter}
    intro = _natural_intro(style, scenario)
    child = _child(child_name, remote)
    graph_variant = "child_plus_private_state"

    if rung == "natural_n1":
        local = None
        if (index + index // len(scenarios)) % 2:
            local = _pick_resource(split, rng, root, "coordinator", integer=True)
            files[local.path] = local.content
            ownership[local.path] = {
                "owner": "coordinator",
                "family": local.family,
                "operation": local.operation,
            }
            graph_variant = "child_plus_local_work_and_private_state"
        result = int(remote.result) + parameter
        answer = {
            scenario.finding_key: int(remote.result),
            scenario.parameter_key: parameter,
            scenario.result_key: result + (int(local.result) if local else 0),
        }
        local_clause = ""
        required = [
            "retain_state",
            f"spawn:{child_name}",
            f"retain_handle:{child_name}",
            "yield",
            f"receive:{child_name}",
            "final_answer",
        ]
        ordering = [
            ("retain_state", f"spawn:{child_name}"),
            (f"spawn:{child_name}", "yield"),
            (f"retain_handle:{child_name}", "yield"),
            ("yield", f"receive:{child_name}"),
            (f"receive:{child_name}", "final_answer"),
        ]
        if local is not None:
            local_clause = (
                f" The coordinator is responsible for {local.path} and must "
                f"{local.operation}. The two reviews are independent and time-sensitive. "
                f"Add that local finding to {scenario.result_key} as well."
            )
            required.insert(3, "coordinator_read_local")
            ordering.extend(
                [
                    (f"spawn:{child_name}", "coordinator_read_local"),
                    ("coordinator_read_local", "yield"),
                ]
            )
        schema = _natural_schema(
            scenario.finding_key,
            scenario.parameter_key,
            scenario.result_key,
        )
        prompt = (
            f"{intro} {child_name} is the designated {scenario.child_role} and the only "
            f"party that receives the private evidence packet identified as {remote.path}; "
            f"the coordinator does not possess a copy. Its finding comes from this job: "
            f"{remote.operation}. The coordinator holds {scenario.parameter_label}={parameter}, "
            f"which is not part of the child review.{local_clause}"
            f"{_natural_control_boundary(style, rung, has_local_work=local is not None)} "
            f"Publish {scenario.result_key} "
            f"as the child finding plus the coordinator-held value"
            f"{' and the local finding' if local else ''}. Return {schema}."
        )
        contract = _contract(
            required,
            [
                "coordinator_read_child_owned",
                "poll",
                "discover_child",
                "delegate_coordinator_owned",
                "delegate_private_value",
            ],
            ordering,
            {
                "spawn_child": 1,
                "child_result_message": 1,
                "parent_to_child_message": 0,
            },
        )
        timing = "one_natural_resume_cycle"
        atoms = ["semantic_ownership", "private_state", "asynchronous_delegation"]
        if local is not None:
            atoms.append("independent_local_work")
    else:
        local = None
        if split == "ood_gen" and (index + index // len(scenarios)) % 2:
            local = _pick_resource(split, rng, root, "coordinator", integer=True)
            files[local.path] = local.content
            ownership[local.path] = {
                "owner": "coordinator",
                "family": local.family,
                "operation": local.operation,
            }
        result = int(remote.result) * parameter
        answer = {
            scenario.finding_key: int(remote.result),
            scenario.parameter_key: parameter,
            scenario.result_key: result + (int(local.result) if local else 0),
        }
        child["message_contract"] = [
            f"after {scenario.milestone}, request the {scenario.parameter_label} from parent",
            f"after the reply, report {scenario.finding_key} and {scenario.result_key}",
        ]
        schema = _natural_schema(
            scenario.finding_key,
            scenario.parameter_key,
            scenario.result_key,
        )
        local_clause = ""
        if local is not None:
            local_clause = (
                f" Independently, the coordinator owns {local.path} and must "
                f"{local.operation} while the reviewer works. Add that local finding "
                f"to {scenario.result_key}."
            )
        prompt = (
            f"{intro} {child_name} is the designated {scenario.child_role} and the only "
            f"party that receives the private evidence packet identified as {remote.path}; "
            f"the coordinator does not possess a copy. It must {remote.operation}. The "
            f"coordinator holds {scenario.parameter_label}={parameter}. Separation of duties "
            f"requires that value to remain private until {scenario.milestone} and {child_name} "
            f"explicitly asks for it. The reviewer then applies it as a multiplier and reports "
            f"the completed {scenario.result_key}.{local_clause}"
            f"{_natural_control_boundary(style, rung, has_local_work=local is not None)} "
            f"Publish the reviewer's {scenario.finding_key}, "
            f"the coordinator-held {scenario.parameter_key}, and the completed result. Return "
            f"{schema}."
        )
        required = [
            "retain_state",
            f"spawn:{child_name}",
            f"retain_handle:{child_name}",
            "yield",
            f"receive_request:{child_name}",
            f"send_followup:{child_name}",
            f"send_followup_value:{child_name}",
            "yield_after_followup",
            f"receive_result:{child_name}",
            "final_answer",
        ]
        ordering = [
            ("retain_state", f"spawn:{child_name}"),
            (f"spawn:{child_name}", "yield"),
            (f"retain_handle:{child_name}", "yield"),
            ("yield", f"receive_request:{child_name}"),
            (f"receive_request:{child_name}", f"send_followup:{child_name}"),
            (f"send_followup:{child_name}", "yield_after_followup"),
            ("yield_after_followup", f"receive_result:{child_name}"),
            (f"receive_result:{child_name}", "final_answer"),
        ]
        graph_variant = "staged_private_parameter_cycle"
        if local is not None:
            required.insert(3, "coordinator_read_local")
            ordering.extend(
                [
                    (f"spawn:{child_name}", "coordinator_read_local"),
                    ("coordinator_read_local", "yield"),
                ]
            )
            graph_variant = "staged_private_parameter_cycle_with_local_work"
        contract = _contract(
            required,
            [
                "coordinator_read_child_owned",
                "poll",
                "discover_child",
                "guess_followup_value",
                "wrong_followup_value",
                "delegate_private_value",
            ],
            ordering,
            {
                "spawn_child": 1,
                "parent_to_child_message": 1,
                "child_to_parent_message": 2,
            },
        )
        timing = "two_natural_resume_cycles"
        atoms = [
            "semantic_ownership",
            "private_state",
            "staged_dependency",
            "request_reply",
        ]
        if local is not None:
            atoms.append("independent_local_work")

    oracle = {
        "expected_route": rung,
        "final_answer": answer,
        "coordinator_state": state,
        "resource_ownership": ownership,
        "private_resources": private_resources,
        "children": [child],
        "request_terms": [scenario.parameter_label, scenario.parameter_key],
        "fault_plan": {"type": "none"},
        "trajectory_contract": contract,
    }
    row = _row(
        split,
        index,
        seed,
        rung,
        style,
        prompt,
        files,
        oracle,
        atoms,
        2 if rung == "natural_n1" else 3,
        timing,
    )
    row["generator_version"] = CURRICULUM_VERSION
    row["metadata"].update(
        {
            "curriculum_rung": rung,
            "natural_stage": "N1" if rung == "natural_n1" else "N2",
            "semantic_family": scenario.key,
            "graph_variant": graph_variant,
            "control_contract_variant": style,
            "private_payload_mode": private_payload_mode,
        }
    )
    row["metadata"]["axis_signature"] = hashlib.sha256(
        json.dumps(row["metadata"], sort_keys=True).encode()
    ).hexdigest()[:16]
    return row


def _row(
    split: Split,
    index: int,
    seed: int,
    family: str,
    style: str,
    prompt: str,
    files: dict[str, str],
    oracle: dict[str, Any],
    atoms: list[str],
    difficulty: int,
    timing: str,
    fault: str = "none",
) -> dict[str, Any]:
    metadata = {
        "episode_family": family,
        "instruction_style": style,
        "resource_families": [
            x["family"] for x in oracle["resource_ownership"].values()
        ],
        "composition_atoms": atoms,
        "fault_mode": fault,
        "timing_regime": timing,
        "difficulty": difficulty,
    }
    metadata["axis_signature"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()[:16]
    eid = f"{split}-{family}-{index:08d}-{seed & 0xFFFFFFF:07x}"
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "episode_id": eid,
        "split": split,
        "index": index,
        "seed": seed,
        "public": {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": prompt,
            "workspace_files": files,
        },
        "oracle": oracle,
        "metadata": metadata,
    }


def generate_curriculum_episode(
    rung: CurriculumRung,
    split: Split,
    index: int,
    master_seed: int = 20260816,
    private_payload_mode: Literal["raw_resource", "finding_card"] = "raw_resource",
) -> dict[str, Any]:
    raw_seed = (
        f"{CURRICULUM_SEED_VERSION}|{master_seed}|{rung}|{split}|{index}".encode()
    )
    seed = int.from_bytes(hashlib.sha256(raw_seed).digest()[:8], "big")
    rng = random.Random(seed)
    style = rng.choice(STYLES[split])
    names = rng.sample(list(CHILD_NAMES[split]), 2)
    children: list[dict[str, Any]] = []

    if rung in {
        "natural_n1a",
        "natural_n1a_local",
        "natural_n1b",
        "natural_direct_control",
    }:
        return _causal_n1_curriculum_episode(
            rung,
            split,
            index,
            seed,
            rng,
            style,
            names[0],
            private_payload_mode,
        )
    if rung in {"natural_n1", "natural_n2"}:
        return _natural_curriculum_episode(
            rung,
            split,
            index,
            seed,
            rng,
            style,
            names[0],
            private_payload_mode,
        )

    if rung == "atomic_state":
        state_name = rng.choice(STATE_NAMES[split])
        state_value = rng.randint(2, 19)
        increment = rng.randint(2, 9)
        prompt = (
            f"Use persistent IPython state across exactly two calls. In the first call, assign {state_name}={state_value} and do nothing else. "
            f"In a later IPython call, read that retained variable, add {increment}, and print the result. Do not delegate, poll, or combine the two calls. "
            f"In the final JSON, marker must be the original retained value and result must be the printed sum. Return {SCHEMAS[rung]}."
        )
        oracle = {
            "expected_route": "direct",
            "final_answer": {"marker": state_value, "result": state_value + increment},
            "coordinator_state": {state_name: state_value},
            "resource_ownership": {},
            "children": [],
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                ["retain_state", f"reuse_state:{state_name}", "final_answer"],
                ["spawn_child", "poll", "discover_child"],
                [
                    ("retain_state", f"reuse_state:{state_name}"),
                    (f"reuse_state:{state_name}", "final_answer"),
                ],
                {"spawn_child": 0, "parent_to_child_message": 0},
            ),
        }
        atoms = ["persistent_state", "later_cell_reuse", "nondelegation"]
        timing = "two_ipython_cells"
    elif rung == "atomic_send":
        left, right = rng.randint(2, 19), rng.randint(2, 19)
        value = left + right
        children.append(
            {
                "name": names[0],
                "resource_path": None,
                "operation": f"compute {left} + {right}",
                "expected_result": value,
                "message_contract": "send result once to parent",
            }
        )
        prompt = (
            f"{names[0]} owns the calculation {left} + {right}. Spawn exactly that named child with a self-contained instruction to compute it and execute an awaited "
            "agent_message.send to parent with the integer result. Retain the child handle, end the turn after spawning, and react only to its explicit message. "
            f"Never poll, sleep, observe, or discover children. Return {SCHEMAS[rung]}."
        )
        required = [
            f"spawn:{names[0]}",
            f"retain_handle:{names[0]}",
            "yield",
            f"receive:{names[0]}",
            "final_answer",
        ]
        oracle = {
            "expected_route": "atomic_send",
            "final_answer": {"value": value},
            "resource_ownership": {},
            "children": children,
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                required,
                ["poll", "discover_child"],
                [
                    (f"spawn:{names[0]}", "yield"),
                    (f"retain_handle:{names[0]}", "yield"),
                    ("yield", f"receive:{names[0]}"),
                    (f"receive:{names[0]}", "final_answer"),
                ],
                {
                    "spawn_child": 1,
                    "child_result_message": 1,
                    "parent_to_child_message": 0,
                },
            ),
        }
        atoms = ["spawn", "retain_handle", "yield", "child_to_parent"]
        timing = "one_resume_cycle"
    elif rung == "atomic_child_request":
        multiplier = rng.randint(2, 19)
        children.append(
            {
                "name": names[0],
                "resource_path": None,
                "operation": "request the retained multiplier",
                "expected_result": multiplier,
                "message_contract": "send 'need multiplier' once to parent",
            }
        )
        prompt = (
            f"Retain multiplier={multiplier} in coordinator state. Spawn exactly one "
            f"named child, {names[0]}, with a self-contained instruction to execute "
            "`await agent_message.send('need multiplier', receiver_role='parent')` "
            "exactly once and then stop. Put that instruction directly in the initial "
            "rlm prompt, retain the returned child handle, and end the waiting turn "
            "immediately. Never poll, sleep, observe, discover children, or send a "
            "message to the child. After the explicit request arrives, return "
            f"{SCHEMAS[rung]}."
        )
        required = [
            "retain_state",
            f"spawn:{names[0]}",
            f"retain_handle:{names[0]}",
            "yield",
            f"receive_request:{names[0]}",
            "final_answer",
        ]
        ordering = [
            ("retain_state", f"spawn:{names[0]}"),
            (f"spawn:{names[0]}", "yield"),
            (f"retain_handle:{names[0]}", "yield"),
            ("yield", f"receive_request:{names[0]}"),
            (f"receive_request:{names[0]}", "final_answer"),
        ]
        oracle = {
            "expected_route": "atomic_child_request",
            "final_answer": {"multiplier": multiplier, "request_received": True},
            "coordinator_state": {"multiplier": multiplier},
            "resource_ownership": {},
            "children": children,
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                required,
                ["poll", "discover_child", f"send_followup:{names[0]}"],
                ordering,
                {
                    "spawn_child": 1,
                    "parent_to_child_message": 0,
                    "child_to_parent_message": 1,
                },
            ),
        }
        atoms = [
            "state",
            "spawn_prompt",
            "retain_handle",
            "yield",
            "child_request",
        ]
        timing = "request_prefix"
    elif rung == "atomic_followup":
        multiplier = rng.randint(2, 19)
        children.append(
            {
                "name": names[0],
                "resource_path": None,
                "operation": "request and echo the retained multiplier",
                "expected_result": multiplier,
                "message_contract": [
                    "send 'need multiplier' to parent",
                    "after reply send multiplier as result",
                ],
            }
        )
        prompt = (
            f"Retain multiplier={multiplier} in coordinator state. Spawn exactly one named child, {names[0]}, instructing it to execute an awaited agent_message.send "
            "asking for the multiplier, then after your reply execute a second awaited send with that integer. End each waiting turn immediately; never poll, sleep, "
            f"observe, or discover children. Return {SCHEMAS[rung]} only after the second explicit child message."
        )
        required = [
            "retain_state",
            f"spawn:{names[0]}",
            f"retain_handle:{names[0]}",
            "yield",
            f"receive_request:{names[0]}",
            f"send_followup:{names[0]}",
            "yield_after_followup",
            f"receive_result:{names[0]}",
            "final_answer",
        ]
        ordering = [
            ("retain_state", f"spawn:{names[0]}"),
            (f"spawn:{names[0]}", "yield"),
            (f"retain_handle:{names[0]}", "yield"),
            ("yield", f"receive_request:{names[0]}"),
            (f"receive_request:{names[0]}", f"send_followup:{names[0]}"),
            (f"send_followup:{names[0]}", "yield_after_followup"),
            ("yield_after_followup", f"receive_result:{names[0]}"),
            (f"receive_result:{names[0]}", "final_answer"),
        ]
        oracle = {
            "expected_route": "atomic_followup",
            "final_answer": {"multiplier": multiplier, "result": multiplier},
            "coordinator_state": {"multiplier": multiplier},
            "resource_ownership": {},
            "children": children,
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                required,
                ["poll", "discover_child", "guess_followup_value"],
                ordering,
                {
                    "spawn_child": 1,
                    "parent_to_child_message": 1,
                    "child_to_parent_message": 2,
                },
            ),
        }
        atoms = [
            "state",
            "spawn",
            "yield",
            "child_to_parent",
            "parent_to_child",
            "resume",
        ]
        timing = "two_resume_cycles"
    elif rung == "atomic_parallel":
        values = [rng.randint(2, 19), rng.randint(2, 19)]
        for name, value in zip(names, values, strict=True):
            children.append(
                {
                    "name": name,
                    "resource_path": None,
                    "operation": f"return the owned integer {value}",
                    "expected_result": value,
                    "message_contract": "send result once to parent",
                }
            )
        prompt = (
            f"{names[0]} owns integer {values[0]}; {names[1]} owns integer {values[1]}. Spawn both named children in separate calls before waiting. Each child must execute "
            "an awaited agent_message.send to parent with its integer. Retain both handles, end the turn, and never poll, sleep, observe, or discover children. "
            f"Return {SCHEMAS[rung]} after both explicit messages."
        )
        required = (
            [
                f"spawn:{names[0]}",
                f"spawn:{names[1]}",
                f"retain_handle:{names[0]}",
                f"retain_handle:{names[1]}",
                "yield",
            ]
            + [f"receive:{name}" for name in names]
            + ["final_answer"]
        )
        ordering = (
            [(f"spawn:{name}", "yield") for name in names]
            + [("yield", f"receive:{name}") for name in names]
            + [(f"receive:{name}", "final_answer") for name in names]
        )
        oracle = {
            "expected_route": "atomic_parallel",
            "final_answer": {
                "alpha": values[0],
                "beta": values[1],
                "result": sum(values),
            },
            "resource_ownership": {},
            "children": children,
            "fault_plan": {"type": "none", "delivery_order": list(reversed(names))},
            "trajectory_contract": _contract(
                required,
                ["poll", "discover_child", "serialized_fanout_wait"],
                ordering,
                {
                    "spawn_child": 2,
                    "child_result_message": 2,
                    "parent_to_child_message": 0,
                },
            ),
        }
        atoms = [
            "parallel",
            "spawn",
            "retain_handle",
            "yield",
            "child_to_parent",
            "fanin",
        ]
        timing = "reverse_delivery"
    else:
        raise ValueError(f"unknown curriculum rung {rung}")

    row = _row(split, index, seed, rung, style, prompt, {}, oracle, atoms, 1, timing)
    row["generator_version"] = CURRICULUM_VERSION
    row["metadata"]["curriculum_rung"] = rung
    row["metadata"]["axis_signature"] = hashlib.sha256(
        json.dumps(row["metadata"], sort_keys=True).encode()
    ).hexdigest()[:16]
    return row


def generate_episode(
    split: Split, index: int, master_seed: int = 20260816
) -> dict[str, Any]:
    seed = _seed(master_seed, split, index)
    rng = random.Random(seed)
    root = _root(split, index, rng)
    family = (OOD_FAMILIES if split == "ood_gen" else TRAIN_FAMILIES)[
        index % (8 if split == "ood_gen" else 6)
    ]
    style = rng.choice(STYLES[split])
    state = (rng.choice(STATE_NAMES[split]), rng.randint(2, 17))
    names = rng.sample(list(CHILD_NAMES[split]), 3)
    own: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}
    children: list[dict[str, Any]] = []

    if family == "direct":
        r = _pick_resource(split, rng, root, "local")
        files[r.path] = r.content
        own[r.path] = {
            "owner": "coordinator",
            "family": r.family,
            "operation": r.operation,
        }
        prompt = f"{_header(style, None)} Coordinator owns {r.path}; {r.operation} locally. Do not delegate. Return {SCHEMAS[family]}."
        oracle = {
            "expected_route": "direct",
            "final_answer": {"result": r.result},
            "resource_ownership": own,
            "children": [],
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                ["coordinator_read_local", "final_answer"],
                ["spawn_child", "poll", "discover_child"],
                [("coordinator_read_local", "final_answer")],
                {"spawn_child": 0},
            ),
        }
        return _row(
            split,
            index,
            seed,
            family,
            style,
            prompt,
            files,
            oracle,
            ["direct", "nondelegation"],
            1,
            "immediate",
        )

    if family in ("parallel", "triple"):
        width = 3 if family == "triple" else 2
        rs = [_pick_resource(split, rng, root, f"remote-{i}") for i in range(width)]
        for name, r in zip(names[:width], rs, strict=True):
            files[r.path] = r.content
            own[r.path] = {
                "owner": f"child:{name}",
                "family": r.family,
                "operation": r.operation,
            }
            children.append(_child(name, r))
        offset = state[1]
        result = _combine([r.result for r in rs], offset)
        labels = ("alpha", "beta", "gamma")
        answer = {labels[i]: rs[i].result for i in range(width)} | {
            "offset": offset,
            "result": result,
        }
        assignments = " ".join(
            f"{n} owns {r.path} and must {r.operation}."
            for n, r in zip(names[:width], rs, strict=True)
        )
        prompt = f"{_header(style, state)} {assignments} Spawn all independent children before waiting; do not inspect their resources or poll. Combine their explicit results with {state[0]}. Return {SCHEMAS[family]}."
        required = (
            ["retain_state", "yield", "final_answer"]
            + [f"spawn:{n}" for n in names[:width]]
            + [f"receive:{n}" for n in names[:width]]
        )
        ordering = (
            [("retain_state", f"spawn:{n}") for n in names[:width]]
            + [(f"spawn:{n}", "yield") for n in names[:width]]
            + [("yield", f"receive:{n}") for n in names[:width]]
            + [(f"receive:{n}", "final_answer") for n in names[:width]]
        )
        oracle = {
            "expected_route": "parallel",
            "final_answer": answer,
            "coordinator_state": {state[0]: offset},
            "resource_ownership": own,
            "children": children,
            "fault_plan": {
                "type": "none",
                "delivery_order": list(reversed(names[:width])),
            },
            "trajectory_contract": _contract(
                required,
                [
                    "coordinator_read_child_owned",
                    "poll",
                    "discover_child",
                    "serialized_fanout_wait",
                ],
                ordering,
                {"spawn_child": width, "child_result_message": width},
            ),
        }
        return _row(
            split,
            index,
            seed,
            family,
            style,
            prompt,
            files,
            oracle,
            ["ownership", "parallel", "yield", "fanin"],
            4 if width == 3 else 3,
            "reverse_delivery",
        )

    remote = _pick_resource(split, rng, root, "remote", integer=family == "followup")
    files[remote.path] = remote.content
    own[remote.path] = {
        "owner": f"child:{names[0]}",
        "family": remote.family,
        "operation": remote.operation,
    }

    if family == "single":
        offset = state[1]
        answer = {
            "child": remote.result,
            "offset": offset,
            "result": _combine([remote.result], offset),
        }
        children = [_child(names[0], remote)]
        prompt = f"{_header(style, state)} {names[0]} owns {remote.path} and must {remote.operation}. Never inspect it in coordinator code. Retain the child handle, yield instead of polling, combine the explicit result with {state[0]}, and return {SCHEMAS[family]}."
        req = [
            "retain_state",
            f"spawn:{names[0]}",
            "retain_handle",
            "yield",
            f"receive:{names[0]}",
            "final_answer",
        ]
        order = [
            ("retain_state", f"spawn:{names[0]}"),
            (f"spawn:{names[0]}", "yield"),
            ("yield", f"receive:{names[0]}"),
            (f"receive:{names[0]}", "final_answer"),
        ]
        oracle = {
            "expected_route": "single",
            "final_answer": answer,
            "coordinator_state": {state[0]: offset},
            "resource_ownership": own,
            "children": children,
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                req,
                [
                    "coordinator_read_child_owned",
                    "poll",
                    "discover_child",
                    "delegate_coordinator_state",
                ],
                order,
                {"spawn_child": 1, "child_result_message": 1},
            ),
        }
        return _row(
            split,
            index,
            seed,
            family,
            style,
            prompt,
            files,
            oracle,
            ["ownership", "single_child", "yield", "fanin"],
            2,
            "child_message",
        )

    if family == "mixed":
        local = _pick_resource(split, rng, root, "local")
        files[local.path] = local.content
        own[local.path] = {
            "owner": "coordinator",
            "family": local.family,
            "operation": local.operation,
        }
        offset = state[1]
        answer = {
            "local": local.result,
            "child": remote.result,
            "offset": offset,
            "result": _combine([local.result, remote.result], offset),
        }
        children = [_child(names[0], remote)]
        prompt = f"{_header(style, state)} Coordinator owns {local.path} and must {local.operation}. {names[0]} owns {remote.path} and must {remote.operation}. Do local work while the child runs; never inspect the child-owned resource. Return {SCHEMAS[family]}."
        req = [
            "retain_state",
            f"spawn:{names[0]}",
            "coordinator_read_local",
            "yield",
            f"receive:{names[0]}",
            "final_answer",
        ]
        order = [
            ("retain_state", f"spawn:{names[0]}"),
            (f"spawn:{names[0]}", "coordinator_read_local"),
            ("coordinator_read_local", "yield"),
            ("yield", f"receive:{names[0]}"),
            (f"receive:{names[0]}", "final_answer"),
        ]
        oracle = {
            "expected_route": "mixed",
            "final_answer": answer,
            "coordinator_state": {state[0]: offset},
            "resource_ownership": own,
            "children": children,
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                req,
                [
                    "coordinator_read_child_owned",
                    "poll",
                    "discover_child",
                    "delegate_coordinator_owned",
                ],
                order,
                {"spawn_child": 1, "child_result_message": 1},
            ),
        }
        return _row(
            split,
            index,
            seed,
            family,
            style,
            prompt,
            files,
            oracle,
            ["direct", "ownership", "single_child", "concurrency", "fanin"],
            3,
            "local_while_child_runs",
        )

    if family == "followup":
        multiplier = state[1]
        subtotal = int(remote.result)
        answer = {
            "subtotal": subtotal,
            "multiplier": multiplier,
            "result": subtotal * multiplier,
        }
        children = [
            _child(
                names[0],
                remote,
                [
                    "send 'need multiplier' to parent",
                    "after reply send subtotal and result",
                ],
            )
        ]
        prompt = f"{_header(style, state)} {names[0]} owns {remote.path}; it must {remote.operation}, retain the subtotal, then ask you for the multiplier. Reply from retained state, yield again, and wait for its explicit final message. Never inspect the child file or poll. Return {SCHEMAS[family]}."
        req = [
            "retain_state",
            f"spawn:{names[0]}",
            "yield",
            f"receive_request:{names[0]}",
            f"send_followup:{names[0]}",
            "yield_after_followup",
            f"receive_result:{names[0]}",
            "final_answer",
        ]
        order = [(req[i], req[i + 1]) for i in range(len(req) - 1)]
        oracle = {
            "expected_route": "followup",
            "final_answer": answer,
            "coordinator_state": {state[0]: multiplier},
            "resource_ownership": own,
            "children": children,
            "fault_plan": {"type": "none"},
            "trajectory_contract": _contract(
                req,
                [
                    "coordinator_read_child_owned",
                    "poll",
                    "discover_child",
                    "guess_followup_value",
                ],
                order,
                {
                    "spawn_child": 1,
                    "parent_to_child_message": 1,
                    "child_to_parent_message": 2,
                },
            ),
        }
        return _row(
            split,
            index,
            seed,
            family,
            style,
            prompt,
            files,
            oracle,
            [
                "ownership",
                "single_child",
                "yield",
                "child_to_parent",
                "parent_to_child",
                "resume",
                "fanin",
            ],
            4,
            "two_resume_cycles",
        )

    if family == "verify":
        manifest = root + "/verification.json"
        digest = hashlib.sha256(remote.content.encode()).hexdigest()[:12]
        files[manifest] = json.dumps({"expected_digest": digest})
        own[manifest] = {
            "owner": "coordinator",
            "family": "verification_manifest",
            "operation": "compare child digest to expected_digest",
        }
        children = [
            _child(names[0], remote, "send JSON with result and 12-char SHA-256 digest")
        ]
        prompt = f"{_header(style, None)} {names[0]} owns {remote.path} and must {remote.operation} plus report its 12-char SHA-256 digest. Coordinator owns {manifest}; verify before accepting. The digest is verification evidence only: after it matches, set both child and result to the child's computed resource result, and do not put the digest in the final JSON. Never inspect the child resource. Return {SCHEMAS[family]}."
        req = [
            f"spawn:{names[0]}",
            "coordinator_read_verification_manifest",
            "yield",
            f"receive:{names[0]}",
            "verify_child_digest",
            "final_answer",
        ]
        order = [(req[i], req[i + 1]) for i in range(len(req) - 1)]
        oracle = {
            "expected_route": "single_verify",
            "final_answer": {
                "child": remote.result,
                "verified": True,
                "result": remote.result,
            },
            "resource_ownership": own,
            "children": children,
            "fault_plan": {"type": "none", "on_mismatch": "request one correction"},
            "trajectory_contract": _contract(
                req,
                [
                    "coordinator_read_child_owned",
                    "poll",
                    "accept_unverified_child_result",
                ],
                order,
                {"spawn_child": 1},
            ),
        }
        return _row(
            split,
            index,
            seed,
            family,
            style,
            prompt,
            files,
            oracle,
            ["ownership", "single_child", "verification", "fanin"],
            4,
            "verify_after_message",
        )

    if family == "reclaim":
        children = [_child(names[0], remote)]
        prompt = f"{_header(style, None)} {names[0]} initially owns {remote.path} and should {remote.operation}. Do not inspect it while the child is healthy. If an explicit child failure arrives, reclaim once, compute locally, and return {SCHEMAS[family]}."
        req = [
            f"spawn:{names[0]}",
            "yield",
            f"receive_failure:{names[0]}",
            "explicit_reclaim",
            "coordinator_read_after_reclaim",
            "final_answer",
        ]
        order = [(req[i], req[i + 1]) for i in range(len(req) - 1)]
        fault = {
            "type": "inject_child_failure",
            "after": f"spawn:{names[0]}",
            "message": "RESOURCE_UNAVAILABLE",
            "ownership_transition": {
                "from": f"child:{names[0]}",
                "to": "coordinator",
                "trigger": "explicit_child_failure",
            },
        }
        oracle = {
            "expected_route": "reclaim",
            "final_answer": {"reclaimed": True, "result": remote.result},
            "resource_ownership": own,
            "children": children,
            "fault_plan": fault,
            "trajectory_contract": _contract(
                req,
                [
                    "coordinator_read_child_owned_before_reclaim",
                    "poll",
                    "reclaim_without_failure",
                    "respawn_same_failed_child",
                ],
                order,
                {"spawn_child": 1, "reclaim": 1},
            ),
        }
        return _row(
            split,
            index,
            seed,
            family,
            style,
            prompt,
            files,
            oracle,
            ["ownership", "single_child", "failure", "reclaim", "direct"],
            5,
            "failure_then_reclaim",
            "inject_child_failure",
        )

    raise AssertionError(family)


def validate_row(row: dict[str, Any]) -> None:
    if row["schema_version"] != SCHEMA_VERSION or "reasoning_content" in json.dumps(
        row
    ):
        raise ValueError("invalid schema or reasoning_content")
    public, oracle = row["public"], row["oracle"]
    files, own = public["workspace_files"], oracle["resource_ownership"]
    private = oracle.get("private_resources", {})
    if set(files) & set(private):
        raise ValueError("resource cannot be both public and private")
    if set(files) | set(private) != set(own):
        raise ValueError("workspace/ownership path mismatch")
    contract = oracle["trajectory_contract"]
    if set(contract["required_atoms"]) & set(contract["forbidden_atoms"]):
        raise ValueError("contradictory contract")
    if oracle["expected_route"] == "direct" and oracle["children"]:
        raise ValueError("direct route has children")
    if oracle["expected_route"] != "direct" and not oracle["children"]:
        raise ValueError("delegated route lacks children")
    is_curriculum = "curriculum_rung" in row["metadata"]
    if row["metadata"].get("natural_stage"):
        prompt = public["user_prompt"].lower()
        forbidden = [term for term in NATURAL_USER_PROMPT_FORBIDDEN if term in prompt]
        if forbidden:
            raise ValueError(
                f"natural user prompt prescribes harness actions: {forbidden}"
            )
        if not row["metadata"].get("semantic_family") or not row["metadata"].get(
            "graph_variant"
        ):
            raise ValueError("natural curriculum lacks structural metadata")
    for child in oracle["children"]:
        path = child["resource_path"]
        if path is None and is_curriculum:
            continue
        if (path not in files and path not in private) or not own[path][
            "owner"
        ].startswith("child:"):
            raise ValueError("invalid child resource ownership")
    if not all(contract["hard_gate"].values()):
        raise ValueError("hard gate must be conjunctive")


def materialize(
    output: Path,
    train_count: int = 4096,
    valid_count: int = 512,
    ood_count: int = 512,
    train_start: int = 0,
    master_seed: int = 20260816,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "procedural-harness-master-v1/manifest/v1",
        "generator_version": GENERATOR_VERSION,
        "master_seed": master_seed,
        "train_stream": {
            "start_index": train_start,
            "count_materialized": train_count,
            "index_space": "non-negative integers",
        },
        "frozen_eval": {
            "valid_gen": {"start_index": 0, "count": valid_count},
            "ood_gen": {"start_index": 0, "count": ood_count},
        },
        "splits": {},
    }
    for split, start, count in (
        ("train_gen", train_start, train_count),
        ("valid_gen", 0, valid_count),
        ("ood_gen", 0, ood_count),
    ):
        rows = [
            generate_episode(split, i, master_seed) for i in range(start, start + count)
        ]
        for row in rows:
            validate_row(row)
        text = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        file = output / f"{split}.jsonl"
        file.write_text(text)
        meta = [r["metadata"] for r in rows]
        manifest["splits"][split] = {
            "episodes": count,
            "file": file.name,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "episode_families": sorted({m["episode_family"] for m in meta}),
            "instruction_styles": sorted({m["instruction_style"] for m in meta}),
            "resource_families": sorted(
                {
                    f
                    for m in meta
                    for f in m["resource_families"]
                    if f != "verification_manifest"
                }
            ),
            "fault_modes": sorted({m["fault_mode"] for m in meta}),
            "axis_signatures": len({m["axis_signature"] for m in meta}),
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--train-count", type=int, default=4096)
    p.add_argument("--valid-count", type=int, default=512)
    p.add_argument("--ood-count", type=int, default=512)
    p.add_argument("--train-start", type=int, default=0)
    p.add_argument("--master-seed", type=int, default=20260816)
    a = p.parse_args()
    if min(a.train_count, a.valid_count, a.ood_count) <= 0 or a.train_start < 0:
        raise SystemExit("counts must be positive and train-start non-negative")
    print(
        json.dumps(
            materialize(
                a.output,
                a.train_count,
                a.valid_count,
                a.ood_count,
                a.train_start,
                a.master_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
