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

SYSTEM_PROMPT = (
    "Coordinate through Prime Agent's persistent IPython kernel. Solve directly when "
    "the coordinator owns the work; delegate only explicitly child-owned resources. "
    "Preserve local state and child handles, spawn independent children before waiting, "
    "yield instead of polling, and treat visible child messages as the completion channel. "
    "Never inspect child-owned resources before an explicit failure and reclaim. Verify "
    "child results when coordinator-owned evidence exists. Return exactly the requested JSON."
)

TRAIN_RESOURCES = (
    "json_sum", "csv_total", "word_count", "md_h2", "log_error",
    "python_defs", "json_max", "sha_prefix",
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
    "train_gen": ("alpha-worker", "beta-worker", "ledger-worker", "table-worker", "relay-worker", "audit-worker"),
    "valid_gen": ("north-worker", "south-worker", "delta-worker", "proof-worker", "signal-worker"),
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
}


@dataclass(frozen=True)
class Resource:
    family: str
    path: str
    content: str
    result: int | str
    operation: str


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
        path = stem + ".json"; values = [rng.randint(-20, 45) for _ in range(7)]
        content = json.dumps(values); result = sum(values); op = "sum the top-level JSON integer list"
    elif family == "csv_total":
        path = stem + ".csv"; rows = [{"id": i, "amount": rng.randint(2, 95)} for i in range(6)]
        buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=["id", "amount"])
        writer.writeheader(); writer.writerows(rows); content = buf.getvalue()
        result = sum(row["amount"] for row in rows); op = "sum the CSV amount column"
    elif family == "word_count":
        path = stem + ".txt"; keyword = rng.choice(("retry", "stable", "green"))
        words = [rng.choice(("ready", keyword, "done", "wait", keyword)) for _ in range(18)]
        content = " ".join(words); result = words.count(keyword); op = f"count exact {keyword!r} tokens"
    elif family == "md_h2":
        path = stem + ".md"; result = rng.randint(3, 7)
        content = "# Report\n\n" + "\n".join(f"## S{i}\nbody" for i in range(result)) + "\n"
        op = "count level-2 Markdown headings"
    elif family == "log_error":
        path = stem + ".log"; levels = [rng.choice(("INFO", "WARN", "ERROR")) for _ in range(16)]
        content = "\n".join(f"{x} event-{i}" for i, x in enumerate(levels)) + "\n"
        result = levels.count("ERROR"); op = "count ERROR-level log lines"
    elif family == "python_defs":
        path = stem + ".py"; n, m = rng.randint(2, 4), rng.randint(1, 3)
        content = "\n".join([f"def f{i}():\n return {i}" for i in range(n)] + [f"async def a{i}():\n return {i}" for i in range(m)])
        tree = ast.parse(content); result = sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) for x in tree.body)
        op = "count top-level sync and async function definitions"
    elif family == "json_max":
        path = stem + ".json"; values = {f"m{i}": rng.randint(-30, 120) for i in range(7)}
        content = json.dumps(values); result = max(values.values()); op = "return the largest JSON integer value"
    elif family == "sha_prefix":
        path = stem + ".bin"; content = f"payload:{rng.getrandbits(96):024x}:{slot}"
        result = hashlib.sha256(content.encode()).hexdigest()[:8]; op = "return the first eight SHA-256 hex characters"
    elif family == "tsv_total":
        path = stem + ".tsv"; scores = [rng.randint(3, 80) for _ in range(7)]
        content = "name\tscore\n" + "\n".join(f"n{i}\t{x}" for i, x in enumerate(scores)) + "\n"
        result = sum(scores); op = "sum the TSV score column"
    elif family == "xml_items":
        path = stem + ".xml"; n = rng.randint(4, 9)
        content = "<root>" + "".join(f"<item id='{i}'/>" for i in range(n)) + "</root>"
        result = len(ET.fromstring(content).findall("item")); op = "count XML item elements"
    elif family == "jsonl_active_sum":
        path = stem + ".jsonl"; rows = [{"active": rng.choice((True, False)), "value": rng.randint(-9, 35)} for _ in range(10)]
        content = "\n".join(json.dumps(x) for x in rows) + "\n"
        result = sum(x["value"] for x in rows if x["active"]); op = "sum value for JSONL records with active=true"
    elif family == "ini_quota_sum":
        path = stem + ".ini"; quotas = [rng.randint(1, 25) for _ in range(5)]
        content = "\n".join(f"[worker{i}]\nquota={q}\n" for i, q in enumerate(quotas))
        parser = configparser.ConfigParser(); parser.read_string(content)
        result = sum(parser.getint(s, "quota") for s in parser.sections()); op = "sum quota across INI sections"
    else:
        raise ValueError(f"unknown resource family {family}")
    return Resource(family, path, content, result, op)


def _pick_resource(split: Split, rng: random.Random, root: str, slot: str, *, integer: bool = False) -> Resource:
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
            "explicit": "Execute this ownership-aware task.", "natural_a": "Handle this job under the stated responsibility split.",
            "natural_b": "Work this request without crossing resource boundaries.", "natural_c": "Coordinate according to the ownership assignments.",
            "compact": "Follow ownership.", "terse": "Respect ownership.", "narrative": "You are coordinating a job with explicit resource owners.",
        }[style]
    name, value = state
    return {
        "explicit": f"Retain {name}={value} in coordinator state.", "natural_a": f"Keep {name}={value} locally while you coordinate.",
        "natural_b": f"Remember {name}={value} for the final combination.", "natural_c": f"Preserve {name}={value} across the interaction.",
        "compact": f"Keep {name}={value}.", "terse": f"Carry {name}={value}.", "narrative": f"Your local context includes {name}={value}; retain it.",
    }[style]


def _contract(required: list[str], forbidden: list[str], ordering: list[tuple[str, str]], cardinality: dict[str, int]) -> dict[str, Any]:
    return {
        "required_atoms": required,
        "forbidden_atoms": forbidden,
        "ordering": [{"before": a, "after": b} for a, b in ordering],
        "cardinality": cardinality,
        "hard_gate": {"final_answer_exact": True, "all_required_atoms": True, "all_forbidden_atoms_false": True, "ordering_satisfied": True, "cardinality_exact": True},
    }


def _child(name: str, resource: Resource, message_contract: Any = "send result once to parent") -> dict[str, Any]:
    return {"name": name, "resource_path": resource.path, "operation": resource.operation, "expected_result": resource.result, "message_contract": message_contract}


def _row(split: Split, index: int, seed: int, family: str, style: str, prompt: str, files: dict[str, str], oracle: dict[str, Any], atoms: list[str], difficulty: int, timing: str, fault: str = "none") -> dict[str, Any]:
    metadata = {"episode_family": family, "instruction_style": style, "resource_families": [x["family"] for x in oracle["resource_ownership"].values()], "composition_atoms": atoms, "fault_mode": fault, "timing_regime": timing, "difficulty": difficulty}
    metadata["axis_signature"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    eid = f"{split}-{family}-{index:08d}-{seed & 0xfffffff:07x}"
    return {"schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION, "episode_id": eid, "split": split, "index": index, "seed": seed, "public": {"system_prompt": SYSTEM_PROMPT, "user_prompt": prompt, "workspace_files": files}, "oracle": oracle, "metadata": metadata}


def generate_episode(split: Split, index: int, master_seed: int = 20260816) -> dict[str, Any]:
    seed = _seed(master_seed, split, index); rng = random.Random(seed); root = _root(split, index, rng)
    family = (OOD_FAMILIES if split == "ood_gen" else TRAIN_FAMILIES)[index % (8 if split == "ood_gen" else 6)]
    style = rng.choice(STYLES[split]); state = (rng.choice(STATE_NAMES[split]), rng.randint(2, 17)); names = rng.sample(list(CHILD_NAMES[split]), 3)
    own: dict[str, dict[str, Any]] = {}; files: dict[str, str] = {}; children: list[dict[str, Any]] = []

    if family == "direct":
        r = _pick_resource(split, rng, root, "local"); files[r.path] = r.content; own[r.path] = {"owner": "coordinator", "family": r.family, "operation": r.operation}
        prompt = f"{_header(style, None)} Coordinator owns {r.path}; {r.operation} locally. Do not delegate. Return {SCHEMAS[family]}."
        oracle = {"expected_route": "direct", "final_answer": {"result": r.result}, "resource_ownership": own, "children": [], "fault_plan": {"type": "none"}, "trajectory_contract": _contract(["coordinator_read_local", "final_answer"], ["spawn_child", "poll", "discover_child"], [("coordinator_read_local", "final_answer")], {"spawn_child": 0})}
        return _row(split, index, seed, family, style, prompt, files, oracle, ["direct", "nondelegation"], 1, "immediate")

    if family in ("parallel", "triple"):
        width = 3 if family == "triple" else 2; rs = [_pick_resource(split, rng, root, f"remote-{i}") for i in range(width)]
        for name, r in zip(names[:width], rs, strict=True): files[r.path] = r.content; own[r.path] = {"owner": f"child:{name}", "family": r.family, "operation": r.operation}; children.append(_child(name, r))
        offset = state[1]; result = _combine([r.result for r in rs], offset); labels = ("alpha", "beta", "gamma"); answer = {labels[i]: rs[i].result for i in range(width)} | {"offset": offset, "result": result}
        assignments = " ".join(f"{n} owns {r.path} and must {r.operation}." for n, r in zip(names[:width], rs, strict=True))
        prompt = f"{_header(style, state)} {assignments} Spawn all independent children before waiting; do not inspect their resources or poll. Combine their explicit results with {state[0]}. Return {SCHEMAS[family]}."
        required = ["retain_state", "yield", "final_answer"] + [f"spawn:{n}" for n in names[:width]] + [f"receive:{n}" for n in names[:width]]
        ordering = [("retain_state", f"spawn:{n}") for n in names[:width]] + [(f"spawn:{n}", "yield") for n in names[:width]] + [("yield", f"receive:{n}") for n in names[:width]] + [(f"receive:{n}", "final_answer") for n in names[:width]]
        oracle = {"expected_route": "parallel", "final_answer": answer, "coordinator_state": {state[0]: offset}, "resource_ownership": own, "children": children, "fault_plan": {"type": "none", "delivery_order": list(reversed(names[:width]))}, "trajectory_contract": _contract(required, ["coordinator_read_child_owned", "poll", "discover_child", "serialized_fanout_wait"], ordering, {"spawn_child": width, "child_result_message": width})}
        return _row(split, index, seed, family, style, prompt, files, oracle, ["ownership", "parallel", "yield", "fanin"], 4 if width == 3 else 3, "reverse_delivery")

    remote = _pick_resource(split, rng, root, "remote", integer=family == "followup"); files[remote.path] = remote.content; own[remote.path] = {"owner": f"child:{names[0]}", "family": remote.family, "operation": remote.operation}

    if family == "single":
        offset = state[1]; answer = {"child": remote.result, "offset": offset, "result": _combine([remote.result], offset)}; children = [_child(names[0], remote)]
        prompt = f"{_header(style, state)} {names[0]} owns {remote.path} and must {remote.operation}. Never inspect it in coordinator code. Retain the child handle, yield instead of polling, combine the explicit result with {state[0]}, and return {SCHEMAS[family]}."
        req = ["retain_state", f"spawn:{names[0]}", "retain_handle", "yield", f"receive:{names[0]}", "final_answer"]; order = [("retain_state", f"spawn:{names[0]}"), (f"spawn:{names[0]}", "yield"), ("yield", f"receive:{names[0]}"), (f"receive:{names[0]}", "final_answer")]
        oracle = {"expected_route": "single", "final_answer": answer, "coordinator_state": {state[0]: offset}, "resource_ownership": own, "children": children, "fault_plan": {"type": "none"}, "trajectory_contract": _contract(req, ["coordinator_read_child_owned", "poll", "discover_child", "delegate_coordinator_state"], order, {"spawn_child": 1, "child_result_message": 1})}
        return _row(split, index, seed, family, style, prompt, files, oracle, ["ownership", "single_child", "yield", "fanin"], 2, "child_message")

    if family == "mixed":
        local = _pick_resource(split, rng, root, "local"); files[local.path] = local.content; own[local.path] = {"owner": "coordinator", "family": local.family, "operation": local.operation}
        offset = state[1]; answer = {"local": local.result, "child": remote.result, "offset": offset, "result": _combine([local.result, remote.result], offset)}; children = [_child(names[0], remote)]
        prompt = f"{_header(style, state)} Coordinator owns {local.path} and must {local.operation}. {names[0]} owns {remote.path} and must {remote.operation}. Do local work while the child runs; never inspect the child-owned resource. Return {SCHEMAS[family]}."
        req = ["retain_state", f"spawn:{names[0]}", "coordinator_read_local", "yield", f"receive:{names[0]}", "final_answer"]; order = [("retain_state", f"spawn:{names[0]}"), (f"spawn:{names[0]}", "coordinator_read_local"), ("coordinator_read_local", "yield"), ("yield", f"receive:{names[0]}"), (f"receive:{names[0]}", "final_answer")]
        oracle = {"expected_route": "mixed", "final_answer": answer, "coordinator_state": {state[0]: offset}, "resource_ownership": own, "children": children, "fault_plan": {"type": "none"}, "trajectory_contract": _contract(req, ["coordinator_read_child_owned", "poll", "discover_child", "delegate_coordinator_owned"], order, {"spawn_child": 1, "child_result_message": 1})}
        return _row(split, index, seed, family, style, prompt, files, oracle, ["direct", "ownership", "single_child", "concurrency", "fanin"], 3, "local_while_child_runs")

    if family == "followup":
        multiplier = state[1]; subtotal = int(remote.result); answer = {"subtotal": subtotal, "multiplier": multiplier, "result": subtotal * multiplier}; children = [_child(names[0], remote, ["send 'need multiplier' to parent", "after reply send subtotal and result"])]
        prompt = f"{_header(style, state)} {names[0]} owns {remote.path}; it must {remote.operation}, retain the subtotal, then ask you for the multiplier. Reply from retained state, yield again, and wait for its explicit final message. Never inspect the child file or poll. Return {SCHEMAS[family]}."
        req = ["retain_state", f"spawn:{names[0]}", "yield", f"receive_request:{names[0]}", f"send_followup:{names[0]}", "yield_after_followup", f"receive_result:{names[0]}", "final_answer"]; order = [(req[i], req[i + 1]) for i in range(len(req) - 1)]
        oracle = {"expected_route": "followup", "final_answer": answer, "coordinator_state": {state[0]: multiplier}, "resource_ownership": own, "children": children, "fault_plan": {"type": "none"}, "trajectory_contract": _contract(req, ["coordinator_read_child_owned", "poll", "discover_child", "guess_followup_value"], order, {"spawn_child": 1, "parent_to_child_message": 1, "child_to_parent_message": 2})}
        return _row(split, index, seed, family, style, prompt, files, oracle, ["ownership", "single_child", "yield", "child_to_parent", "parent_to_child", "resume", "fanin"], 4, "two_resume_cycles")

    if family == "verify":
        manifest = root + "/verification.json"; digest = hashlib.sha256(remote.content.encode()).hexdigest()[:12]; files[manifest] = json.dumps({"expected_digest": digest}); own[manifest] = {"owner": "coordinator", "family": "verification_manifest", "operation": "compare child digest to expected_digest"}; children = [_child(names[0], remote, "send JSON with result and 12-char SHA-256 digest")]
        prompt = f"{_header(style, None)} {names[0]} owns {remote.path} and must {remote.operation} plus report its 12-char SHA-256 digest. Coordinator owns {manifest}; verify before accepting. The digest is verification evidence only: after it matches, set both child and result to the child's computed resource result, and do not put the digest in the final JSON. Never inspect the child resource. Return {SCHEMAS[family]}."
        req = [f"spawn:{names[0]}", "coordinator_read_verification_manifest", "yield", f"receive:{names[0]}", "verify_child_digest", "final_answer"]; order = [(req[i], req[i + 1]) for i in range(len(req) - 1)]
        oracle = {"expected_route": "single_verify", "final_answer": {"child": remote.result, "verified": True, "result": remote.result}, "resource_ownership": own, "children": children, "fault_plan": {"type": "none", "on_mismatch": "request one correction"}, "trajectory_contract": _contract(req, ["coordinator_read_child_owned", "poll", "accept_unverified_child_result"], order, {"spawn_child": 1})}
        return _row(split, index, seed, family, style, prompt, files, oracle, ["ownership", "single_child", "verification", "fanin"], 4, "verify_after_message")

    if family == "reclaim":
        children = [_child(names[0], remote)]; prompt = f"{_header(style, None)} {names[0]} initially owns {remote.path} and should {remote.operation}. Do not inspect it while the child is healthy. If an explicit child failure arrives, reclaim once, compute locally, and return {SCHEMAS[family]}."
        req = [f"spawn:{names[0]}", "yield", f"receive_failure:{names[0]}", "explicit_reclaim", "coordinator_read_after_reclaim", "final_answer"]; order = [(req[i], req[i + 1]) for i in range(len(req) - 1)]; fault = {"type": "inject_child_failure", "after": f"spawn:{names[0]}", "message": "RESOURCE_UNAVAILABLE", "ownership_transition": {"from": f"child:{names[0]}", "to": "coordinator", "trigger": "explicit_child_failure"}}
        oracle = {"expected_route": "reclaim", "final_answer": {"reclaimed": True, "result": remote.result}, "resource_ownership": own, "children": children, "fault_plan": fault, "trajectory_contract": _contract(req, ["coordinator_read_child_owned_before_reclaim", "poll", "reclaim_without_failure", "respawn_same_failed_child"], order, {"spawn_child": 1, "reclaim": 1})}
        return _row(split, index, seed, family, style, prompt, files, oracle, ["ownership", "single_child", "failure", "reclaim", "direct"], 5, "failure_then_reclaim", "inject_child_failure")

    raise AssertionError(family)


def validate_row(row: dict[str, Any]) -> None:
    if row["schema_version"] != SCHEMA_VERSION or "reasoning_content" in json.dumps(row):
        raise ValueError("invalid schema or reasoning_content")
    public, oracle = row["public"], row["oracle"]; files, own = public["workspace_files"], oracle["resource_ownership"]
    if set(files) != set(own): raise ValueError("workspace/ownership path mismatch")
    contract = oracle["trajectory_contract"]
    if set(contract["required_atoms"]) & set(contract["forbidden_atoms"]): raise ValueError("contradictory contract")
    if oracle["expected_route"] == "direct" and oracle["children"]: raise ValueError("direct route has children")
    if oracle["expected_route"] != "direct" and not oracle["children"]: raise ValueError("delegated route lacks children")
    for child in oracle["children"]:
        if child["resource_path"] not in files or not own[child["resource_path"]]["owner"].startswith("child:"):
            raise ValueError("invalid child resource ownership")
    if not all(contract["hard_gate"].values()): raise ValueError("hard gate must be conjunctive")


def materialize(output: Path, train_count: int = 4096, valid_count: int = 512, ood_count: int = 512, train_start: int = 0, master_seed: int = 20260816) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True); manifest: dict[str, Any] = {"schema_version": "procedural-harness-master-v1/manifest/v1", "generator_version": GENERATOR_VERSION, "master_seed": master_seed, "train_stream": {"start_index": train_start, "count_materialized": train_count, "index_space": "non-negative integers"}, "frozen_eval": {"valid_gen": {"start_index": 0, "count": valid_count}, "ood_gen": {"start_index": 0, "count": ood_count}}, "splits": {}}
    for split, start, count in (("train_gen", train_start, train_count), ("valid_gen", 0, valid_count), ("ood_gen", 0, ood_count)):
        rows = [generate_episode(split, i, master_seed) for i in range(start, start + count)]
        for row in rows: validate_row(row)
        text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows); file = output / f"{split}.jsonl"; file.write_text(text)
        meta = [r["metadata"] for r in rows]; manifest["splits"][split] = {"episodes": count, "file": file.name, "sha256": hashlib.sha256(text.encode()).hexdigest(), "episode_families": sorted({m["episode_family"] for m in meta}), "instruction_styles": sorted({m["instruction_style"] for m in meta}), "resource_families": sorted({f for m in meta for f in m["resource_families"] if f != "verification_manifest"}), "fault_modes": sorted({m["fault_mode"] for m in meta}), "axis_signatures": len({m["axis_signature"] for m in meta})}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"); return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("output", type=Path); p.add_argument("--train-count", type=int, default=4096); p.add_argument("--valid-count", type=int, default=512); p.add_argument("--ood-count", type=int, default=512); p.add_argument("--train-start", type=int, default=0); p.add_argument("--master-seed", type=int, default=20260816); a = p.parse_args()
    if min(a.train_count, a.valid_count, a.ood_count) <= 0 or a.train_start < 0: raise SystemExit("counts must be positive and train-start non-negative")
    print(json.dumps(materialize(a.output, a.train_count, a.valid_count, a.ood_count, a.train_start, a.master_seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
