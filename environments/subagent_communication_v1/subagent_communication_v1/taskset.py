"""Native Prime Agent depth-one delegation and messaging tasks."""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.types import AssistantMessage, ToolMessage, UserMessage, content_text

Family = Literal["direct", "single", "parallel", "handshake", "followup"]
InstructionLevel = Literal["standard", "guided"]

FAMILIES: tuple[Family, ...] = ("direct", "single", "parallel", "followup")
TRAIN_VARIANTS = (0, 1, 2, 3)
EVAL_VARIANTS = (4, 5)
COMPLETION_GATE_PATH = "/workspace/.subagent-communication/completion_gate.py"
OPSD_TEMPLATE = (
    "<Question>\n{question}\n"
    "This is an example for a response to the question:\n"
    "<Demonstration>\n{demonstration}\n"
    "Now answer with a response of your own, including the thinking process:"
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

class SubagentCommunicationData(vf.TaskData):
    family: Family
    template_variant: int
    instruction_level: InstructionLevel = "standard"
    answer: dict[str, int]
    expected_children: tuple[str, ...] = ()
    child_paths: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    followup_secret: int | None = None
    demonstration: str | None = None
    demonstrations: dict[str, str] | None = None


def _weighted(values: list[int]) -> int:
    return sum((index + 1) * value for index, value in enumerate(values))


def _values(rng: random.Random, count: int) -> list[int]:
    return [rng.randint(-19, 29) for _ in range(count)]


def _json(values: list[int]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _child_instruction(path: str) -> str:
    return (
        f"Read {path}, compute its weighted checksum, then send the integer checksum to "
        "your parent with agent_message before answering."
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
) -> str | None:
    if family not in {"single", "parallel"}:
        return None
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
            "End the turn without polling. After shard-worker's explicit message arrives, bind its "
            "body as remote, compute total = local + remote, and return exactly:\n"
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
    else:
        return demonstrations
    for child_name, answer_key in children:
        path = child_paths[child_name]
        child_question = f"[task from parent]\n\n{_child_instruction(path)}"
        demonstrations[child_question] = _child_demonstration(path, answer[answer_key])
    return demonstrations


def _completion_gate_source(expected_keys: tuple[str, ...], family: Family) -> str:
    required_child_messages = {
        "single": {"shard-worker": 1},
        "parallel": {"alpha-worker": 1, "beta-worker": 1},
        "followup": {"key-worker": 2},
        "handshake": {"relay-worker": 2},
    }.get(family, {})
    if family == "followup":
        feedback = (
            "completion gate: final JSON is not ready. Preserve the existing delegation: "
            "do not inspect the delegated shard, spawn another child, or redo the child's work. "
            "If the latest key-worker message requests the multiplier, send the multiplier stated "
            "in the original task to that existing child. If no new child message is present, end "
            "this continuation with a brief waiting status and no tool call; this active autonomous "
            "run will accept future child messages. Return strict JSON only after key-worker sends "
            "its final subtotal and result."
        )
    elif family == "handshake":
        feedback = (
            "completion gate: final JSON is not ready. Preserve the existing delegation: do not "
            "spawn another child. If the latest relay-worker message requests the nonce, send the "
            "nonce stated in the original task to that existing child. If no new child message is "
            "present, end this continuation with a brief waiting status and no tool call; this active "
            "autonomous run will accept future child messages. Return strict JSON only after "
            "relay-worker echoes the nonce."
        )
    elif family in {"single", "parallel"}:
        feedback = (
            "completion gate: final JSON is not ready. Preserve every existing delegation: do not "
            "inspect a delegated shard, spawn replacement children, or redo their work. If no new "
            "child message is present, end this continuation with a brief waiting status and no tool "
            "call; this active autonomous run will accept future child messages. Return strict JSON "
            "only after every existing child sends its result."
        )
    else:
        feedback = (
            "completion gate: final JSON is not ready. Complete the coordinator-local computation "
            "and return one JSON object with exactly the requested keys and integer values."
        )
    return f'''import json
import os
import sys
from pathlib import Path

EXPECTED_KEYS = {expected_keys!r}
REQUIRED_CHILD_MESSAGES = {required_child_messages!r}


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


agent_dir = Path(os.environ.get("PRIME_AGENT_CODING_AGENT_DIR", ""))
session_files = sorted(
    (agent_dir / "sessions").rglob("*.jsonl"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
for path in session_files:
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        continue
    header = entries[0]
    if header.get("rlmDepth") not in (None, 0):
        continue
    if header.get("parentSession") or header.get("parentSessionId"):
        continue
    final_payload = None
    final_index = None
    for index in range(len(entries) - 1, -1, -1):
        message = session_message(entries[index])
        if message.get("role") != "assistant":
            continue
        try:
            final_payload = json.loads(content_text(message.get("content")).strip())
        except (TypeError, json.JSONDecodeError):
            break
        final_index = index
        break
    if final_index is None:
        continue
    child_message_counts = {{name: 0 for name in REQUIRED_CHILD_MESSAGES}}
    seen_child_message_ids = set()
    for entry in entries[:final_index]:
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
        child_message_counts[child_name] += 1
    child_evidence_ready = all(
        child_message_counts.get(name, 0) >= count
        for name, count in REQUIRED_CHILD_MESSAGES.items()
    )
    if (
        child_evidence_ready
        and isinstance(final_payload, dict)
        and set(final_payload) == set(EXPECTED_KEYS)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in final_payload.values()
        )
    ):
        raise SystemExit(0)

print(
    {feedback!r},
    file=sys.stderr,
)
raise SystemExit(1)
'''


def _task_prompt(
    family: Family,
    variant: int,
    instance: int,
    seed: int,
    instruction_level: InstructionLevel,
) -> tuple[
    str,
    dict[str, int],
    tuple[str, ...],
    dict[str, str],
    dict[str, str],
    int | None,
]:
    rng = random.Random(seed * 1_000_003 + variant * 10_007 + instance * 101)
    prefix = (
        "Return one JSON object with exactly the requested keys and integer values. "
        "A shard checksum is sum((index + 1) * value)."
    )

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
        guidance = (
            f"Do not open {remote_path} in the coordinator. Spawn the child before computing the "
            f"local checksum, using handle = await rlm(\"Read {remote_path}, compute its weighted "
            "checksum, then send the integer checksum to your parent with agent_message before "
            "answering.\", name='shard-worker'). Retain handle and compute the local checksum in "
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
        guidance = (
            f"Do not open {alpha_path} or {beta_path} in the coordinator. In separate IPython "
            "calls, assign alpha = await rlm(\"Read "
            f"{alpha_path}, compute its weighted checksum, then make a separate final IPython "
            "call exactly await agent_message.send(str(checksum), receiver_role='parent'). Your "
            "task is incomplete until that call succeeds.\", "
            "name='alpha-worker') and beta = await rlm(\"Read "
            f"{beta_path}, compute its weighted checksum, then make a separate final IPython call "
            "exactly await agent_message.send(str(checksum), receiver_role='parent'). Your task "
            "is incomplete until that call succeeds.\", "
            "name='beta-worker'). Retain both handles, then stop calling tools for this turn."
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
        guidance = (
            f"First retain nonce = {secret} in coordinator IPython state. Then assign child = await "
            "rlm(\"You are relay-worker, my child. Do not call rlm or message "
            "a child. In a separate IPython call exactly await agent_message.send('need nonce', "
            "receiver_role='parent'). End that turn and resume only when my parent follow-up "
            "arrives. In the resumed turn, bind the exact body of the latest [from parent] message "
            "to a nonce variable rather than guessing it. Then make a second successful "
            "receiver_role='parent' message containing that nonce. Your task is incomplete until "
            "both parent messages succeed.\", "
            "name='relay-worker'). Do not put the nonce in that string. Retain child and stop "
            "calling tools for this turn. After its request arrives, use await "
            "agent_message.send(str(nonce), receiver_role='child', "
            "receiver_name=child.name), then wait for its final reply."
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
            "multiplier from you through an agent message. Send it only in a direct follow-up, wait "
            'for the child\'s final result message, then return {"subtotal": value, '
            '"multiplier": value, "result": value}.'
        )
        children = ("key-worker",)
        child_paths = {"key-worker": remote_path}
        files = {remote_path: json.dumps(remote)}
        guidance = (
            f"Do not open {remote_path} in the coordinator. First retain multiplier = {secret} in "
            "coordinator IPython state. Then assign child = await rlm(\"You are "
            f"key-worker, my child. Read {remote_path} and retain its subtotal. Do not call rlm "
            "or message a child. In a separate IPython call exactly await "
            "agent_message.send('need multiplier', receiver_role='parent'). End that turn and "
            "resume only when my parent follow-up arrives. In the resumed turn, bind the integer "
            "body of the latest [from parent] message to multiplier with int(...) rather than "
            "guessing or hardcoding it. Then multiply the retained subtotal and make a second "
            "successful receiver_role='parent' message containing subtotal and result. Your task "
            "is incomplete until both parent messages succeed.\", "
            "name='key-worker'). Do not put the multiplier in that string. Retain child and stop "
            "calling tools for this turn. After its request arrives, use await "
            "agent_message.send(str(multiplier), receiver_role='child', "
            "receiver_name=child.name), then wait for its final reply."
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
    )


@dataclass
class IpythonEvent:
    code: str
    call_id: str
    node_index: int
    output: str = ""


def _ipython_events(trace: vf.Trace) -> list[IpythonEvent]:
    events: list[IpythonEvent] = []
    by_node_call: dict[tuple[int, str], IpythonEvent] = {}
    for node_index, node in enumerate(trace.nodes):
        message = node.message
        if isinstance(message, AssistantMessage):
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
        elif isinstance(message, ToolMessage) and (
            event := by_node_call.get((node.parent, message.tool_call_id))
        ):
            event.output = content_text(message.content)
    return events


def _duplicate_cells(trace: vf.Trace, events: list[IpythonEvent]) -> int:
    by_branch: dict[int, list[str]] = {}
    for event in events:
        branch_root = event.node_index
        visited: set[int] = set()
        while trace.nodes[branch_root].parent is not None:
            if branch_root in visited:
                break
            visited.add(branch_root)
            branch_root = trace.nodes[branch_root].parent
        by_branch.setdefault(branch_root, []).append(event.code.strip())
    return sum(
        count - 1
        for branch in by_branch.values()
        for count in Counter(branch).values()
        if count > 1
    )


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


def _incoming_child_messages(trace: vf.Trace) -> list[ChildMessage]:
    messages: list[ChildMessage] = []
    for node in trace.nodes:
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
        messages.append(
            ChildMessage(
                name=matched.group(1),
                body=body,
                message_id=message_id.group(1) if message_id else None,
            )
        )
    return messages


def _spawn_name(call: ast.Call, output: str) -> str | None:
    configured = _keyword(call, "name")
    if isinstance(configured, str):
        return configured
    matched = re.search(r"\bname='([^']+)'", output)
    return matched.group(1) if matched else None


def _spawn_prompt(call: ast.Call) -> str | None:
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return None
    value = call.args[0].value
    return value if isinstance(value, str) else None


def _protocol_behavior(
    trace: vf.Trace,
    family: Family,
    expected_children: tuple[str, ...],
    child_paths: dict[str, str],
    followup_secret: int | None,
) -> dict[str, float]:
    events = _ipython_events(trace)
    calls: list[tuple[ast.Call, bool, IpythonEvent]] = []
    for event in events:
        try:
            tree = ast.parse(event.code)
        except SyntaxError:
            continue
        assigned = _assigned_call_names(tree)
        calls.extend(
            (node, id(node) in assigned, event) for node in ast.walk(tree) if isinstance(node, ast.Call)
        )

    attempted_spawns = [(call, retained, event) for call, retained, event in calls if _call_name(call) == "rlm"]
    spawns = [item for item in attempted_spawns if not _failed(item[2].output)]
    names = {_spawn_name(call, event.output) for call, _, event in spawns}
    parent_messages = _incoming_child_messages(trace)
    parent_message_names = {message.name for message in parent_messages}
    child_messages = [
        (call, event)
        for call, _, event in calls
        if _call_name(call) == "agent_message.send"
        and _keyword(call, "receiver_role") == "child"
        and _message_sent(event.output)
    ]
    parent_sends = [
        (call, event)
        for call, _, event in calls
        if _call_name(call) == "agent_message.send"
        and _keyword(call, "receiver_role") == "parent"
        and _message_sent(event.output)
    ]
    list_calls = sum(
        _call_name(call) in {"rlm.list_subagents", "agent_message.list_agents"} and not _failed(event.output)
        for call, _, event in calls
    )
    observation_calls = sum(
        (_call_name(call) or "").startswith("agent_observe.") and not _failed(event.output)
        for call, _, event in calls
    )
    repeated = _duplicate_cells(trace, events)
    retained = sum(retained for _, retained, _ in spawns)
    delegated = {
        name
        for name, path in child_paths.items()
        if any(
            _spawn_name(call, event.output) == name and path in (_spawn_prompt(call) or "")
            for call, _, event in spawns
        )
    }
    secret_withheld = True
    if followup_secret is not None and spawns:
        first_prompt = _spawn_prompt(spawns[0][0])
        secret_withheld = bool(first_prompt and str(followup_secret) not in first_prompt)

    expected_messages = [message for message in parent_messages if message.name in expected_children]
    request_phrase = "need nonce" if family == "handshake" else "need multiplier"
    request_message = next(
        (message for message in expected_messages if request_phrase in message.body.lower()),
        None,
    )

    def originating_send_index(message: ChildMessage | None) -> int | None:
        if message is None or message.message_id is None:
            return None
        return next(
            (event.node_index for _, event in parent_sends if message.message_id in event.output),
            None,
        )

    request_index = originating_send_index(request_message)
    result_messages = [
        (message, index)
        for message in expected_messages
        if message is not request_message and (index := originating_send_index(message)) is not None
    ]
    followup_request_sent = request_index is not None
    followup_after_request = any(
        request_index is not None and request_index < child_event.node_index
        for _, child_event in child_messages
    )
    result_after_followup = any(
        request_index is not None and request_index < child_event.node_index < result_index
        for _, child_event in child_messages
        for _, result_index in result_messages
    )
    followup_causal = followup_request_sent and followup_after_request and result_after_followup
    result_matches_secret = family != "handshake" or any(
        request_index is not None
        and request_index < child_event.node_index < result_index
        and followup_secret is not None
        and str(followup_secret) in message.body
        for _, child_event in child_messages
        for message, result_index in result_messages
    )
    followup_phase_score = (
        float(followup_request_sent) + float(followup_after_request) + float(result_after_followup)
    ) / 3
    retained_ready = retained == len(expected_children)
    stateful_control_progress = 0.0
    if family in {"followup", "handshake"} and retained_ready:
        stateful_control_progress = (
            1.0
            + float(followup_request_sent)
            + float(followup_after_request)
            + float(result_after_followup)
            + float(secret_withheld)
            + float(repeated == 0)
        ) / 6

    if family == "direct":
        checks = [not spawns, not parent_messages, not child_messages, repeated == 0]
    elif family == "single":
        checks = [
            len(spawns) == 1,
            retained == len(expected_children),
            set(expected_children) <= names,
            set(expected_children) <= delegated,
            set(expected_children) <= parent_message_names,
            repeated == 0,
        ]
    elif family == "parallel":
        checks = [
            len(spawns) == 2,
            retained == len(expected_children),
            set(expected_children) <= names,
            set(expected_children) <= delegated,
            set(expected_children) <= parent_message_names,
            repeated == 0,
        ]
    else:
        checks = [
            len(spawns) == 1,
            retained == len(expected_children),
            set(expected_children) <= names,
            set(expected_children) <= delegated,
            secret_withheld,
            len(child_messages) >= 1,
            len(expected_messages) >= 2,
            followup_request_sent,
            followup_after_request,
            result_after_followup,
            *([result_matches_secret] if family == "handshake" else []),
            repeated == 0,
        ]
    return {
        "protocol_score": sum(checks) / len(checks),
        "protocol_aligned": float(all(checks)),
        "spawn_calls": float(len(spawns)),
        "failed_spawn_calls": float(len(attempted_spawns) - len(spawns)),
        "retained_handles": float(retained),
        "named_children": float(len(set(expected_children) & names)),
        "delegated_payloads": float(len(delegated)),
        "messages_to_parent": float(len(parent_messages)),
        "messages_to_child": float(len(child_messages)),
        "roster_calls": float(list_calls),
        "observation_calls": float(observation_calls),
        "secret_withheld": float(secret_withheld),
        "followup_request_sent": float(followup_request_sent),
        "followup_after_request": float(followup_after_request),
        "result_after_followup": float(result_after_followup),
        "followup_result_matches_secret": float(result_matches_secret),
        "followup_phase_score": followup_phase_score,
        "followup_causal": float(followup_causal),
        "stateful_control_progress": stateful_control_progress,
        "duplicate_cells": float(repeated),
    }


def _answer_score(reply: str, expected: dict[str, int]) -> float:
    try:
        actual = json.loads(reply.strip())
    except (AttributeError, json.JSONDecodeError):
        return 0.0
    if not isinstance(actual, dict):
        return 0.0
    return sum(actual.get(key) == value for key, value in expected.items()) / len(expected)


class SubagentCommunicationTask(vf.Task[SubagentCommunicationData]):
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
        accuracy = _answer_score(trace.last_reply, self.data.answer)
        if self.data.family == "direct":
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
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["protocol_score"]

    @vf.reward(weight=1.0)
    async def stateful_control_progress(self, trace: vf.Trace) -> float:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )["stateful_control_progress"]

    @vf.metric
    async def answer_accuracy(self, trace: vf.Trace) -> float:
        return _answer_score(trace.last_reply, self.data.answer)

    @vf.metric
    async def delegation_behavior(self, trace: vf.Trace) -> dict[str, float]:
        return _protocol_behavior(
            trace,
            self.data.family,
            self.data.expected_children,
            self.data.child_paths,
            self.data.followup_secret,
        )


class SubagentCommunicationConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"
    families: tuple[Family, ...] = Field(FAMILIES, min_length=1)
    instruction_level: InstructionLevel = "standard"
    instances_per_template: int = Field(4, ge=1)
    instance_offset: int = Field(0, ge=0)
    seed: int = 20260809
    teacher_conditioned: bool = False


class SubagentCommunicationTaskset(vf.Taskset[SubagentCommunicationTask, SubagentCommunicationConfig]):
    def load(self) -> list[SubagentCommunicationTask]:
        variants = TRAIN_VARIANTS if self.config.split == "train" else EVAL_VARIANTS
        tasks = []
        instances = range(
            self.config.instance_offset,
            self.config.instance_offset + self.config.instances_per_template,
        )
        for instance in instances:
            for variant in variants:
                for family in self.config.families:
                    prompt, answer, children, child_paths, files, secret = _task_prompt(
                        family,
                        variant,
                        instance,
                        self.config.seed,
                        self.config.instruction_level,
                    )
                    demonstration = _expert_demonstration(
                        family,
                        prompt,
                        answer,
                        child_paths,
                    )
                    demonstrations = _branch_demonstrations(
                        family,
                        prompt,
                        answer,
                        child_paths,
                        demonstration,
                    )
                    if self.config.teacher_conditioned:
                        if demonstration is None:
                            raise ValueError(
                                "teacher_conditioned preflight requires a supported demonstration family"
                            )
                        prompt = OPSD_TEMPLATE.format(
                            question=prompt,
                            demonstration=demonstration,
                        )
                    tasks.append(
                        SubagentCommunicationTask(
                            SubagentCommunicationData(
                                idx=len(tasks),
                                name=f"{family}-v{variant}-i{instance}",
                                prompt=prompt,
                                system_prompt=SYSTEM_PROMPT,
                                family=family,
                                template_variant=variant,
                                instruction_level=self.config.instruction_level,
                                answer=answer,
                                expected_children=children,
                                child_paths=child_paths,
                                files=files,
                                followup_secret=secret,
                                demonstration=demonstration,
                                demonstrations=demonstrations,
                            ),
                            self.config.task,
                        )
                    )
        return tasks
