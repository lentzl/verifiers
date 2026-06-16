import verifiers as vf
from harnesses import RLM, RLMConfig


def _completion_messages(state) -> list[dict]:
    completion = state.get("completion") or []
    return completion if isinstance(completion, list) else []


def _message_text(message) -> str:
    if not isinstance(message, dict):
        return ""
    parts = [str(message.get("content") or ""), str(message.get("reasoning_content") or "")]
    for tool_call in message.get("tool_calls") or []:
        parts.append(str(tool_call))
    return "\n".join(parts)


def _completion_text(state) -> str:
    return "\n".join(_message_text(message) for message in _completion_messages(state))


def _tool_text(state) -> str:
    parts = []
    for message in _completion_messages(state):
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            parts.append(str(tool_call))
    return "\n".join(parts)


def _used_ipython(state) -> bool:
    return "ipython" in _tool_text(state)


def _contains_all(text: str, markers: list[str]) -> bool:
    return all(marker in text for marker in markers)


@vf.reward(weight=0.8)
async def exact_answer_with_ipython(task, state) -> float:
    return float(str(task["answer"]) in _completion_text(state) and _used_ipython(state))


@vf.reward(weight=0.2)
async def expected_repl_markers(task, state) -> float:
    markers = list(task.get("tool_markers") or [])
    if not markers:
        return 1.0
    return float(_contains_all(_tool_text(state), markers))


@vf.metric
async def used_ipython(state) -> float:
    return float(_used_ipython(state))


@vf.metric
async def expected_repl_markers_present(task, state) -> float:
    markers = list(task.get("tool_markers") or [])
    if not markers:
        return 1.0
    return float(_contains_all(_tool_text(state), markers))


def load_tasks(split: vf.TaskSplit = "train"):
    rows = [
        {
            "family": "verify_arithmetic",
            "question": (
                "Use IPython to verify this calculation rather than doing it only in your head. "
                "Compute (37 * 41) - (19 * 23), then print exactly: "
                "RLM_ORIENTATION_ANSWER: 1080"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 1080",
            "tool_markers": ["37 * 41", "19 * 23"],
            "split": "train",
        },
        {
            "family": "verify_arithmetic",
            "question": (
                "Use IPython to compute and check the remainder of 98765 divided by 97. "
                "Print exactly: RLM_ORIENTATION_ANSWER: 19"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 19",
            "tool_markers": ["98765", "97"],
            "split": "train",
        },
        {
            "family": "helper_creation",
            "question": (
                "Use IPython to define a helper function named normalize_tokens that lowercases "
                "strings, strips surrounding whitespace, and drops empty results. Apply it to "
                "['  Alpha ', '', 'BETA', ' gamma  '] and print exactly: "
                "RLM_ORIENTATION_ANSWER: alpha,beta,gamma"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: alpha,beta,gamma",
            "tool_markers": ["def normalize_tokens", "strip", "lower"],
            "split": "train",
        },
        {
            "family": "helper_creation",
            "question": (
                "Use IPython to define a helper function named count_prefixes. It should count "
                "how many words start with each first letter. Apply it to "
                "['stone', 'star', 'river', 'road', 'reed'] and print exactly: "
                "RLM_ORIENTATION_ANSWER: r=3,s=2"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: r=3,s=2",
            "tool_markers": ["def count_prefixes", "for", "startswith"],
            "split": "train",
        },
        {
            "family": "stateful_reasoning",
            "question": (
                "Use IPython as a scratchpad. Store the list [4, 9, 15, 16, 23, 42] in a named "
                "variable, compute the sum of the even values and the sum of the odd values, "
                "then print exactly: RLM_ORIENTATION_ANSWER: even=62,odd=47"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: even=62,odd=47",
            "tool_markers": ["values", "even", "odd"],
            "split": "train",
        },
        {
            "family": "error_recovery",
            "question": (
                "Use IPython to demonstrate recovery from a small coding mistake. In one code "
                "cell, use try/except around int('forty-two'), recover by using the integer 42, "
                "then print exactly: RLM_ORIENTATION_ANSWER: recovered=42"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: recovered=42",
            "tool_markers": ["try", "except", "forty-two"],
            "split": "train",
        },
        {
            "family": "verify_arithmetic",
            "question": (
                "Use IPython to verify the calculation. Compute (28 * 34) + (17 * 19), "
                "then print exactly: RLM_ORIENTATION_ANSWER: 1275"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 1275",
            "tool_markers": ["28 * 34", "17 * 19"],
            "split": "eval",
        },
        {
            "family": "helper_creation",
            "question": (
                "Use IPython to define a helper function named compact_words that strips "
                "whitespace, lowercases words, sorts them, and joins with '|'. Apply it to "
                "[' Pear ', 'apple', '  plum'] and print exactly: "
                "RLM_ORIENTATION_ANSWER: apple|pear|plum"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: apple|pear|plum",
            "tool_markers": ["def compact_words", "sorted", "join"],
            "split": "eval",
        },
        {
            "family": "stateful_reasoning",
            "question": (
                "Use IPython as a scratchpad. Store {'red': 5, 'blue': 8, 'green': 13} in a "
                "named variable, compute blue + green - red, then print exactly: "
                "RLM_ORIENTATION_ANSWER: 16"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 16",
            "tool_markers": ["red", "blue", "green"],
            "split": "eval",
        },
    ]
    return [row for row in rows if row["split"] == split]


class RLMOrientationV2TasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["exact_answer_with_ipython", "expected_repl_markers"]
    metrics: list[str] = ["used_ipython", "expected_repl_markers_present"]


class RLMOrientationV2Taskset(vf.Taskset[RLMOrientationV2TasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(split)


def load_taskset(config: RLMOrientationV2TasksetConfig) -> RLMOrientationV2Taskset:
    return RLMOrientationV2Taskset(config=config)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
