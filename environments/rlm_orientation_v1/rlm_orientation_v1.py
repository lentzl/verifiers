import verifiers as vf
from harnesses import RLM, RLMConfig


def _completion_messages(state) -> list[dict]:
    completion = state.get("completion") or []
    return completion if isinstance(completion, list) else []


def _completion_text(state) -> str:
    parts = []
    for message in _completion_messages(state):
        if isinstance(message, dict):
            parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


def _used_ipython(state) -> bool:
    for message in _completion_messages(state):
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls") or []:
            if "ipython" in str(tool_call):
                return True
    return False


@vf.reward(weight=1.0)
async def exact_answer_with_ipython(task, state) -> float:
    return float(str(task["answer"]) in _completion_text(state) and _used_ipython(state))


@vf.metric
async def used_ipython(state) -> float:
    return float(_used_ipython(state))


def load_tasks(split: vf.TaskSplit = "train"):
    rows = [
        {
            "question": (
                "Use the available IPython tool to compute 19 * 23. "
                "Print exactly: RLM_ORIENTATION_ANSWER: 437"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 437",
            "split": "train",
        },
        {
            "question": (
                "Use the available IPython tool to compute 17 * 29. "
                "Print exactly: RLM_ORIENTATION_ANSWER: 493"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 493",
            "split": "train",
        },
        {
            "question": (
                "Use the available IPython tool to compute 31 + 47 + 58. "
                "Print exactly: RLM_ORIENTATION_ANSWER: 136"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 136",
            "split": "train",
        },
        {
            "question": (
                "Use the available IPython tool to compute 144 // 12. "
                "Print exactly: RLM_ORIENTATION_ANSWER: 12"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 12",
            "split": "eval",
        },
        {
            "question": (
                "Use the available IPython tool to compute 7 * 8 + 9. "
                "Print exactly: RLM_ORIENTATION_ANSWER: 65"
            ),
            "answer": "RLM_ORIENTATION_ANSWER: 65",
            "split": "eval",
        },
    ]
    return [row for row in rows if row["split"] == split]


class RLMOrientationTasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["exact_answer_with_ipython"]
    metrics: list[str] = ["used_ipython"]


class RLMOrientationTaskset(vf.Taskset[RLMOrientationTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(split)


def load_taskset(config: RLMOrientationTasksetConfig) -> RLMOrientationTaskset:
    return RLMOrientationTaskset(config=config)


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
