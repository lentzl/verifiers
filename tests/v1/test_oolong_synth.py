import sys
from types import SimpleNamespace

from oolong_synth_v1.taskset import OolongSynthConfig, OolongSynthTaskset


def test_taskset_materializes_requested_matching_slice(monkeypatch) -> None:
    rows = [
        {
            "context_len": context_len,
            "question": f"question-{index}",
            "answer": str(index),
            "context_window_text": f"context-{index}",
            "context_window_text_with_labels": f"labeled-{index}",
        }
        for index, context_len in enumerate([8192, 16384, 16384, 8192, 16384, 16384])
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: iter(rows)),
    )

    tasks = OolongSynthTaskset(
        OolongSynthConfig(
            context_len=16384,
            example_offset=1,
            num_examples=2,
        )
    ).load()

    assert [task.data.idx for task in tasks] == [2, 4]
    assert [task.data.question for task in tasks] == ["question-2", "question-4"]
