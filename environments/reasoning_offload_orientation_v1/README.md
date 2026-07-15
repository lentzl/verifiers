# reasoning-offload-orientation-v1

File-backed synthetic tasks for measuring whether an agent uses the RLM harness as
a persistent control environment. The six balanced families cover direct-answer
controls, inspection, runtime state, helper creation, verification, and repair.

Correctness is the only reward. Tool use and recovery are recorded as metrics, so
an agent cannot improve reward merely by producing a longer trajectory. The train
and eval splits use disjoint generator variants.

## Develop

Install and inspect the taskset:

```bash
uv pip install -e environments/reasoning_offload_orientation_v1
uv run eval reasoning-offload-orientation-v1 --taskset.split eval -n 24
```

Use the built-in `rlm` harness for model evaluation. A 24-task eval covers each of
the 12 held-out templates twice. The repository includes matched base and snapshot
capability-gate configs under `configs/reasoning_offload_orientation_*.toml`.
