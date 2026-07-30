# reasoning-offload-orientation-v1

File-backed synthetic tasks for measuring whether an agent uses the RLM harness as
a persistent control environment. The seven balanced families cover direct-answer
controls, inspection, runtime state, helper creation, provided-module reuse,
verification, and repair.

Correctness is the only reward. Tool use and recovery are recorded as metrics, so
an agent cannot improve reward merely by producing a longer trajectory. The train
and eval splits use disjoint generator variants.

Early curriculum stages can set `taskset.instruction_level = "explicit"` to make
the required environment operation concrete for state, verification, and repair
tasks. The task contents and answers remain unchanged. The repair hint identifies
the file-write operation but not the defect, correction, or answer. Later stages
can fade back to the default `"standard"` prompts without changing the capability
being measured. Holdout evaluations should always use the standard level.

Repair process alignment is sequence-aware. A trace must observe a failure, write
`inputs/buggy.py`, and then observe `VERIFIED`; notebook-only experiments after a
failure no longer count as an aligned repair.

## Develop

Install and inspect the taskset:

```bash
uv pip install -e environments/reasoning_offload_orientation_v1
uv run eval reasoning-offload-orientation-v1 --taskset.split eval -n 28
```

Use the built-in `rlm` harness for model evaluation. A 28-task eval covers each of
the 14 held-out templates twice. The repository includes matched base and snapshot
capability-gate configs under `configs/reasoning_offload_orientation_*.toml`.
