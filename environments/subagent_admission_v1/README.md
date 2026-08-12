# Subagent Admission V1

This development environment adds a dense reward for the first parent transition in
the Prime Agent single-child protocol. It imports `subagent-communication-v1` for task
generation, runtime setup, full-protocol metrics, and held-out variants; the frozen V1
environment is not modified. Inherited rewards are recorded as zero by default so the
training advantage comes only from admission control. Set
`task.reward_mode = "mixed"` only to reproduce the initial mixed-reward preflight.
The default `task.reward_shape = "strict"` reproduces the original all-or-nothing
first-cell atom accounting. `task.reward_shape = "dense"` keeps the same six target
events but can credit retained exact spawns that occur late, while first-cell ordering
and local work after admission remain necessary for full reward.

The admission reward measures whether the first coordinator IPython cell:

- calls `rlm` before coordinator-local checksum work;
- retains the awaited child handle;
- names `shard-worker` and includes the exact delegated path and reply contract;
- does not read the delegated shard in the parent; and
- computes coordinator-local evidence only after admission.

Use `families = ["single"]` and short non-autonomous rollouts when optimizing this
transition. Promotion still requires the complete `subagent-communication-v1`
development gates and Frozen Capacity Battery V1.
