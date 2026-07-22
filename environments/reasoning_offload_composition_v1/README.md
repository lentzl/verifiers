# reasoning-offload-composition-v1

File-backed synthetic tasks for the first compositional stage after basic RLM
harness orientation. Each task requires the agent to run an end-to-end checker,
repair one provided operation from executable feedback, inspect a pipeline
manifest, load a target into runtime state, and apply the repaired operations in
the declared order.

Correctness is the only reward. Sequence-aware repair, manifest inspection,
target loading, state reuse, and provided-operation reuse are recorded as metrics.
Train and eval use disjoint pipeline variants.

## Develop

```bash
uv pip install -e environments/reasoning_offload_composition_v1
uv run eval reasoning-offload-composition-v1 --taskset.split eval -n 4
```
