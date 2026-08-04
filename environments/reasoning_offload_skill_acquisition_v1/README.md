# reasoning-offload-skill-acquisition-v1

Hidden-family synthetic tasks for evaluating persistent skill acquisition with a
frozen coding-agent policy.

The task prompts expose only natural file-processing problems. Repeated operation
families are retained as evaluator metadata so an experiment can measure transfer
and unrelated-family regressions without giving those labels to the learner.

The fixed splits are:

- `discovery`: six variants per family for collecting trajectories and proposing a skill
- `validation`: three fresh variants per family for accepting or rejecting the proposal
- `test`: three untouched variants per family for final arm comparison

Correctness is the only reward. IPython use and state reuse are diagnostics for the
current RLM-backed experiments; the task data and scoring are harness-independent.
Persistent skills are supplied by the selected environment rather than owned by the
taskset.

```bash
uv run eval reasoning-offload-skill-acquisition-v1 \
  --env.taskset.split discovery \
  --env.agent.harness.id rlm \
  -n 24
```
