# Programmatic Episodic Memory v2

Executable Prime Agent tasks backed by the frozen JSONL splits produced by
`scripts/generate_programmatic_episodic_memory_v2.py` in the Harness Mastery
research branch.

The taskset materializes each row's workspace files, preserves any follow-up
user turns in one persistent interaction, exposes the expert trajectory as the
top-level `demonstration` field expected by Prime-RL's `opsd` algorithm, and
scores answer correctness, retrieve/no-retrieve policy, grounding, stale-state
resolution, context-reset recovery, persistent-index reuse, current-turn
override, repeated cells, and call efficiency.

Set `condition_on_demonstration = true` only for the pre-gradient self-teacher
admission audit. OPSD training should leave the environment unconditioned: the
algorithm itself prepends the demonstration while scoring the sampled policy
tokens.
