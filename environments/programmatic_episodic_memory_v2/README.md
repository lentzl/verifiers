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

Set `causal_feedback_retries = 1` to give an unsuccessful turn one bounded
retry with diagnostic environment feedback. Feedback names the violated
retrieval, state-reuse, error-repair, or event-semantics rule without exposing
the expected answer. The trace records feedback and final per-request answers,
so repaired trajectories can receive ordinary RL credit while unrepaired
attempts retain trustworthy feedback for later conditioned replay.
