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

Set `record_causal_feedback = true` with zero retries for failure-only SDPO.
The environment records the diagnostic in `trace.info["feedback"]` without
sending it back to the student, so the sampled branch contains only the
original attempt. Successful attempts record no feedback and therefore receive
no SDPO target unless a separate sibling-solution route is configured.

## Typed causal-feedback contract

`programmatic_episodic_memory_v2.feedback` freezes a machine-readable v1
failure taxonomy independently of runtime trace extraction. Runtime adapters
should reduce a failed attempt to `MemoryFailureSignals`, call
`diagnose_memory_failure()`, render the model-facing text with
`render_memory_feedback()`, and retain `feedback_contract_payload()` alongside
the legacy feedback string for routing and audit.

The stable leaf codes are:

- `required_tool_missing`
- `unnecessary_tool_use`
- `required_history_not_retrieved`
- `unnecessary_history_retrieval`
- `tool_execution_error`
- `retrieval_too_broad`
- `persistent_state_not_reused`
- `output_contract_violation`
- `event_semantics_mismatch`

Classification precedence is causal: routing and retrieval mistakes are
identified before downstream answer-format or semantic failures. The contract
contains only bounded audit signals and booleans derived from correctness; it
never contains expected-answer text. Model-facing messages are deterministic,
and unknown family names are not interpolated into feedback.

The current schema identifier is
`programmatic-episodic-memory-v2/causal-feedback/v1`. Add a new schema/code when
a meaning changes rather than silently reusing an existing label. This lets
Prime-RL route only diagnostically understood failures to SDPO while preserving
ordinary GRPO for native-clean success and zero manufactured target for
unclassified failures.
