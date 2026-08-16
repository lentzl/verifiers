# Procedural Harness Master V1

Executable Prime Agent coordinator tasks generated from fresh procedural episodes.
Only the public prompt and workspace files reach the model. The hidden oracle is
retained in task data for deterministic scoring.

The primary reward is a conjunctive hard gate:

```text
final answer exact
AND every required trajectory atom observed
AND no forbidden atom observed
AND all ordering constraints satisfied
AND all cardinalities exact
```

`train_gen` is an unbounded index stream. `valid_gen` and `ood_gen` use frozen
indices and held-out generation axes. The `reclaim` family withholds the failed
child's resource until the environment emits the explicit failure transition;
the coordinator may access it only after that transition.

## Harness-action curriculum

`curriculum_rung` selects a strict executable subset of the same Prime Agent
event contract without changing the default generated benchmark:

- `atomic_state`: persist coordinator state and reuse it in a later IPython call.
- `atomic_send`: retain one child handle, yield, and accept one explicit message.
- `atomic_followup`: preserve state across two causal resume/message cycles.
- `atomic_parallel`: retain two handles, spawn both before yielding, then fan in.

Payloads are deliberately trivial, while every required action remains a hard
conjunction over real runtime events. Child completion notices without an
explicit `agent_message.send` delivery do not count as result messages.
