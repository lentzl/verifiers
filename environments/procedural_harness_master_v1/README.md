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
- `atomic_child_request`: encode a request protocol in the initial child prompt,
  retain its handle, yield, and accept one explicit request without replying.
- `atomic_followup`: preserve state across two causal resume/message cycles.
- `atomic_parallel`: retain two handles, spawn both before yielding, then fan in.

Payloads are deliberately trivial, while every required action remains a hard
conjunction over real runtime events. Child completion notices without an
explicit `agent_message.send` delivery do not count as result messages.

## Natural complete-policy curriculum

The `natural_n1` and `natural_n2` rungs broaden semantic context without
weakening the same executable event contract:

- `natural_n1`: complete asynchronous delegation with child-owned evidence,
  coordinator-private state, optional independent coordinator work, one visible
  child result, and final synthesis.
- `natural_n2`: a staged separation-of-duties dependency in which the child
  first completes an independent milestone, requests a coordinator-private
  parameter, receives it, and then reports the completed result.

User prompts describe the job, ownership boundary, semantic dependency, and
requested output only. They are validated to exclude Prime API names and
prescriptive action words such as spawn, yield, polling, or handle retention.
The unchanged Prime Agent system/runtime still documents the available harness.
Train, validation, and OOD splits use disjoint semantic scenario families;
resource schemas, names, path styles, wording, private values, and N1 graph
variants remain procedural. OOD also holds out an N2 composition graph that
adds independent coordinator-local work to the staged child dependency. Hidden
oracles preserve strict hard scoring for
state, ownership, child identity, handle retention, message provenance,
ordering, cardinality, private-value withholding, and exact final output.
