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
