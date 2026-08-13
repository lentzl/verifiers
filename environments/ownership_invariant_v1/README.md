# Ownership Invariant V1

This environment turns matched ownership examples into executable first-decision
tasks. It does not expose demonstrations or synthetic reasoning. The child-owned arm
is intended for grouped native-sibling SDPO: a strict success is one complete
coordinator transition that retains local state, admits exactly the assigned child,
keeps the resource path inside that child's prompt, retains the handle, and yields
without polling or further action.

The coordinator-owned arm is a matched restraint gate. It requires local state,
direct resource access, no delegation, and the correct result. It must not be mixed
into the initial SDPO source.

The task reward defaults to `strict`. Broad GRPO curricula may set
`task.reward_shape = "dense"` to receive the mean structural-atom score plus a strict
completion bonus. Strict success remains the promotion metric; dense shaping only
provides contrast between incomplete on-policy decisions and cannot outscore a
complete decision.

The frozen splits are:

- `admission`: eight training resource families with two alternating ownership phrasings;
- `heldout_phrasing`: the same resources under two disjoint phrasings;
- `heldout_resource`: TSV and XML tasks absent from the admission split.

Synthetic resources define executable tasks and verifier state only. They are never
teacher policy. Any privileged SDPO solution must come from a strict native success
in the same rollout group.
