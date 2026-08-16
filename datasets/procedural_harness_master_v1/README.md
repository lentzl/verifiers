# Procedural Harness Master Benchmark V1 — data contract

This directory defines the first data distribution for **benchmaxxing Prime Agent Harness Mastery itself**.

The benchmark is not a static prompt list. The generator emits deterministic executable episodes from a split/index seed. Training can draw arbitrarily many fresh indices; validation and OOD use frozen index banks.

## Objective

An episode is successful only when the final answer is correct **and** the trajectory respects the hidden harness contract. Correct answers obtained by crossing ownership boundaries, unnecessary delegation, polling instead of yielding, serialized fan-out, unverified child results, or premature finalization are failures.

The data format separates `public` model/runtime state, a hidden `oracle` containing answer/ownership/fault/invariant state, and `metadata` generation axes. No demonstrations, golden trajectories, or `reasoning_content` are generated.

## Splits

`train_gen` is an effectively unlimited deterministic stream over non-negative indices. The default materialization is 4,096 examples, and training can continue with fresh `--train-start` windows. It composes direct non-delegation, single-child ownership/fan-in, parallel fan-out/fan-in, mixed local+child work, bidirectional follow-up/resumption, and child-result verification.

`valid_gen` is a frozen 512-example bank using the same semantic/resource families but disjoint wording styles, child-name vocabulary, state-variable vocabulary, and path shapes.

`ood_gen` is a frozen 512-example bank using unseen TSV/XML/JSONL/INI resource types, disjoint wording/path/name vocabularies, width-3 parallel composition, and explicit child failure followed by legal ownership reclaim. Direct controls remain present so unfamiliarity cannot become a delegation shortcut.

## Intended score

The hidden `trajectory_contract.hard_gate` is conjunctive:

`HarnessScore = final_answer_exact AND all_required_atoms AND all_forbidden_atoms_false AND ordering_satisfied AND cardinality_exact`.

Diagnostic metrics explain failures but never compensate for a hard invariant violation.

The eventual official Prime Agent taskset should materialize only `public.workspace_files`, keep `oracle` verifier-side, map real trace events to contract atoms, train against fresh `train_gen`, tune against `valid_gen`, and reserve `ood_gen` for transfer/admission.
