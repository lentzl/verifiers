# oolong-synth-v1

Oolong synthetic long-context tasks solved by an agent in a sandbox: each task's long context window is uploaded to a file so the agent can scan it from a REPL and write a single-token answer to `/workspace/answer.txt`. Tasks are scored with the deterministic official Oolong synth rules (exact match, with partial credit for numeric and date answers), or by an optional host-side binary LLM judge when configured.

## Taskset

- **Source:** [oolongbench/oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth)

Use `example_offset` and `num_examples` to materialize disjoint slices from a
context-length bucket for training and evaluation.
- **Size:** 1300 tasks (`validation` split), filtered at load time to a single `context_len` token bucket (default 262144)

## Changelog

- 2026-06-24: Initial v1 taskset.
