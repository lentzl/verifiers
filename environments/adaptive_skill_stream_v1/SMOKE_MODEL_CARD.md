---
base_model: Qwen/Qwen3.5-2B
library_name: transformers
pipeline_tag: text-generation
tags:
  - prime-rl
  - reinforcement-learning
  - qwen3.5
  - agent-skills
---

# Qwen3.5-2B Prime Agent Adaptive Skills Smoke R1

This is a self-contained merged checkpoint from the first four-step GRPO smoke
of `adaptive-skill-stream-v1`. It starts from the private Prime Agent orientation
adapter over `Qwen/Qwen3.5-2B`, merges those weights, applies four PrimeRL LoRA
updates, and merges the final adapter again. No machine-local base model or
adapter stacking is required to load this snapshot.

The matching environment and train/eval configurations are stored under
`library/`. Run metrics and component logs are stored under `run/`. The training
code used PrimeRL commit `61d3fe8e4` and Verifiers commit `061091b97`; later
documentation-only commits describe the observed smoke result.

## Result

The run completed on one RTX 6000 Ada in 37 minutes with BF16 model and reduction
dtypes, rank-16 LoRA, AdamW, and a frozen vision encoder. All four optimizer
updates were finite, mismatch KL stayed at or below 0.0002, peak trainer memory
was 10.1 GiB, and no rollout errored.

On six held-out tasks with the less explicit `standard` prompts, mean stream
accuracy increased from about 0.02 before training to 0.25 after Step 4.
Installed-skill transfer improved most, and one stable-procedure stream was
partially correct. Ephemeral adaptation remained at zero, and the model did not
yet author a valid portable skill. This is an experimental continuation point,
not a claim that the full durable/adaptive skill curriculum is solved.

## Serving

Use a 32768-token context window. When serving from a local checkout with vLLM,
set the Qwen tool parser explicitly because parser auto-detection cannot infer
the model family from an arbitrary filesystem path:

```bash
uv run inference \
  --model.name /path/to/model \
  --model.max-model-len 32768 \
  --model.tool-call-parser qwen3_coder
```
