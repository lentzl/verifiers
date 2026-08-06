# adaptive-skill-stream-v1

This curriculum drives one policy through four related batches in one Prime Agent
session. It trains the decision boundary between three forms of adaptation:

- reuse an installed executable skill when it already covers the procedure;
- retain and promote a repeatedly validated procedure as a portable Agent Skill;
- keep changing, task-specific information local instead of polluting durable state.

Every answer receives only pass/fail feedback before the next batch. Stream
correctness is the primary reward. A smaller lifecycle reward checks whether an
installed skill was reused, a stable frontier procedure was packaged with valid
`SKILL.md` metadata, or an ephemeral codebook was correctly left unpromoted.
Calls to Prime Agent's `refine` skill and continual-harness CRUD are metrics only.
Answers are plain JSON rather than XML-tagged: Qwen3.5's native tool parser treats
an `<answer>` element as a tool invocation, which prevents Prime Agent from
yielding a final textual reply.

The training recipe starts with `instruction_level = "explicit"`. These hints
name the exact installed-skill call or one-cell operation, but never expose a
batch answer. The held-out eval remains `standard`. This gives early GRPO groups
non-zero outcome variance while preserving a clear fade-out gate: switch training
back to standard once concise four-batch completion is reliable.

The promoted artifact uses the common `.agents/skills/<name>/SKILL.md` layout and
the portable Agent Skills frontmatter shared by current OpenAI, Anthropic, and
Prime-compatible harnesses. The bundled record-normalization fixture additionally
contains a Prime Agent Python package, because it represents an already-installed
executable skill rather than a newly authored portable procedure.

## Develop

```bash
uv pip install -e environments/adaptive_skill_stream_v1
uv run eval adaptive-skill-stream-v1 \
  --env.agent.harness.id prime-agent \
  --env.agent.harness.save-session true \
  --env.agent.harness.skills \
    environments/adaptive_skill_stream_v1/skills/portable_record_normalization \
  --env.agent.runtime.type docker \
  --taskset.split eval \
  -n 6
```

On the rented training host, the reproducible pre-training gate is:

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_adaptive_skills_eval.toml
```

## First Real Run

The starting checkpoint
`lentzl/rlm-prime-agent-qwen35-orientation-r1-20260806` is a private PEFT
adapter over `Qwen/Qwen3.5-2B`. Hydrate and merge it once on the rented machine
before PrimeRL starts; this preserves the orientation weights while allowing the
new run to create a fresh LoRA adapter:

```bash
export HF_TOKEN="$HF_KEY"
uv pip install peft huggingface_hub
uv run python deps/verifiers/scripts/merge_hf_lora_checkpoint.py \
  lentzl/rlm-prime-agent-qwen35-orientation-r1-20260806 \
  /ephemeral/models/qwen35-orientation-r1-merged
```

The merge resolves the EOS ID by tokenizing the source tokenizer's declared EOS
token, applies it to the merged model metadata, and validates every numeric
`eos_token_id` in the exported JSON files. Do not upload a model if this
validation fails; generation runtimes rely on these fields to stop at the chat
template's assistant-turn boundary.

Run these commands from the PrimeRL repository root after updating its
`deps/verifiers` submodule. Install this environment into the same environment
used by the PrimeRL orchestrator, then launch from that root so the configured
skill path resolves correctly:

```bash
uv sync --all-extras
uv pip install -e deps/verifiers/environments/adaptive_skill_stream_v1
uv run rl @ deps/verifiers/configs/prime_agent_qwen35_adaptive_skills_r2.toml
```

The `--all-extras` setup is required on x86_64 CUDA hosts. In particular, the
trainer imports `ring_flash_attn`, which imports `flash_attn` during startup.
A core-only sync can pass configuration validation but fail before model loading.

The run uses BF16 model and reduction dtypes, rank-16 LoRA, one GPU, and no
recursive child agents. The Qwen3.5 renderer is pinned with thinking disabled;
the local merged model path cannot be auto-mapped, and the 2B checkpoint's
Instruct-style default is the concise behavior we intend to preserve.
`save_session = true` gives Prime Agent a real session-local continual harness;
every rollout remains isolated and the task harvests its harness metrics before
cleanup. Four batches per rollout make this materially more expensive than the
orientation curriculum, so the first gate is a six-task held-out eval and a
short four-step smoke run before all 64 steps.

As in the one-GPU orientation run, the TOML expects a separately launched
OpenAI-compatible student server at ports 8000/8100. For the real run, serve
`/ephemeral/models/qwen35-orientation-r1-merged` as the base with LoRA enabled;
do not serve vanilla Qwen plus the old orientation adapter, because vLLM cannot
stack that adapter underneath the fresh training adapter. Set the server's model
context length to at least 32768; the four-batch Prime Agent trajectory does not
fit the orientation server's earlier 8192-token limit.

When the model is served from a local path, specify
`--model.tool-call-parser qwen3_coder`. Automatic parser selection cannot infer
Qwen from the path, and harness-based evals otherwise receive an HTTP 400 for
`tool_choice="auto"`. PrimeRL training uses its direct renderer client and does
not expose this omission, so retain the held-out eval as a setup gate.

## First Smoke Result

The four-step GRPO gate on 2026-08-06 completed on one RTX 6000 Ada in 37
minutes. All four optimizer steps were finite, mismatch KL stayed at or below
0.0002, peak trainer memory was 10.1 GiB, and no rollout errored. Two initially
sampled batches had zero within-group advantage and were safely resampled.

On the six held-out `standard` tasks, mean stream accuracy increased from about
0.02 before training to 0.25 after Step 4. Installed-skill transfer improved
most; stable-procedure transfer produced one partially correct stream. Ephemeral
adaptation remained at zero, and no rollout authored a valid portable skill.
This validates the end-to-end training path but also sets the next gate: increase
reward density and bootstrap evidence-based skill promotion before committing to
the 64-step recipe.
