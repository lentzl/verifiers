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

Run these commands from the PrimeRL repository root after updating its
`deps/verifiers` submodule. Install this environment into the same environment
used by the PrimeRL orchestrator, then launch from that root so the configured
skill path resolves correctly:

```bash
uv pip install -e deps/verifiers/environments/adaptive_skill_stream_v1
uv run rl @ deps/verifiers/configs/prime_agent_qwen35_adaptive_skills_r2.toml
```

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
stack that adapter underneath the fresh training adapter.
