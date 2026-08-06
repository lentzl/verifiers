# ipython-foundations-v1

This curriculum trains the notebook semantics that later Prime Agent capabilities
depend on. Each rollout keeps one Prime Agent session and IPython kernel alive across
three related requests.

- `assignment` makes a deliberately silent assignment useful in a later call;
- `state` removes the source file after the first request, forcing cross-turn reuse;
- `recovery` executes stale Python operations, exposes their real IPython errors, and
  requires a changed corrective call using state that survived the failure;
- `subprocess` preserves a downloaded document path, exposes a real nonzero process
  result, and requires complete result inspection plus an error-directed CLI repair.

Stream answer accuracy is the primary reward. A smaller notebook-semantics reward
requires the family-specific state behavior and rejects consecutive identical cells.
Subprocess streams also reject raw-byte PDF fallbacks and require the corrected
`pdftotext` stdout convention.
Training uses explicit operational scaffolding without revealing answers; held-out
variants use standard instructions.

The recovery matrix covers `NameError`, missing imports, omitted `await`, bytes/text
mismatches, confusing `CompletedProcess` with its stdout, path quoting, missing files,
incorrect dictionary keys, empty parser output, and a nonzero subprocess promoted to
`CalledProcessError`. The environment never inserts fabricated traceback text: the
prompt supplies a stale operation, Prime Agent runs it in the persistent kernel, and
the next sampled action receives the kernel's actual feedback.

## Develop

```bash
uv pip install -e environments/ipython_foundations_v1
uv run eval ipython-foundations-v1 \
  --env.agent.harness.id prime-agent \
  --env.agent.harness.save-session true \
  --env.agent.runtime.type docker \
  --taskset.split eval \
  -n 10
```

## First Run

The PrimeRL recipe starts directly from the private, self-contained adaptive-skills
snapshot `lentzl/rlm-prime-agent-qwen35-adaptive-skills-smoke-r1-20260806`.
Run from the PrimeRL repository after updating its Verifiers submodule and installing
this environment:

```bash
uv sync --all-extras
uv pip install -e deps/verifiers/environments/ipython_foundations_v1
uv run hf download \
  lentzl/rlm-prime-agent-qwen35-adaptive-skills-smoke-r1-20260806 \
  --revision f453c92bc67453c03c82b6e40481abc71e1c3772 \
  --local-dir /ephemeral/models/qwen35-adaptive-skills-smoke-r1
uv run inference @ \
  configs/debug/ipython-foundations/inference.toml
```

After the inference router is healthy on port 8000 and its engine is healthy on
port 8100, launch the bounded smoke:

```bash
uv run rl @ configs/debug/ipython-foundations/rl.toml \
  --max-steps 4 \
  --output-dir /ephemeral/outputs/prime-agent-qwen35-ipython-foundations-smoke-r1
```

The smoke gate must complete all four optimizer steps and the six held-out streams
before the full 48-step recipe is launched. The colocated inference profile reserves
`0.19` of the GPU, uses a 32768-token context, and enables rank-16 LoRA weight updates.
The environment provides no installed task skill: this rung measures the model's use
of Prime Agent's native persistent IPython kernel.
