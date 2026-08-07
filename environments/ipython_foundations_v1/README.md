# ipython-foundations-v1

This curriculum trains the notebook semantics that later Prime Agent capabilities
depend on. The first two families isolate one-request control behavior; the remaining
families keep one Prime Agent session and IPython kernel alive across related requests.

- `completion` obtains one non-empty IPython result and returns it immediately;
- `assignment` makes a deliberately silent assignment useful in a later call;
- `state` removes the source file after the first request, forcing cross-turn reuse;
- `recovery` executes stale Python operations, exposes their real IPython errors, and
  requires a changed corrective call using state that survived the failure;
- `subprocess` preserves a downloaded document path, exposes a real nonzero process
  result, and requires complete result inspection plus an error-directed CLI repair.
- `document_recovery` alternates direct absolute paths and structured download results,
  then exposes a real package/import/API failure before requiring introspection,
  extraction, and a grounded summary from retained text.
- `file_processing` isolates short, single-request document trajectories that inspect
  structured results, retain paths, select parsers by file evidence, and convert real
  failures into bounded corrections rather than repeated cells.

Notebook process is the primary reward; answer accuracy has half its weight. The
process score gives partial credit for completed repair stages and discounts repeated
unchanged cells and unnecessary extra calls, while `process_aligned` remains the strict
diagnostic. This prevents
correct answers produced by rereading or recomputing from dominating trajectories that
actually use persistent state. Subprocess streams also penalize raw-byte PDF fallbacks
and repeated failures while rewarding complete result inspection, a changed operation,
and the corrected `pdftotext` stdout convention.
Training can use guided invariant-level hints or explicit operational scaffolding
without revealing answers; held-out variants use standard instructions. Guided hints
name the notebook behavior but deliberately omit executable code so a rung can fade
exact demonstrations before its held-out gate.

The recovery matrix covers `NameError`, missing imports, omitted `await`, bytes/text
mismatches, confusing `CompletedProcess` with its stdout, path quoting, missing files,
incorrect dictionary keys, empty parser output, and a nonzero subprocess promoted to
`CalledProcessError`. An intentionally unavailable dependency requires one evidenced
availability check followed by a structured limitation instead of repeated imports,
API invention, or arbitrary installation. The environment never inserts fabricated traceback text: the
prompt supplies a stale operation, Prime Agent runs it in the persistent kernel, and
the next sampled action receives the kernel's actual feedback.

Document recovery uses small executable parser fixtures with realistic distribution,
import, and public-API boundaries: `pymupdf` maps to `fitz.open`, while
`pdfminer.six` maps to `pdfminer.high_level.extract_text`. The fixtures keep the rung
offline and deterministic; they are not prompt-inserted traceback strings. Process
metrics separately expose source inspection, operation revision, package/API
introspection, text extraction, and summary reuse. Repeated error signatures,
additional repair errors, unnecessary list/download calls, and raw-byte decoding all
reduce reward. This keeps a correct final summary from hiding an API-guessing loop.

File processing covers plain text and Markdown with `Path.read_text`, CSV with the
standard library, JSON with `json.load`, PDF page extraction, DOCX paragraphs, and
unknown formats through MIME and magic-byte inspection. Controlled failures include a
missing `pdftotext`, wrong encoding, malformed CSV, invalid JSON, scanned PDFs with no
extractable text, and password-protected PDFs. The process score separately measures
structured-result inspection, `download["path"]` selection and reuse, parser choice,
traceback-informed revision, progress after silent imports, nonempty extraction, and
evidenced terminal limitations. This makes parser/API knowledge secondary to the
intended control loop: inspect, retain, attempt, observe, constrain, and proceed.

Run `prime_agent_qwen35_ipython_recovery_eval.toml` for the new capability and
`prime_agent_qwen35_ipython_foundation_regression_eval.toml` for completion, silent
assignment, and cross-message continuity. Omitted-`await` remains in the held-out
recovery matrix, while direct-path document variants detect unnecessary acquisition
calls. These diagnostics must be reviewed independently.

Run `prime_agent_qwen35_file_processing_eval.toml` as a separate gate for typed file
handling. It enumerates the full held-out scenario matrix and records
`grounded_file_answer` independently from process alignment, so a memorized answer
cannot hide a failed extraction and a sound unsupported-file diagnosis is not scored
as an extraction failure.

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
port 8100, evaluate the held-out continuity tasks and launch the bounded smoke:

```bash
uv run eval @ \
  deps/verifiers/configs/prime_agent_qwen35_ipython_continuity_eval.toml
uv run rl @ configs/debug/ipython-foundations/continuity-rl.toml \
  --max-steps 4 \
  --output-dir /ephemeral/outputs/prime-agent-qwen35-ipython-continuity-smoke-r1
```

The foundations are trained as three separate gates: immediate completion, one-request
silent assignment recovery, and cross-request state reuse. Each next rung starts from
the previous rung's merged weight snapshot only after held-out standard-instruction
evaluation improves its family-specific process metric without increasing tool calls.
Recovery and subprocess families are introduced only after all three gates. The
colocated inference profile reserves `0.17` of the GPU and enables rank-16 LoRA weight
updates. The environment provides no installed task skill: this rung measures the
model's use of Prime Agent's native persistent IPython kernel.
