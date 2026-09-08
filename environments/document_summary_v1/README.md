# document-summary-v1

This environment measures an applied English document workflow rather than a
language-learning exercise. Three chapters of a fictional operations playbook are
summarized into concise, source-grounded bullet points.

`worker_probe` gives one chapter directly to the existing terminal-worker lineage.
`owner` delegates all three chapter jobs through native Prime Agent children and
assembles their typed reports. Job files carry the full summarization contract so a
coordinator cannot accidentally paraphrase away critical worker instructions.

`evidence_probe` uses two phases of the same terminal worker in its existing
Prime Agent session. The worker reads `source.md`, writes its own source-linked
`notes.md`, then stops. The workflow gate saves those bytes as `notes-extracted.md`
and asks the worker to read them and write `summary.md`. Notes preserve obligations
and qualifications without the final-summary word limit. The original source
remains available. No teacher notes are provided, and no new expert is introduced.

The gate enforces artifact presence, source-ID presence in notes and phase order,
not content quality. Missing IDs produce a one-write repair instruction because
repeated `write_text` calls overwrite earlier records. Continuation interception
removes the generic failed-quality-gate wrapper and describes the next file step;
it supplies no source facts or authored answers. Its
completion reward only means both artifacts exist; it must not be reported as a
summary success rate. The original source, scratch notes, captured notes and final
summary are saved in `trace.info`. Inspect source-to-notes and notes-to-summary
meaning separately. Keyword-group proxies, paragraph-ID presence and word counts
are diagnostics, not semantic certification. The captured file is retained by the
workflow, not a security boundary against a worker rewriting its own sandbox.

For an ordinary UTF-8 text or Markdown chapter, use `evidence_probe` with
`--env.taskset.chapter-path /absolute/path/to/chapter.md`. Blank-line-separated
paragraphs receive stable in-document IDs and content hashes. The file is read by
the evaluator and copied into the worker's isolated workspace. This path provides
no teacher notes or keyword reference groups: inspect the source, captured notes
and summary for factual faithfulness. It is not an automatic semantic pass and
does not turn a previously inspected document into fresh confirmation.

The runtime includes source paragraphs, stable IDs, and output contracts only.
Hidden fact groups are used by the evaluator as keyword-group proxies;
they are never written into the sandbox.

The `confirmation` split contains a separate facilities handbook used for the
completed post-training text probes. Its chapters did not enter that SFT exporter,
but the observed results now inform development. Use a new document for future
fresh-confirmation claims; do not reuse this split as unseen evidence.
