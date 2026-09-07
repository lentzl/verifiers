# document-translation-v1

A single authored English-to-German development task for the native Prime Agent
harness. The root session owns the document artifact and delegates three coherent
source jobs to named terminal workers. Workers return typed translation rows through
`agent_message`; the owner validates and persists the complete artifact.

The authored German reference remains in host-side task data for diagnostic scoring.
It is never written into the runtime or included in an agent prompt.

Run the deterministic task tests from the Verifiers checkout:

```bash
uv run pytest environments/document_translation_v1/tests -q
```

For a live run, select the built-in Prime Agent harness and configure its completion
gate as `/workspace/document-translation-v1/completion_gate.py`.
