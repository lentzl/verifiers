---
name: portable-record-normalization
description: Normalize free-form record labels and sum integer amounts by canonical label.
---

# Portable Record Normalization

Use this skill when records contain `label` and `amount` fields and the task asks
for totals grouped by a stable canonical label. Canonicalization strips outer
whitespace, lowercases text, replaces each run of non-alphanumeric characters
with one hyphen, and removes leading or trailing hyphens.

From Prime Agent IPython:

```python
result = portable_record_normalization.summarize(records)
```

`summarize(records)` returns a dictionary sorted by canonical label. Read records
from the task's actual input file; do not transcribe them into the call.
