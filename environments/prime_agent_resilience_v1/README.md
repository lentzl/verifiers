# Prime Agent Resilience V1

This environment is the fault-recovery supplement to the frozen Prime Agent mastery
battery. It does not replace or modify that regression comparator.

The three families expose only behavior visible in normal Prime Agent traces:

- `malformed_result_repair`: a child first returns a schema-invalid result; the
  coordinator must diagnose it, send one constrained correction to the retained
  child, and accept the repaired result.
- `delayed_result`: a child reports that work started, runs a supplied delayed
  executable, and later sends the result; the coordinator must yield without
  polling or replacing the child.
- `message_type_repair`: a child receives a real `TypeError` after passing a mapping
  to `agent_message.send`, then changes the call and successfully sends serialized
  JSON while preserving its computed payload.

Strict success requires the exact final answer, one retained named child, authentic
message order, family-specific recovery evidence, no repeated cells, no roster or
observation polling, and no replacement delegation. Train and held-out variants are
deterministic and disjoint.
