# tau2-bench-v1

### Overview
- **Environment ID**: `tau2-bench-v1`
- **Short description**: Tau2's multi-domain customer-service benchmark as a native Verifiers v1 taskset and harness.
- **Tags**: tau2, tool-agent-user, tool-use, multi-turn, user-sim, v1

### Datasets
- **Primary dataset(s)**: Tau2 base tasks for the `airline`, `retail`, `telecom`, and `telecom-workflow` domains.
- **Source links**: https://github.com/sierra-research/tau2-bench
- **Split sizes**: Determined by Tau2's pinned `base` split for the selected domain.

### Task
- **Type**: Multi-turn tool use with an LLM user simulator.
- **Output format expectations (optional)**: Natural-language customer support responses and OpenAI-compatible tool calls.
- **Rubric overview**: Official Tau2 evaluation of database state, environment assertions, actions, and required communication.

### Quickstart
Set `PRIME_API_KEY` for inference and `PRIME_TEAM_ID` for team billing.

```bash
uv run eval tau2-bench-v1 \
  --harness.id tau2-bench-v1 \
  -m openai/gpt-4.1-mini \
  -n 1 -r 1
```

Select a domain:

```bash
uv run eval tau2-bench-v1 \
  --harness.id tau2-bench-v1 \
  --taskset.domain retail
```

The harness sends evaluated-model requests through Verifiers' interception endpoint
while running Tau's assistant tools and user simulator directly in-process.

### Taskset Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `domain` | `airline \| retail \| telecom \| telecom-workflow` | `telecom` | Tau2 domain and task set to evaluate. |

### Harness Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `runtime` | `RuntimeConfig` | `subprocess` | Runtime used for the interception-backed harness. |

Rollout turn, token, and wall-clock limits are supplied through the standard
Verifiers eval config rather than owned by this environment.

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `tau2_reward` | Official Tau2 scalar reward for the completed simulation. |

### Changelog

#### v0.2.0
- Rebuilt Tau2 around the native Verifiers v1 taskset and harness APIs.
- Run Tau tools and the user simulator in-process without MCP or legacy environment adapters.
- Preserve selectable classic Tau2 domains and official Tau2 scoring.
- Store the simulation and evaluation breakdown in `trace.info["tau2"]`.
