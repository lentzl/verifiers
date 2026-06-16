# rlm-orientation-v1

### Overview
- **Environment ID**: `rlm-orientation-v1`
- **Short description**: Minimal Prime-native RLM orientation environment that requires `rlm-harness` to use the IPython tool.
- **Tags**: v1, rlm, sandbox, orientation

### Task
- **Type**: command harness through `harnesses.RLM`
- **Rubric overview**: Rewards exact stdout answer only when RLM session metrics show IPython use.

### Quickstart
```bash
prime eval run rlm-orientation-v1
```

### Notes
- This is the first no-skills smoke target for Prime-native RLM training.
- It intentionally does not mount or depend on the old local helper library.
