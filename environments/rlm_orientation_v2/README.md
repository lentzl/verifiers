# rlm-orientation-v2

### Overview
- **Environment ID**: `rlm-orientation-v2`
- **Short description**: Prime-native RLM orientation environment for reasoning through IPython instead of reasoning only in text.
- **Tags**: v2, rlm, sandbox, orientation

### Task
- **Type**: command harness through `harnesses.RLM`
- **Rubric overview**: Rewards exact final answers with IPython use, plus a small reward for expected REPL behavior such as defining a helper or using recovery code.

### Quickstart
```bash
prime eval run rlm-orientation-v2
```

### Notes
- This environment still does not mount the old helper library.
- The first target behavior is: think briefly, use IPython to verify or externalize state, then give a concise final answer.
