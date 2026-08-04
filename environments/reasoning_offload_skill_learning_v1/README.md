# reasoning-offload-skill-learning-v1

A multi-agent Verifiers environment that trains a model to externalize reusable work
into portable, instruction-first Agent Skills.

Each episode has three fresh contexts:

- The author sees only solved discovery examples and proposes a canonical SKILL.md
  plus scripts and optional supporting resources.
- The skill user receives that artifact under /task/agent-skills/<name>/, discovers it
  from SKILL.md, and solves a held-out task.
- The baseline user solves the same task through an otherwise identical fresh runtime
  without the candidate artifact.

Only the author trace is trainable. Its skill_utility reward is
skill_user_correct - baseline_correct, so a redundant skill receives zero and a
harmful skill receives a negative score. Package validity, consultation, and both arm
scores are retained as metrics. Invalid packages become non-leaking author feedback
instead of failing the episode.

The learned contract deliberately uses only the common Agent Skills core: a kebab-case
directory, SKILL.md with name and description frontmatter, and progressively disclosed
files under scripts/, references/, or assets/. The model does not author Prime Python
packaging, OpenAI UI metadata, or Anthropic-specific tool policy. Those are
deterministic deployment adapters, not learned semantics. The downstream trial
likewise reads the canonical artifact directly instead of importing a Prime-only
callable.

Keeping only the author trainable is intentional for SDPO: author and consumer replies
must not become demonstrations for each other's unlike prompts. A separate consumer
curriculum should train skill discovery and use with pre-existing portable skills; both
rungs can share the same model while retaining role-homogeneous rollout groups.

The initial taskset carries solved synthetic discovery examples as bootstrap evidence.
The same experiences wire field can later be populated from successful natural agent
traces without changing the learning environment or reward.

    uv run eval @ configs/reasoning_offload_skill_learning.toml
