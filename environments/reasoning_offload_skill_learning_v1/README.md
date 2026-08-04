# reasoning-offload-skill-learning-v1

A multi-agent Verifiers environment that trains a model to externalize reusable work
into portable, instruction-first Agent Skills.

Each episode has three roles in fresh contexts:

- The author sees only solved discovery examples and proposes a canonical SKILL.md
  plus scripts and optional supporting resources.
- The skill user receives that artifact under /task/agent-skills/<name>/, reads
  SKILL.md, and executes its referenced script against absolute task-file paths.
- The baseline user solves each hidden task through an otherwise identical fresh
  runtime without the candidate artifact.

Only the author trace is trainable. A configurable hidden panel contains one to three
fresh tasks from the same latent family. Its skill_utility reward is the exact average
of skill_user_correct - baseline_correct across that panel, so a redundant skill
receives zero and a harmful skill receives a negative score. Package validity,
consultation rate, panel size, and both arm scores are retained as metrics. Invalid
packages become non-leaking author feedback instead of failing the episode.

A downstream agent that exhausts its configured execution budget is scored as an
incorrect arm rather than failing every sibling trace in the episode. The original
error remains in trace metadata. A failed skill user can therefore penalize the
candidate, while a failed matched baseline suppresses the author reward because its
marginal utility is not identifiable.

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
