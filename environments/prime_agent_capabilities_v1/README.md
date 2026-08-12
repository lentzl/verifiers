# Prime Agent Capabilities V1

This environment exposes Prime Agent's native harness capabilities as executable,
parameterized tasks. It scores real IPython execution and persistent kernel state
from captured segments, and scores subagent lifecycle, quiescence, continual harness
state, and cancellation through the ACP metadata emitted by Prime Agent.

The environment is a capability and infrastructure layer, not a complete agent
curriculum. Pair it with natural task environments such as `oolong-synth-v1` and
with higher-level delegation environments for routing, ownership, fan-in, follow-up,
and recovery semantics.
