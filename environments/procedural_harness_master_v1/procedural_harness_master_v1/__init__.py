import os

from procedural_harness_master_v1.followup_feedback import FEEDBACK_SCHEMA_VERSION
from procedural_harness_master_v1.natural_yield_scaffold import (
    SCAFFOLD_SCHEMA_VERSION,
    keep_scaffolded_natural_yield_response,
    scaffold_audit,
)
from procedural_harness_master_v1.taskset import (
    ProceduralHarnessMasterConfig,
    ProceduralHarnessMasterEnv,
    ProceduralHarnessMasterTaskset,
    keep_followup_feedback_response,
)

if os.environ.get("PROCEDURAL_NATURAL_YIELD_SCAFFOLD") == "1":
    from procedural_harness_master_v1.natural_yield_scaffold import (
        install_natural_yield_scaffold,
    )

    install_natural_yield_scaffold()

__all__ = [
    "FEEDBACK_SCHEMA_VERSION",
    "SCAFFOLD_SCHEMA_VERSION",
    "ProceduralHarnessMasterConfig",
    "ProceduralHarnessMasterEnv",
    "ProceduralHarnessMasterTaskset",
    "keep_followup_feedback_response",
    "keep_scaffolded_natural_yield_response",
    "scaffold_audit",
]
