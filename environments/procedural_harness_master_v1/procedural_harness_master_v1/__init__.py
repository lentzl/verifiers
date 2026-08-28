import os

from procedural_harness_master_v1.followup_feedback import FEEDBACK_SCHEMA_VERSION
from procedural_harness_master_v1.interaction_curriculum import (
    CURRICULUM_ENV_VAR,
    CURRICULUM_SCHEMA_VERSION,
    InteractionCurriculumPhase,
    configured_phase,
    curriculum_audit,
)
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

if os.environ.get(CURRICULUM_ENV_VAR) is not None:
    configured_phase()
    from procedural_harness_master_v1.interaction_curriculum import (
        install_interaction_curriculum,
    )
    from procedural_harness_master_v1.natural_yield_scaffold import (
        install_natural_yield_scaffold,
    )

    install_natural_yield_scaffold()
    install_interaction_curriculum()

__all__ = [
    "CURRICULUM_ENV_VAR",
    "CURRICULUM_SCHEMA_VERSION",
    "FEEDBACK_SCHEMA_VERSION",
    "SCAFFOLD_SCHEMA_VERSION",
    "InteractionCurriculumPhase",
    "ProceduralHarnessMasterConfig",
    "ProceduralHarnessMasterEnv",
    "ProceduralHarnessMasterTaskset",
    "configured_phase",
    "curriculum_audit",
    "keep_followup_feedback_response",
    "keep_scaffolded_natural_yield_response",
    "scaffold_audit",
]
