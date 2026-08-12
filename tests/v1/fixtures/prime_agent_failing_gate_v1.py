"""Prime Agent gate failure metadata fixture."""

import verifiers.v1 as vf
from verifiers.v1.utils.prime_agent_metadata import gate_failure_reported


class PrimeAgentFailingGateTask(vf.Task):
    @vf.reward(weight=1.0)
    async def failing_gate_is_loud(self, trace: vf.Trace) -> float:
        return float(gate_failure_reported(trace))


class PrimeAgentFailingGateEnv(vf.SingleAgentEnv):
    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn(
                "Create a file named gate.txt containing ok, then reply with exactly DONE."
            )


class PrimeAgentFailingGateTaskset(
    vf.Taskset[PrimeAgentFailingGateTask, vf.TasksetConfig]
):
    def load(self) -> list[PrimeAgentFailingGateTask]:
        return [
            PrimeAgentFailingGateTask(
                vf.TaskData(
                    idx=0,
                    prompt=None,
                    system_prompt="Follow instructions exactly when handling gates.",
                )
            )
        ]


__all__ = ["PrimeAgentFailingGateEnv", "PrimeAgentFailingGateTaskset"]
