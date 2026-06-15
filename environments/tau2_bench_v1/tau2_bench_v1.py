import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, cast

os.environ.setdefault(
    "TAU2_DATA_DIR", str(Path.home() / ".cache" / "tau2-bench-v1" / "data")
)

import verifiers.v1 as vf
from openai import APIStatusError, AsyncOpenAI
from pydantic import ValidationError
from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    is_valid_agent_history_message,
)
from tau2.config import DEFAULT_MAX_ERRORS
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task as TauTask
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.registry import registry
from tau2.run import load_tasks
from tau2.user.user_simulator import (
    UserSimulator,
    is_valid_user_history_message,
)
from tau2.utils.llm_utils import to_litellm_messages
from tau2.utils.utils import DATA_DIR, format_time, get_now
from verifiers.utils.client_utils import load_prime_config

logger = logging.getLogger(__name__)

TAU2_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
TAU2_REVISION = "337326e62d8e0ca74c353b004a9c5d748e0ba914"
Tau2Domain = Literal["airline", "retail", "telecom", "telecom-workflow"]


class Tau2TasksetConfig(vf.TasksetConfig):
    domain: Tau2Domain = "telecom"


class Tau2Task(vf.Task):
    domain: Tau2Domain
    tau: TauTask


class Tau2Taskset(vf.Taskset[Tau2Task, Tau2TasksetConfig]):
    def load_tasks(self) -> list[Tau2Task]:
        data_domain = (
            "telecom"
            if self.config.domain == "telecom-workflow"
            else self.config.domain
        )
        marker = DATA_DIR / ".tau2_revision"
        if not (
            (DATA_DIR / "tau2" / "domains" / data_domain).exists()
            and marker.exists()
            and marker.read_text() == TAU2_REVISION
        ):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="tau2_bench_v1_") as temp_dir:
                subprocess.run(
                    ["git", "init", temp_dir],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        temp_dir,
                        "fetch",
                        "--depth",
                        "1",
                        TAU2_REPOSITORY,
                        TAU2_REVISION,
                    ],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    [
                        "git",
                        "-C",
                        temp_dir,
                        "checkout",
                        "FETCH_HEAD",
                        "--",
                        "data",
                    ],
                    check=True,
                    capture_output=True,
                )
                shutil.copytree(Path(temp_dir) / "data", DATA_DIR, dirs_exist_ok=True)
                marker.write_text(TAU2_REVISION)

        environment = registry.get_env_constructor(self.config.domain)()
        system_prompt = SYSTEM_PROMPT.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=environment.get_policy(),
        )
        return [
            Tau2Task(
                idx=index,
                name=task.id,
                instruction=DEFAULT_FIRST_AGENT_MESSAGE.content or "",
                system_prompt=system_prompt,
                domain=self.config.domain,
                tau=task,
            )
            for index, task in enumerate(
                load_tasks(
                    task_set_name=self.config.domain,
                    task_split_name="base",
                )
            )
        ]

    @vf.reward
    async def tau2_reward(self, task: Tau2Task, trace: vf.Trace) -> float:
        simulation = SimulationRun.model_validate(trace.info["tau2"]["simulation"])
        reward = await asyncio.to_thread(
            evaluate_simulation,
            simulation=simulation,
            task=task.tau,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=task.domain,
        )
        trace.info["tau2"]["evaluation"] = reward.model_dump(mode="json")
        return float(reward.reward)


class Tau2HarnessConfig(vf.HarnessConfig):
    id: Literal["tau2-bench-v1"] = "tau2-bench-v1"
    runtime: vf.RuntimeConfig = vf.SubprocessConfig()


class Tau2Harness(vf.Harness[Tau2HarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_TASK_TOOLS = False

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        del runtime, mcp_urls
        task = cast(Tau2Task, trace.task)
        initial_state = task.tau.initial_state
        initial_history = (
            deepcopy(initial_state.message_history or []) if initial_state else []
        )
        history_start = datetime.now() - timedelta(seconds=len(initial_history))
        for index, message in enumerate(initial_history):
            message.turn_idx = None
            message.timestamp = format_time(history_start + timedelta(seconds=index))
        tau_messages = list(initial_history)
        termination_reason = TerminationReason.AGENT_ERROR
        num_errors = 0
        started_at = datetime.now().isoformat()

        try:
            environment = registry.get_env_constructor(task.domain)()
            environment.set_state(
                initialization_data=(
                    initial_state.initialization_data if initial_state else None
                ),
                initialization_actions=(
                    initial_state.initialization_actions if initial_state else None
                ),
                message_history=initial_history,
            )

            prime = load_prime_config()
            user_args: dict[str, object] = {
                "temperature": 0,
                "api_base": prime["inference_url"],
                "api_key": os.getenv("PRIME_API_KEY") or prime["api_key"],
                "timeout": 86400,
            }
            team_id = os.getenv("PRIME_TEAM_ID") or prime.get("team_id")
            if team_id:
                user_args["extra_headers"] = {"X-Prime-Team-ID": team_id}
            try:
                user_tools = environment.get_user_tools()
            except ValueError:
                user_tools = None
            user = UserSimulator(
                tools=user_tools,
                instructions=str(task.tau.user_scenario),
                llm="custom_openai/openai/gpt-4.1",
                llm_args=user_args,
            )
            user_history = initial_history
            if initial_history and (
                isinstance(initial_history[-1], AssistantMessage)
                or isinstance(initial_history[-1], ToolMessage)
                and initial_history[-1].requestor == "user"
            ):
                user_history = initial_history[:-1]
            user_state = user.get_init_state(
                [
                    message
                    for message in user_history
                    if is_valid_user_history_message(message)
                ]
            )

            system_prompt, _ = self.resolve_prompt(task)
            messages: list[dict[str, object]] = []
            pending_user_input: Message | None = None
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if initial_history:
                messages.extend(
                    to_litellm_messages(
                        [
                            message
                            for message in initial_history
                            if is_valid_agent_history_message(message)
                        ]
                    )
                )
            else:
                first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
                first_message.timestamp = get_now()
                tau_messages.append(first_message)
                messages.extend(to_litellm_messages([first_message]))
                pending_user_input = first_message

            if initial_history:
                last_message = initial_history[-1]
                if isinstance(last_message, AssistantMessage):
                    if last_message.tool_calls:
                        tool_messages = [
                            environment.get_response(tool_call)
                            for tool_call in last_message.tool_calls
                        ]
                        tau_messages.extend(tool_messages)
                        num_errors += sum(
                            tool_message.error for tool_message in tool_messages
                        )
                        messages.extend(
                            {
                                "role": "tool",
                                "tool_call_id": tool_message.id,
                                "content": tool_message.content or "",
                            }
                            for tool_message in tool_messages
                        )
                        if num_errors >= DEFAULT_MAX_ERRORS:
                            termination_reason = TerminationReason.TOO_MANY_ERRORS
                            return vf.ProgramResult(exit_code=0, stdout="", stderr="")
                    else:
                        pending_user_input = last_message
                elif isinstance(last_message, UserMessage):
                    if UserSimulator.is_stop(last_message):
                        termination_reason = TerminationReason.USER_STOP
                        trace.stop("user_completed")
                        return vf.ProgramResult(exit_code=0, stdout="", stderr="")
                    if last_message.tool_calls:
                        tool_messages = [
                            environment.get_response(tool_call)
                            for tool_call in last_message.tool_calls
                        ]
                        tau_messages.extend(tool_messages)
                        num_errors += sum(
                            tool_message.error for tool_message in tool_messages
                        )
                        if num_errors >= DEFAULT_MAX_ERRORS:
                            termination_reason = TerminationReason.TOO_MANY_ERRORS
                            return vf.ProgramResult(exit_code=0, stdout="", stderr="")
                        pending_user_input = (
                            tool_messages[0]
                            if len(tool_messages) == 1
                            else MultiToolMessage(
                                role="tool", tool_messages=tool_messages
                            )
                        )
                elif (
                    isinstance(last_message, ToolMessage)
                    and last_message.requestor == "user"
                ):
                    pending_user_input = last_message

            async with AsyncOpenAI(
                base_url=endpoint, api_key=secret, timeout=None
            ) as client:
                while True:
                    if pending_user_input is not None:
                        user_message, user_state = await asyncio.to_thread(
                            user.generate_next_message,
                            pending_user_input,
                            user_state,
                        )
                        tau_messages.append(user_message)
                        while user_message.tool_calls:
                            tool_messages = [
                                environment.get_response(tool_call)
                                for tool_call in user_message.tool_calls
                            ]
                            tau_messages.extend(tool_messages)
                            num_errors += sum(
                                tool_message.error for tool_message in tool_messages
                            )
                            if num_errors >= DEFAULT_MAX_ERRORS:
                                termination_reason = TerminationReason.TOO_MANY_ERRORS
                                break
                            user_message, user_state = await asyncio.to_thread(
                                user.generate_next_message,
                                (
                                    tool_messages[0]
                                    if len(tool_messages) == 1
                                    else MultiToolMessage(
                                        role="tool",
                                        tool_messages=tool_messages,
                                    )
                                ),
                                user_state,
                            )
                            tau_messages.append(user_message)

                        if num_errors >= DEFAULT_MAX_ERRORS:
                            break
                        if UserSimulator.is_stop(user_message):
                            termination_reason = TerminationReason.USER_STOP
                            trace.stop("user_completed")
                            break
                        messages.append(
                            {
                                "role": "user",
                                "content": user_message.content or "",
                            }
                        )
                        pending_user_input = None

                    try:
                        completion = await client.chat.completions.create(
                            model=ctx.model,
                            messages=messages,
                            tools=[
                                tool.openai_schema for tool in environment.get_tools()
                            ],
                        )
                    except APIStatusError as error:
                        if trace.stop_condition is None:
                            raise vf.ProgramError(str(error)) from error
                        termination_reason = (
                            TerminationReason.CONTEXT_WINDOW_EXCEEDED
                            if trace.stop_condition == "context_length"
                            else TerminationReason.TIMEOUT
                            if trace.stop_condition == "harness_timeout"
                            else TerminationReason.MAX_STEPS
                        )
                        break

                    message = completion.choices[0].message
                    messages.append(
                        {
                            **message.model_dump(exclude_none=True),
                            "content": message.content or "",
                        }
                    )
                    tool_calls = [
                        ToolCall(
                            id=call.id,
                            name=call.function.name,
                            arguments=json.loads(call.function.arguments),
                            requestor="assistant",
                        )
                        for call in message.tool_calls or []
                    ]
                    assistant_message = AssistantMessage(
                        role="assistant",
                        content=message.content,
                        tool_calls=tool_calls or None,
                    )
                    tau_messages.append(assistant_message)
                    try:
                        assistant_message.validate()
                    except ValueError:
                        termination_reason = TerminationReason.AGENT_ERROR
                        break

                    if tool_calls:
                        tool_messages = [
                            environment.get_response(tool_call)
                            for tool_call in tool_calls
                        ]
                        tau_messages.extend(tool_messages)
                        num_errors += sum(
                            tool_message.error for tool_message in tool_messages
                        )
                        messages.extend(
                            {
                                "role": "tool",
                                "tool_call_id": tool_message.id,
                                "content": tool_message.content or "",
                            }
                            for tool_message in tool_messages
                        )
                        if num_errors >= DEFAULT_MAX_ERRORS:
                            termination_reason = TerminationReason.TOO_MANY_ERRORS
                            break
                        continue

                    pending_user_input = assistant_message
        except (json.JSONDecodeError, ValidationError) as error:
            logger.exception("Tau2 received invalid JSON during rollout %s", trace.id)
            raise vf.ProgramError("Tau2 received invalid JSON") from error
        except asyncio.CancelledError:
            termination_reason = TerminationReason.TIMEOUT
            raise
        finally:
            ended_at = datetime.now().isoformat()
            trace.info["tau2"] = {
                "simulation": SimulationRun(
                    id=trace.id,
                    task_id=task.tau.id,
                    messages=tau_messages,
                    termination_reason=termination_reason,
                    timestamp=started_at,
                    start_time=started_at,
                    end_time=ended_at,
                    duration=0.0,
                    agent_cost=0.0,
                    user_cost=0.0,
                ).model_dump(mode="json")
            }
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


def load_taskset(config: Tau2TasksetConfig) -> Tau2Taskset:
    return Tau2Taskset(config)


def load_harness(config: Tau2HarnessConfig) -> Tau2Harness:
    return Tau2Harness(config)
