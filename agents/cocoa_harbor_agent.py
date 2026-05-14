"""
Harbor BaseInstalledAgent wrapper for CocoaAgent.
CocoaAgent is pre-installed in the Docker image — no install step needed.
Harness files live in /harness/ inside the container.
"""
import json
import os
import shlex

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Paths inside the container
PYTHON = "/opt/python3.12/bin/python"
RUNNER = "/harness/run_task.py"
DEFAULT_CONFIG = "/harness/configs/harbor-config.json"


class CocoaHarborAgent(BaseInstalledAgent):
    """Wraps CocoaAgent as a Harbor InstalledAgent."""

    # run_task.py writes /logs/agent/trajectory.json (ATIF-shaped) + cocoa-agent.txt
    SUPPORTS_ATIF: bool = True

    @staticmethod
    def name() -> str:
        return "cocoa-agent"

    async def install(self, environment: BaseEnvironment) -> None:
        """Cocoa harness and runtime are baked into the task image."""
        pass

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        config_path = getattr(self, "COCOA_CONFIG", None) or os.environ.get(
            "COCOA_CONFIG", DEFAULT_CONFIG
        )
        task_name = os.environ.get("HARBOR_TASK_NAME", "harbor-task")

        cmd = (
            f"{PYTHON} {RUNNER}"
            f" --instruction {shlex.quote(instruction)}"
            f" --task-name {shlex.quote(task_name)}"
            f" --config {shlex.quote(config_path)}"
            f" --output /logs/agent/result.json"
        )

        env: dict[str, str] = {}
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "LLM_BASE_URL",
            "LLM_MODEL",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "COCOA_MAX_ITERATIONS",
            "COCOA_LOG_LEVEL",
            "HARBOR_COCOA_LOG_LEVEL",
            "HARBOR_KEEP_RESULT_IMAGES",
        ):
            val = getattr(self, key, None) or os.environ.get(key, "")
            if val:
                env[key] = str(val)
        if self.model_name:
            env["LLM_MODEL"] = str(self.model_name)

        await self.exec_as_agent(environment, command=cmd, env=env or None)

    def populate_context_post_run(self, context: AgentContext) -> None:
        result_path = self.logs_dir / "result.json"
        if not result_path.exists():
            return
        try:
            result = json.loads(result_path.read_text())
            cost = result.get("api_cost_stats", {}) or {}
            if cost:
                context.n_input_tokens = cost.get("total_input_tokens")
                context.n_output_tokens = cost.get("total_output_tokens")
                context.cost_usd = cost.get("total_cost_usd")
        except Exception:
            pass
