"""
CodexAgent — wraps the OpenAI Codex CLI for use inside Harbor containers.

Implements the agent protocol expected by harness/run_task.py:
  - setup_environment(task)
  - run_task(task) -> dict
  - cleanup_environment()

Codex CLI is invoked as a subprocess with --json output, and the JSONL
events are parsed into the standard result dict format.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)


class CodexAgent:
    agent_type = "codex"

    def __init__(self, config: dict):
        self.config = config
        self.model = (
            config.get("controller", {}).get("args", {}).get("model")
            or os.environ.get("LLM_MODEL")
            or "codex-mini-latest"
        )
        self.api_key = (
            config.get("controller", {}).get("args", {}).get("api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.base_url = (
            config.get("controller", {}).get("args", {}).get("base_url")
            or os.environ.get("OPENAI_BASE_URL", "")
            or os.environ.get("LLM_BASE_URL", "")
        )
        self.max_iterations = (
            config.get("sandbox", {}).get("max_iterations")
            or int(os.environ.get("COCOA_MAX_ITERATIONS", "0"))
            or None
        )
        self._work_dir = "/tmp/codex-workspace"

    def setup_environment(self, task: dict) -> None:
        os.makedirs(self._work_dir, exist_ok=True)

    def run_task(self, task: dict) -> dict:
        instruction = task.get("instruction", "") or task.get("description", "")
        wrapped = self._wrap_instruction(instruction)

        cmd = self._build_command(wrapped)
        env = self._build_env()

        logger.info("Running codex: %s", " ".join(cmd))
        start = time.time()

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self._work_dir,
            env=env,
            timeout=self._timeout(),
        )

        elapsed = time.time() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            logger.warning("Codex exited %d: %s", proc.returncode, stderr[:500])

        result = self._parse_output(stdout)
        result["task_name"] = task.get("task_name", "unknown")
        result["elapsed_seconds"] = elapsed

        # Codex may exit non-zero due to MCP errors even when the task succeeded.
        # Treat it as completed if we got a task_result or conversation.
        if proc.returncode == 0 or result.get("task_result") or result.get("conversation"):
            result["status"] = "completed"
        else:
            result["status"] = "error"
            result["error"] = stderr[:2000] or f"codex exited with code {proc.returncode}"

        # Collect any files created in the workspace
        file_contents = self._collect_artifacts()
        if file_contents:
            artifact_text = ""
            for fname, content in file_contents.items():
                artifact_text += f"\n\n=== FILE: {fname} ===\n{content}\n=== END FILE ==="
            result["task_result"] = (result.get("task_result") or "") + artifact_text
            result["artifacts"] = file_contents

        return result

    def cleanup_environment(self) -> None:
        if os.path.isdir(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)

    # ── internal helpers ─────────────────────────────────────────────────

    def _wrap_instruction(self, instruction: str) -> str:
        return (
            "Complete the following task immediately without asking any clarifying questions. "
            "Make reasonable assumptions and proceed. "
            "For any visual, interactive, or game output, create a single self-contained HTML file "
            "(no external dependencies) rather than a terminal or desktop application. "
            "After creating any files, print each file's complete contents to stdout "
            "wrapped like this: === FILE: <filename> ===\n<contents>\n=== END FILE ===\n\n"
            "Task:\n" + instruction
        )

    def _build_command(self, prompt: str) -> list[str]:
        cmd = ["codex", "exec", "--json", "--full-auto"]
        if self.model:
            cmd += ["--model", self.model]
        if self.base_url:
            cmd += [
                "-c", 'model_provider="openai-responses"',
                "-c", 'model_providers.openai-responses.name="OpenAI Responses"',
                "-c", f'model_providers.openai-responses.base_url="{self.base_url}"',
                "-c", 'model_providers.openai-responses.env_key="OPENAI_API_KEY"',
                "-c", 'model_providers.openai-responses.wire_api="responses"',
            ]
        cmd += ["-C", self._work_dir]
        cmd += ["--skip-git-repo-check"]
        cmd += [prompt]
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = {**os.environ}
        if self.api_key:
            env["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            env["OPENAI_BASE_URL"] = self.base_url
        return env

    def _timeout(self) -> int:
        if self.max_iterations:
            return max(self.max_iterations * 60, 600)
        return 1800  # 30 min default

    def _parse_output(self, stdout: str) -> dict:
        """Parse codex --json JSONL output into a standard result dict."""
        conversation = []
        execution_trace = []
        usage = {}

        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            # item.completed — agent messages and command executions
            if etype == "item.completed":
                item = event.get("item", {})
                itype = item.get("type", "")
                if itype == "agent_message":
                    text = item.get("text", "")
                    if text.strip():
                        conversation.append({"role": "assistant", "content": text})
                elif itype == "command_execution":
                    execution_trace.append({
                        "action": {"action_type": "shell", "command": item.get("command", "")},
                        "feedback": {
                            "message": item.get("aggregated_output", ""),
                            "exit_code": item.get("exit_code"),
                        },
                    })
                continue

            # Old format: response_item with message payload
            if etype == "response_item":
                payload = event.get("payload", {})
                if payload.get("type") != "message":
                    continue
                role = payload.get("role", "")
                content = payload.get("content", [])
                if isinstance(content, list):
                    parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    text = "\n\n".join(p for p in parts if p.strip())
                else:
                    text = str(content) if content else ""
                if text.strip():
                    conversation.append({"role": role, "content": text})
                continue

            # Token usage from turn.completed
            if etype == "turn.completed":
                turn_usage = event.get("usage", {})
                if turn_usage:
                    usage["total_input_tokens"] = (
                        usage.get("total_input_tokens", 0) + turn_usage.get("input_tokens", 0)
                    )
                    usage["total_output_tokens"] = (
                        usage.get("total_output_tokens", 0) + turn_usage.get("output_tokens", 0)
                    )

        # Last assistant message is the task result
        task_result = ""
        for msg in reversed(conversation):
            if msg["role"] == "assistant" and msg["content"].strip():
                task_result = msg["content"]
                break

        result = {
            "task_result": task_result,
            "conversation": conversation,
            "execution_trace": execution_trace,
            "execution_summary": "",
        }
        if usage:
            result["api_cost_stats"] = usage
        return result

    def _collect_artifacts(self) -> dict[str, str]:
        """Collect files created in the workspace directory."""
        files = {}
        if not os.path.isdir(self._work_dir):
            return files

        extensions = {".html", ".py", ".js", ".ts", ".sh", ".json", ".txt", ".csv", ".md"}
        for root, _, filenames in os.walk(self._work_dir):
            for fname in filenames:
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath) as f:
                            content = f.read()
                        rel = os.path.relpath(fpath, self._work_dir)
                        files[rel] = content
                    except Exception:
                        pass
        return files
