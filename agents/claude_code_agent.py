"""
ClaudeCodeAgent — wraps the Claude Code CLI for use inside Harbor containers.

Implements the agent protocol expected by harness/run_task.py:
  - setup_environment(task)
  - run_task(task) -> dict
  - cleanup_environment()

Claude Code CLI is invoked via `claude --print --output-format stream-json`
and the JSONL events are parsed into the standard result dict format.
"""
import json
import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)


def _shell_quote(s: str) -> str:
    """Quote a string for safe use in a shell command."""
    import shlex
    return shlex.quote(s)


class ClaudeCodeAgent:
    agent_type = "claude_code"

    def __init__(self, config: dict):
        self.config = config
        self.model = (
            config.get("controller", {}).get("args", {}).get("model")
            or "sonnet"
        )
        self.api_key = (
            config.get("controller", {}).get("args", {}).get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        # ANTHROPIC_BASE_URL should NOT include /v1 — the SDK appends it.
        # If we get an OpenAI-style URL like https://proxy/v1, strip the /v1.
        raw_base = (
            config.get("controller", {}).get("args", {}).get("base_url")
            or os.environ.get("ANTHROPIC_BASE_URL", "")
            or os.environ.get("LLM_BASE_URL", "")
        )
        self.base_url = raw_base.rstrip("/").removesuffix("/v1") if raw_base else ""
        self.max_budget = config.get("max_budget_usd", 1.0)
        self._work_dir = "/tmp/claude-workspace"

    def setup_environment(self, task: dict) -> None:
        os.makedirs(self._work_dir, exist_ok=True)

    def run_task(self, task: dict) -> dict:
        instruction = task.get("instruction", "") or task.get("description", "")
        wrapped = self._wrap_instruction(instruction)

        cmd = self._build_command(wrapped)
        env = self._build_env()

        logger.info("Running claude: %s", " ".join(cmd))
        start = time.time()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self._work_dir,
                env=env,
                timeout=self._timeout(),
            )
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            return {
                "task_name": task.get("task_name", "unknown"),
                "status": "error",
                "error": f"claude timed out after {elapsed:.0f}s",
                "task_result": (e.stdout or "").strip() if e.stdout else "",
                "conversation": [],
                "execution_summary": "",
                "elapsed_seconds": elapsed,
            }

        elapsed = time.time() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            logger.warning("Claude exited %d: %s", proc.returncode, stderr[:500])

        result = self._parse_output(stdout)
        result["task_name"] = task.get("task_name", "unknown")
        result["elapsed_seconds"] = elapsed

        if proc.returncode == 0 or result.get("task_result") or result.get("conversation"):
            result["status"] = "completed"
        else:
            result["status"] = "error"
            result["error"] = stderr[:2000] or f"claude exited with code {proc.returncode}"

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
        base_args = [
            "--print",
            "--verbose",
            "--output-format", "stream-json",
            "--bare",
            "--dangerously-skip-permissions",
        ]
        if self.model:
            base_args += ["--model", self.model]
        if self.max_budget:
            base_args += ["--max-budget-usd", str(self.max_budget)]
        base_args += ["--", prompt]

        # In containers running as root, Claude blocks --dangerously-skip-permissions.
        # Run as non-root user via su if we're root.
        if os.getuid() == 0:
            self._ensure_non_root_user()
            escaped_args = " ".join(_shell_quote(a) for a in base_args)
            # Pass env vars explicitly since su doesn't inherit them
            env_exports = []
            env = self._build_env()
            # Override HOME to claude-user's home dir
            env["HOME"] = "/home/claude-user"
            for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_SIMPLE",
                         "HOME", "PATH", "NODE_PATH"):
                val = env.get(key) or os.environ.get(key, "")
                if val:
                    env_exports.append(f"export {key}={_shell_quote(val)}")
            env_str = " && ".join(env_exports) + " && " if env_exports else ""
            return ["su", "-s", "/bin/bash", "claude-user", "-c",
                    f"{env_str}claude {escaped_args}"]
        return ["claude"] + base_args

    def _ensure_non_root_user(self) -> None:
        """Create a non-root user for running Claude Code in containers."""
        import pwd
        try:
            pwd.getpwnam("claude-user")
        except KeyError:
            subprocess.run(
                ["useradd", "-m", "-s", "/bin/bash", "claude-user"],
                check=True, capture_output=True,
            )
        # Ensure the workspace is writable
        os.makedirs(self._work_dir, exist_ok=True)
        subprocess.run(["chown", "-R", "claude-user", self._work_dir], check=False)

    def _build_env(self) -> dict[str, str]:
        env = {**os.environ}
        if self.api_key:
            env["ANTHROPIC_API_KEY"] = self.api_key
        if self.base_url:
            env["ANTHROPIC_BASE_URL"] = self.base_url
        # Disable interactive features in container
        env["CLAUDE_CODE_SIMPLE"] = "1"
        # Suppress any MCP/hook noise
        env.pop("CLAUDE_MCP_CONFIG", None)
        return env

    def _timeout(self) -> int:
        return 1800  # 30 min default

    def _parse_output(self, stdout: str) -> dict:
        """Parse claude --output-format stream-json JSONL output."""
        conversation = []
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
            msg = event.get("message", {})
            role = msg.get("role", etype)

            if role not in ("assistant", "user"):
                # Check for result event (final output)
                if etype == "result":
                    text = event.get("result", "")
                    if text and text.strip():
                        conversation.append({"role": "assistant", "content": text})
                    result_usage = event.get("usage", {})
                    if result_usage:
                        usage["total_input_tokens"] = result_usage.get("input_tokens", 0)
                        usage["total_output_tokens"] = result_usage.get("output_tokens", 0)
                        usage["total_cost_usd"] = result_usage.get("cost_usd")
                continue

            content = msg.get("content", "")
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

        # Last assistant message is the task result
        task_result = ""
        for msg in reversed(conversation):
            if msg["role"] == "assistant" and msg["content"].strip():
                task_result = msg["content"]
                break

        result = {
            "task_result": task_result,
            "conversation": conversation,
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
