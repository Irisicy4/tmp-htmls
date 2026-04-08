"""
Harbor-inside-Modal runner: fire-and-forget multi-agent experiments.

Uses Harbor's agent ecosystem (claude-code, codex, gemini-cli, …) inside Modal
Sandboxes, without requiring Harbor's own Modal integration (-e modal).
The orchestrator runs as a deployed Modal Function so the laptop can disconnect.

Deploy once:
  modal deploy harbor_modal_runner.py

Trigger and disconnect:
  python trigger_harbor.py --agent claude-code --tasks-dir tasks/batch-1
  python trigger_harbor.py --agent claude-code --first-n 3

Collect results later (no laptop needed):
  modal run harbor_modal_runner.py --collect <run-tag>
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

# ── App & volume ───────────────────────────────────────────────────────────────
app = modal.App("evolve-bench-harbor")
results_volume = modal.Volume.from_name("evolve-bench-results", create_if_missing=True)

# ── Image ──────────────────────────────────────────────────────────────────────
# Includes Harbor (for agent management) plus evaluation deps.
# Task files are baked in so the container is self-contained.
harbor_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "harbor==0.1.45",
        "modal>=0.73",
        "anthropic>=0.75.0",
        "openai>=2.6.1",
        "pyyaml>=6.0.3",
        "tenacity>=8.0",
        "jinja2>=3.0",
        "pydantic>=2.0",
        "shortuuid>=1.0",
    )
    .add_local_dir("tasks", "/harbor-bench/tasks", copy=True)
    .add_local_file("harness/skill_store.py", "/harness/skill_store.py", copy=True)
    .add_local_file("harness/skill_extractor.py", "/harness/skill_extractor.py", copy=True)
)

# ── Constants ─────────────────────────────────────────────────────────────────
TASKS_ROOT = Path("/harbor-bench/tasks")
RESULTS_BASE = Path("/results")

# Base sandbox image: browser + Python, no agent.  Harbor installs the agent.
SANDBOX_BASE_IMAGE = "ghcr.io/agent-infra/sandbox:latest"

# Directory inside the sandbox where agents should save output files
AGENT_OUTPUT_DIR = "/output"

# Appended to every task instruction so agents know where to save files
OUTPUT_DIR_INSTRUCTION = f"""

---
IMPORTANT: If you create any output files (reports, data, code, spreadsheets,
images, HTML, etc.), save them to {AGENT_OUTPUT_DIR}/ directory. Create the
directory if it doesn't exist. Always save a copy of your final answer/result
as {AGENT_OUTPUT_DIR}/result.txt as well.
"""


# ── Task metadata ─────────────────────────────────────────────────────────────

def _load_task_metadata(task_dir: Path) -> dict:
    """Parse task.toml for category, tags, timeouts — attached to every result."""
    meta = {"category": "unknown", "tags": [], "timeout_sec": 900.0}
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return meta
    try:
        section = ""
        for line in toml_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("["):
                section = line.strip("[] ")
            elif "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if section == "metadata" and key == "category":
                    meta["category"] = val
                elif section == "metadata" and key == "tags":
                    meta["tags"] = [
                        t.strip().strip('"').strip("'")
                        for t in val.strip("[]").split(",")
                        if t.strip()
                    ]
                elif section == "agent" and key == "timeout_sec":
                    meta["timeout_sec"] = float(val)
    except Exception:
        pass
    return meta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tasks_base(tasks_dir: str) -> Path:
    """Resolve tasks_dir (e.g. 'tasks/updated-deprivacy-100') to container path."""
    return TASKS_ROOT / Path(tasks_dir).name


def _discover_tasks(tasks_dir: str, first_n: int, task_names_csv: str) -> list[dict]:
    """Return ordered task dicts (name + metadata) from the given tasks_dir."""
    base = _tasks_base(tasks_dir)
    if task_names_csv:
        names = [t.strip() for t in task_names_csv.split(",") if t.strip()]
    else:
        names = sorted(
            d.name
            for d in base.iterdir()
            if d.is_dir() and (d / "instruction.md").exists()
        )
        if first_n > 0:
            names = names[:first_n]

    tasks = []
    for name in names:
        meta = _load_task_metadata(base / name)
        tasks.append({"task_name": name, **meta})
    return tasks


async def _read_stdout(exec_result) -> str:
    """Read stdout from a sandbox exec result, handling both str and bytes."""
    raw = await exec_result.stdout.read.aio()
    if isinstance(raw, bytes):
        return raw.decode(errors="replace").strip()
    return raw.strip()


def _parse_claude_code_output(stdout: str) -> dict:
    """
    Convert claude-code stream-json stdout into our result dict format:
      {"task_result": <last assistant text>, "conversation": [...], "execution_summary": ""}
    test_task.py's _extract_response() checks task_result first, then conversation.
    """
    conversation = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        msg = event.get("message", {})
        role = msg.get("role", etype)
        if role not in ("assistant", "user"):
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

    task_result = ""
    for msg in reversed(conversation):
        if msg["role"] == "assistant" and msg["content"].strip():
            task_result = msg["content"]
            break

    return {
        "task_result": task_result,
        "conversation": conversation,
        "execution_summary": "",
    }


def _parse_codex_output(stdout: str) -> dict:
    """
    Parse codex --json JSONL output.
    Supports two event formats emitted by different codex CLI versions:
      - New:  {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
      - Old:  {"type": "response_item", "payload": {"type": "message", "role": "assistant", ...}}
    """
    conversation = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        # New format: item.completed with agent_message
        if etype == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text.strip():
                    conversation.append({"role": "assistant", "content": text})
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

    task_result = ""
    for msg in reversed(conversation):
        if msg["role"] == "assistant" and msg["content"].strip():
            task_result = msg["content"]
            break

    return {
        "task_result": task_result,
        "conversation": conversation,
        "execution_summary": "",
    }


def _wrap_codex_instruction(instruction: str) -> str:
    """
    Prepend a directive so codex doesn't ask clarifying questions,
    and append a request to print created file contents to stdout.
    """
    return (
        "Complete the following task immediately without asking any clarifying questions. "
        "Make reasonable assumptions and proceed. "
        "After creating any files, print each file's complete contents to stdout "
        "wrapped like this: === FILE: <filename> ===\n<contents>\n=== END FILE ===\n\n"
        "Task:\n" + instruction
    )


def _extract_files_from_codex_response(task_result: str) -> dict[str, str]:
    """Extract file contents embedded in codex text output."""
    import re
    files = {}
    pattern = r"=== FILE: (.+?) ===\n(.*?)\n=== END FILE ==="
    for m in re.finditer(pattern, task_result, re.DOTALL):
        files[m.group(1).strip()] = m.group(2)
    return files


def _parse_generic_output(stdout: str) -> dict:
    """
    Fallback parser for agents that write plain text.
    Passes the full stdout as task_result — the LLM judge handles it.
    """
    return {
        "task_result": stdout.strip(),
        "conversation": [],
        "execution_summary": "",
    }


AGENT_OUTPUT_PARSERS = {
    "claude-code": _parse_claude_code_output,
    "codex": _parse_codex_output,
}


# ── Proxy config injection ───────────────────────────────────────────────────

async def _inject_proxy_config(agent_name: str, extra_env: dict, harbor_env):
    """Write proxy config files into the sandbox AFTER agent install, BEFORE run.

    Codex CLI: writes config.toml with a custom model_provider pointing to the proxy.
    Claude Code: wraps the claude binary so the correct env vars are always set,
                 because Harbor merges AUTH_TOKEN into API_KEY which breaks proxies.
    """
    openai_base = extra_env.get("OPENAI_BASE_URL", "")
    anthropic_base = extra_env.get("ANTHROPIC_BASE_URL", "")
    anthropic_key = extra_env.get("ANTHROPIC_API_KEY", "")
    model_name = extra_env.get("_CLAUDE_MODEL", "")

    if agent_name == "codex" and openai_base:
        from harbor.models.trial.paths import EnvironmentPaths
        codex_home = EnvironmentPaths.agent_dir.as_posix()

        config_toml = (
            f'model_provider = "uniapi"\n'
            f'model_reasoning_effort = "high"\n'
            f'\n'
            f'[model_providers.uniapi]\n'
            f'name = "uniapi"\n'
            f'base_url = "{openai_base}"\n'
            f'env_key = "OPENAI_API_KEY"\n'
        )
        write_cmd = f'cat > "{codex_home}/config.toml" << \'TOMLEOF\'\n{config_toml}TOMLEOF'

        try:
            result = await harbor_env._sandbox.exec.aio("bash", "-c", write_cmd)
            await result.wait.aio()
            print(f"  [proxy] Wrote codex config.toml -> {openai_base}")
        except Exception as e:
            print(f"  [proxy] Failed to write codex config.toml: {e}")

    elif agent_name == "claude-code" and anthropic_base and anthropic_key:
        model = model_name or "claude-sonnet-4-20250514"

        wrapper_script = f"""#!/bin/bash
# Proxy wrapper — overrides Harbor's env vars for Claude Code CLI
export ANTHROPIC_API_KEY="{anthropic_key}"
export ANTHROPIC_BASE_URL="{anthropic_base}"
export ANTHROPIC_MODEL="{model}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="{model}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="{model}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="{model}"
export CLAUDE_CODE_SUBAGENT_MODEL="{model}"
exec /root/.local/bin/claude.real "$@"
"""
        rename_and_wrap = (
            "if [ -f /root/.local/bin/claude ] && [ ! -f /root/.local/bin/claude.real ]; then "
            "  mv /root/.local/bin/claude /root/.local/bin/claude.real && "
            "  cat > /root/.local/bin/claude << 'WRAPEOF'\n"
            f"{wrapper_script}"
            "WRAPEOF\n"
            "  chmod +x /root/.local/bin/claude && "
            "  echo '[proxy] Claude wrapper installed'; "
            "fi"
        )

        try:
            result = await harbor_env._sandbox.exec.aio("bash", "-c", rename_and_wrap)
            await result.wait.aio()
            stdout = await _read_stdout(result)
            print(f"  [proxy] Claude Code wrapper -> {anthropic_base} (model: {model})")
            if stdout:
                print(f"  [proxy] {stdout}")
        except Exception as e:
            print(f"  [proxy] Failed to install Claude Code wrapper: {e}")


def _evaluate(task_name: str, result: dict, tasks_dir: str) -> dict:
    """
    Load test_task.py for the given task and call its test() function.
    Returns the full eval_result dict.
    """
    test_script = _tasks_base(tasks_dir) / task_name / "tests" / "test_task.py"
    if not test_script.exists():
        print(f"[{task_name}] No test_task.py — skipping evaluation")
        return {"passed": False, "overall_score": 0.0, "feedback": "no evaluator"}

    spec = importlib.util.spec_from_file_location("test_task", test_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        eval_result = mod.test(result)
    except Exception as exc:
        print(f"[{task_name}] Evaluator error: {exc}")
        return {"passed": False, "overall_score": 0.0, "feedback": str(exc)}

    return eval_result


# ── Enriched result builder ──────────────────────────────────────────────────

def _build_result(
    task_name: str,
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    task_meta: dict,
    started_at: str,
    score: float,
    passed: bool,
    eval_result: dict | None = None,
    error: str = "",
    artifacts: list[str] | None = None,
    extracted_skill: str | None = None,
) -> dict:
    """Build the enriched result dict with all metadata for cross-agent analysis."""
    finished_at = datetime.now(timezone.utc).isoformat()
    eval_result = eval_result or {}
    details = eval_result.get("details", {}) or {}
    dimension_scores = details.get("dimension_scores", {}) or {}

    model_name = agent_kwargs.get("model_name", "")
    if not model_name:
        model_name = (agent_kwargs.get("extra_env", {}) or {}).get("LLM_MODEL", "default")

    result = {
        # --- Identity ---
        "task": task_name,
        "agent": agent_name,
        "model": model_name,
        "run_tag": run_tag,

        # --- Task metadata ---
        "category": task_meta.get("category", "unknown"),
        "tags": task_meta.get("tags", []),

        # --- Scores ---
        "score": score,
        "passed": passed,
        "dimension_scores": dimension_scores,

        # --- Evaluation detail ---
        "feedback": eval_result.get("feedback", error or ""),
        "evidence_summary": details.get("evidence_summary", ""),
        "dimension_reasoning": details.get("dimension_reasoning", {}),

        # --- Timing ---
        "started_at": started_at,
        "finished_at": finished_at,

        # --- Artifacts ---
        "artifacts": artifacts or [],

        # --- Error ---
        "error": error,
    }
    if extracted_skill:
        result["extracted_skill"] = extracted_skill
    return result


# ── PatchedModalEnvironment ────────────────────────────────────────────────────

def _make_harbor_env(session_id: str, trial_paths):
    """
    Returns a ModalEnvironment subclass instance that:
    1. Uses ghcr.io/agent-infra/sandbox:latest instead of a per-task Dockerfile.
    2. Uses session_id as the Modal app name to avoid the "__harbor__" locking bug.
    3. Cleans up the ephemeral app on stop to avoid hitting the 200 app limit.
    """
    import modal as _modal
    from harbor.environments.modal import ModalEnvironment
    from harbor.models.task.config import EnvironmentConfig as TaskEnvConfig
    from harbor.models.trial.paths import EnvironmentPaths

    class _PatchedEnv(ModalEnvironment):
        def _validate_definition(self):
            pass  # skip Dockerfile check — we override start()

        async def start(self, force_build: bool) -> None:
            self._image = _modal.Image.from_registry(SANDBOX_BASE_IMAGE)
            # Each task uses a unique Modal app name → avoids __harbor__ locking
            self._app = await _modal.App.lookup.aio(
                name=self.session_id,
                create_if_missing=True,
            )
            self._sandbox = await self._create_sandbox(
                gpu_config=None,
                secrets_config=[],
                volumes_config={},
            )
            await self._sandbox.mkdir.aio(
                str(EnvironmentPaths.agent_dir), parents=True
            )
            await self._sandbox.mkdir.aio(
                str(EnvironmentPaths.verifier_dir), parents=True
            )

        async def stop(self, delete: bool = True) -> None:
            """Stop sandbox and delete the ephemeral app to avoid hitting the 200 app limit."""
            try:
                await super().stop(delete=delete)
            except Exception as e:
                print(f"[{self.session_id}] sandbox stop error (non-fatal): {e}")
            # App.stop() doesn't exist in the SDK — use the gRPC API directly
            try:
                from modal.client import _Client
                from modal_proto import api_pb2
                client = await _Client.from_env.aio()
                req = api_pb2.AppGetByDeploymentNameRequest(
                    name=self.session_id, environment_name=""
                )
                resp = await client.stub.AppGetByDeploymentName(req)
                if resp.app_id:
                    stop_req = api_pb2.AppStopRequest(
                        app_id=resp.app_id,
                        source=api_pb2.APP_STOP_SOURCE_CLI,
                    )
                    await client.stub.AppStop(stop_req)
                    print(f"[{self.session_id}] ephemeral app stopped")
            except Exception as e:
                print(f"[{self.session_id}] app cleanup failed: {e}")

    return _PatchedEnv(
        environment_dir=Path("/tmp"),   # unused — start() is overridden
        environment_name=session_id,
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=TaskEnvConfig(cpus=1.0, memory_mb=1024),
    )


# ── Core task runner (runs inside Modal container) ────────────────────────────

async def _run_task(
    task_name: str,
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    task_meta: dict,
    tasks_dir: str = "tasks/updated-deprivacy-100",
    skill_config: dict | None = None,
) -> dict:
    """
    Run one task end-to-end:
    1. Create a Harbor-managed Modal Sandbox with the base sandbox image.
    2. Harbor installs the agent (claude-code, codex, …) in the sandbox.
    3. Agent runs; Harbor captures stdout to local (volume-mounted) logs.
    4. Parse stdout, evaluate with test_task.py, return enriched result dict.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.context import AgentContext
    from harbor.models.agent.name import AgentName
    from harbor.models.trial.paths import TrialPaths

    skill_config = skill_config or {}
    started_at = datetime.now(timezone.utc).isoformat()

    task_dir = _tasks_base(tasks_dir) / task_name
    instruction = (task_dir / "instruction.md").read_text().strip()

    # Tell agents to save output files to /output/
    instruction += OUTPUT_DIR_INSTRUCTION

    # Phase 2: inject relevant skills into instruction
    if skill_config.get("use_skills"):
        sys.path.insert(0, "/harness")
        from skill_store import SkillStore
        store = SkillStore(skills_dir=skill_config.get("skills_dir", "/skills"))
        skill_bodies = store.search_skills(instruction, top_k=3)
        if skill_bodies:
            combined = "\n\n---\n\n".join(skill_bodies)
            instruction = (
                "Here are some relevant strategies from similar past tasks:\n\n"
                f"{combined}\n\n---\n\nNow, here is your actual task:\n\n"
                + instruction
            )
            print(f"[{task_name}] Injected {len(skill_bodies)} skill(s)")

    # Agent-specific instruction wrapping
    if agent_name == "codex":
        instruction = _wrap_codex_instruction(instruction)
    elif agent_name == "claude-code":
        instruction = (
            "IMPORTANT: Do NOT use plan mode. Implement the solution directly and immediately. "
            "Do not ask clarifying questions. Make reasonable assumptions and proceed. "
            "After creating any files, print each file's complete contents to stdout "
            "wrapped like this: === FILE: <filename> ===\n<contents>\n=== END FILE ===\n\n"
            "Task:\n" + instruction
        )
    elif agent_name == "aider":
        instruction = (
            "Complete the following task immediately without asking any clarifying questions. "
            "Make reasonable assumptions and proceed. "
            "After creating any files, print each file's complete contents to stdout "
            "wrapped like this: === FILE: <filename> ===\n<contents>\n=== END FILE ===\n\n"
            "Task:\n" + instruction
        )

    trial_dir = RESULTS_BASE / run_tag / "trials" / task_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    # Modal app names: alphanumeric + dashes + dots + underscores only, max 64 chars
    safe_tag = run_tag.replace("/", "-").replace(" ", "-")
    session_id = f"hm-{safe_tag}-{task_name}"[:63].rstrip("-")

    env = _make_harbor_env(session_id, trial_paths)

    # Expose agent API keys so Harbor's create_run_agent_commands() picks them up.
    for k, v in agent_kwargs.get("extra_env", {}).items():
        os.environ[k] = v  # force override

    # Judge credentials: set by trigger_harbor.py, separate from agent credentials.
    for k, v in agent_kwargs.get("judge_env", {}).items():
        os.environ[k] = v

    # Filter out judge_env before passing to Harbor — it doesn't know about it
    factory_kwargs = {k: v for k, v in agent_kwargs.items() if k != "judge_env"}
    agent = AgentFactory.create_agent_from_name(
        AgentName(agent_name),
        logs_dir=trial_paths.agent_dir,
        **factory_kwargs,
    )

    context = AgentContext()

    print(f"[{task_name}] Starting sandbox …")
    try:
        await env.start(force_build=False)
    except Exception as exc:
        print(f"[{task_name}] Sandbox start failed: {exc}")
        return _build_result(
            task_name, agent_name, agent_kwargs, run_tag, task_meta,
            started_at, score=0.0, passed=False, error=str(exc),
        )

    collected_artifacts: list[str] = []
    try:
        print(f"[{task_name}] Setting up agent ({agent_name}) …")
        await agent.setup(env)

        # Inject proxy config after agent install, before run
        extra = agent_kwargs.get("extra_env", {})
        await _inject_proxy_config(agent_name, extra, env)

        print(f"[{task_name}] Running agent …")
        await agent.run(instruction, env, context)
    except Exception as exc:
        print(f"[{task_name}] Agent error: {exc}")
    finally:
        # Collect artifacts from /output/
        artifacts_dir = RESULTS_BASE / run_tag / "artifacts" / task_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            result_obj = await env._sandbox.exec.aio(
                "bash", "-c",
                f"find {AGENT_OUTPUT_DIR} -type f 2>/dev/null | head -50"
            )
            await result_obj.wait.aio()
            output_files = await _read_stdout(result_obj)

            all_files = set()
            if output_files:
                all_files.update(output_files.splitlines())

            for fpath in sorted(all_files):
                fpath = fpath.strip()
                if not fpath:
                    continue
                try:
                    cat_obj = await env._sandbox.exec.aio("cat", fpath)
                    await cat_obj.wait.aio()
                    content = await _read_stdout(cat_obj)
                    fname = fpath.split("/")[-1]

                    (artifacts_dir / fname).write_text(content)
                    (trial_paths.agent_dir / f"artifact_{fname}").write_text(content)
                    collected_artifacts.append(fname)
                    print(f"[{task_name}] Collected: {fpath} ({len(content)} chars)")
                except Exception:
                    pass

            if all_files:
                await results_volume.commit.aio()
        except Exception as collect_exc:
            print(f"[{task_name}] Artifact collection skipped: {collect_exc}")

        try:
            await env.stop(delete=True)
        except Exception:
            pass

    # ── Evaluate ──────────────────────────────────────────────────────────────
    agent_files = list(trial_paths.agent_dir.rglob("*")) if trial_paths.agent_dir.exists() else []
    print(f"[{task_name}] Agent dir files: {[str(f.relative_to(trial_paths.agent_dir)) for f in agent_files if f.is_file()]}")

    agent_stdout = ""
    for cmd_idx in (1, 0):
        stdout_file = trial_paths.agent_dir / f"command-{cmd_idx}" / "stdout.txt"
        if stdout_file.exists():
            agent_stdout = stdout_file.read_text()
            if agent_stdout.strip():
                print(f"[{task_name}] Using command-{cmd_idx} stdout ({len(agent_stdout)} chars)")
                break

    parser = AGENT_OUTPUT_PARSERS.get(agent_name, _parse_generic_output)
    result_dict = parser(agent_stdout)

    # Append collected artifact contents to task_result so the judge can evaluate them
    artifact_texts = []
    for artifact_file in sorted(trial_paths.agent_dir.glob("artifact_*")):
        content = artifact_file.read_text()
        fname = artifact_file.name[len("artifact_"):]
        artifact_texts.append(f"\n\n=== FILE: {fname} ===\n{content}\n=== END FILE ===")
    if artifact_texts:
        result_dict["task_result"] = (result_dict.get("task_result") or "") + "".join(artifact_texts)

    # Save the raw result dict for debugging
    (trial_dir / "agent_result.json").write_text(
        json.dumps(result_dict, indent=2, default=str)
    )

    eval_result = _evaluate(task_name, result_dict, tasks_dir)

    score = float(eval_result.get("details", {}).get("overall_score", 0)
                  or eval_result.get("overall_score", 0))
    passed = bool(eval_result.get("passed", False))

    # Phase 1: extract and store skill if result is good enough
    extracted_skill = None
    if skill_config.get("store_skills"):
        sys.path.insert(0, "/harness")
        from skill_extractor import SkillExtractor
        from skill_store import SkillStore

        task_data = {"task_name": task_name, "instruction": instruction}
        result_for_skill = {**result_dict, "instruction": instruction}
        config_for_skill = {
            "store_skills": True,
            "skill_score_threshold": skill_config.get("skill_score_threshold", 3.0),
            "skill_model": skill_config.get("skill_model", "gpt-4o-mini"),
            "skills_dir": skill_config.get("skills_dir", f"/results/{run_tag}/skills"),
        }

        extractor = SkillExtractor(model=config_for_skill["skill_model"])
        decision = extractor.should_store(result_for_skill, eval_result, config_for_skill)
        if decision.get("store"):
            print(f"[{task_name}] Extracting skill: {decision.get('reason', '')}")
            skill_md = extractor.extract_skill(task_data, result_for_skill, eval_result)
            store = SkillStore(skills_dir=config_for_skill["skills_dir"])
            metadata = {
                "score": score,
                "task_type": decision.get("task_type", ""),
                "tags": decision.get("tags", []),
            }
            path = store.save_skill(skill_md, task_name, metadata)
            extracted_skill = skill_md
            print(f"[{task_name}] Saved skill: {path}")
        else:
            print(f"[{task_name}] Not storing skill: {decision.get('reason', '')}")

    result = _build_result(
        task_name, agent_name, agent_kwargs, run_tag, task_meta,
        started_at, score=score, passed=passed,
        eval_result=eval_result,
        artifacts=collected_artifacts,
        extracted_skill=extracted_skill,
    )

    (trial_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{task_name}] Done — score={score:.2f} passed={passed}")
    return result


# ── Modal Functions ───────────────────────────────────────────────────────────

@app.function(
    image=harbor_image,
    timeout=3600,
    volumes={str(RESULTS_BASE): results_volume},
    max_containers=20,
)
async def run_harbor_task(
    task_name: str,
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    task_meta: dict,
    tasks_dir: str = "tasks/updated-deprivacy-100",
    skill_config: dict | None = None,
) -> dict:
    """Run a single task with a Harbor agent. Used by run_harbor_experiment via starmap."""
    return await _run_task(task_name, agent_name, agent_kwargs, run_tag, task_meta, tasks_dir, skill_config)


@app.function(
    image=harbor_image,
    timeout=14400,
    volumes={str(RESULTS_BASE): results_volume},
)
async def run_harbor_experiment(
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    tasks_dir: str = "tasks/updated-deprivacy-100",
    first_n: int = 0,
    task_names_csv: str = "",
    skill_config: dict | None = None,
) -> dict:
    """
    Orchestrate a full experiment run.
    Dispatches one run_harbor_task per task (parallel via starmap).
    """
    tasks = _discover_tasks(tasks_dir, first_n, task_names_csv)
    model_name = agent_kwargs.get("model_name", "")

    print(f"\n=== Harbor Experiment: {run_tag} ===")
    print(f"Agent : {agent_name}  Model: {model_name}")
    print(f"Tasks dir: {tasks_dir}")
    print(f"Tasks : {len(tasks)}")
    if skill_config:
        print(f"Skills: {skill_config}")

    results = []
    async for result in run_harbor_task.starmap.aio(
        [(t["task_name"], agent_name, agent_kwargs, run_tag, t, tasks_dir, skill_config) for t in tasks]
    ):
        results.append(result)
        n_done = len(results)
        scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
        print(
            f"  [{n_done}/{len(tasks)}] {result['task']:50s}  "
            f"score={result.get('score', '?'):.2f}"
        )
        if scores:
            print(f"   running avg = {sum(scores) / len(scores):.3f}")

    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    errors = [r for r in results if r.get("error")]
    skills = {}
    for r in results:
        if r.get("extracted_skill"):
            skills[r["task"]] = r["extracted_skill"]

    # Per-category breakdown
    cat_scores: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        cat_scores.setdefault(cat, []).append(r.get("score", 0.0))
    category_summary = {
        cat: {
            "count": len(ss),
            "mean_score": round(sum(ss) / len(ss), 3) if ss else 0,
            "pass_rate": round(100 * sum(1 for s in ss if s >= 3.0) / len(ss), 1) if ss else 0,
        }
        for cat, ss in sorted(cat_scores.items())
    }

    # Per-dimension averages
    dim_totals: dict[str, list[float]] = {}
    for r in results:
        for dim, val in (r.get("dimension_scores") or {}).items():
            if val is not None:
                dim_totals.setdefault(dim, []).append(float(val))
    dimension_averages = {
        dim: round(sum(vs) / len(vs), 3) for dim, vs in sorted(dim_totals.items()) if vs
    }

    summary = {
        "run_tag": run_tag,
        "agent": agent_name,
        "model": model_name,
        "tasks_dir": tasks_dir,
        "n_tasks": len(results),
        "n_passed": sum(1 for r in results if r.get("passed")),
        "n_errors": len(errors),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "pass_rate_pct": round(100 * sum(1 for r in results if r.get("passed")) / len(results), 1) if results else 0,
        "skills_extracted": len(skills),
        "category_summary": category_summary,
        "dimension_averages": dimension_averages,
        "results": results,
    }

    out_path = RESULTS_BASE / run_tag / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    await results_volume.commit.aio()

    print(f"\n=== Experiment Complete ===")
    print(f"Mean score : {summary['mean_score']:.3f}")
    print(f"Pass rate  : {summary['n_passed']}/{summary['n_tasks']} ({summary['pass_rate_pct']}%)")
    print(f"Errors     : {summary['n_errors']}")
    if category_summary:
        print(f"\nCategory breakdown:")
        for cat, info in category_summary.items():
            print(f"  {cat:30s}  n={info['count']:3d}  avg={info['mean_score']:.2f}  pass={info['pass_rate']}%")
    if dimension_averages:
        print(f"\nDimension averages:")
        for dim, avg in dimension_averages.items():
            print(f"  {dim:30s}  {avg:.2f}")
    if skills:
        print(f"\nSkills     : {len(skills)} extracted")

    return {"summary": summary, "skills": skills}


@app.function(
    image=harbor_image,
    timeout=28800,  # 8 hours for full 2-phase experiment
    volumes={str(RESULTS_BASE): results_volume},
)
async def run_harbor_skill_experiment(
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    tasks_dir: str = "tasks/updated-deprivacy-100",
    first_n: int = 0,
    task_names_csv: str = "",
) -> dict:
    """Full skill experiment: Phase 1 → extract skills → Phase 2 → compare."""

    skills_dir = f"/results/{run_tag}/skills"

    # ── Phase 1: run tasks, extract skills ──
    phase1_tag = f"{run_tag}/phase1"
    phase1_skill_config = {
        "store_skills": True,
        "skills_dir": skills_dir,
        "skill_score_threshold": 3.0,
        "skill_model": "gpt-4o-mini",
    }
    phase1_out = await run_harbor_experiment.remote.aio(
        agent_name, agent_kwargs, phase1_tag,
        tasks_dir=tasks_dir, first_n=first_n, task_names_csv=task_names_csv,
        skill_config=phase1_skill_config,
    )

    skills = phase1_out.get("skills", {})
    print(f"\n>>> Phase 1 complete. Extracted {len(skills)} skill(s).")

    # Ensure skills are persisted to Volume
    if skills:
        await results_volume.commit.aio()

    # ── Phase 2: run tasks with skill injection ──
    phase2_tag = f"{run_tag}/phase2"
    phase2_skill_config = {
        "use_skills": True,
        "skills_dir": skills_dir,
    }
    phase2_out = await run_harbor_experiment.remote.aio(
        agent_name, agent_kwargs, phase2_tag,
        tasks_dir=tasks_dir, first_n=first_n, task_names_csv=task_names_csv,
        skill_config=phase2_skill_config,
    )

    # ── Compare ──
    p1 = phase1_out["summary"]
    p2 = phase2_out["summary"]

    p1_by_task = {r["task"]: r for r in p1["results"]}
    p2_by_task = {r["task"]: r for r in p2["results"]}
    all_tasks = sorted(set(p1_by_task) | set(p2_by_task))

    improved, declined, unchanged = 0, 0, 0
    for name in all_tasks:
        s1 = p1_by_task.get(name, {}).get("score", 0)
        s2 = p2_by_task.get(name, {}).get("score", 0)
        delta = float(s2) - float(s1)
        if delta > 0:
            improved += 1
        elif delta < 0:
            declined += 1
        else:
            unchanged += 1

    print("\n" + "=" * 60)
    print("  RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Phase 1':>10} {'Phase 2':>10} {'Delta':>10}")
    print("-" * 52)
    for key in ["mean_score", "n_passed"]:
        v1, v2 = p1[key], p2[key]
        delta = v2 - v1
        sign = "+" if delta > 0 else ""
        if isinstance(v1, float):
            print(f"  {key:<18} {v1:>10.2f} {v2:>10.2f} {sign}{delta:>9.2f}")
        else:
            print(f"  {key:<18} {v1:>10d} {v2:>10d} {sign}{delta:>9d}")
    print(f"\n  Improved: {improved}  |  Declined: {declined}  |  Unchanged: {unchanged}")
    print(f"  Skills extracted: {len(skills)}")

    comparison = {
        "run_tag": run_tag,
        "agent": agent_name,
        "phase1": p1,
        "phase2": p2,
        "skills_extracted": len(skills),
        "improved": improved,
        "declined": declined,
        "unchanged": unchanged,
    }
    comp_path = RESULTS_BASE / run_tag / "_comparison.json"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    comp_path.write_text(json.dumps(comparison, indent=2))
    await results_volume.commit.aio()

    return comparison


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(collect: str = "", list_runs: bool = False):
    """
    modal run harbor_modal_runner.py --collect <run-tag>
    modal run harbor_modal_runner.py --list-runs
    """
    import shutil
    modal_bin = shutil.which("modal") or os.path.expanduser("~/.local/bin/modal")

    if list_runs:
        subprocess.run([modal_bin, "volume", "ls", "evolve-bench-results"])
        return

    if collect:
        out_dir = Path(f"results/harbor/{collect}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {collect} → {out_dir}")
        subprocess.run(
            [modal_bin, "volume", "get", "evolve-bench-results", collect, str(out_dir), "--force"],
            check=True,
        )
        print(f"Done. Results in {out_dir}")
        return

    print("Use trigger_harbor.py to dispatch experiments.")
    print("  python trigger_harbor.py --agent claude-code --tasks-dir tasks/batch-1")
