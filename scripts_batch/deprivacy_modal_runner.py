"""
Modal runner for updated-deprivacy-100 tasks across ALL agent combinations.

This is a modified version of harbor_modal_runner.py that:
  1. Bakes in tasks/updated-deprivacy-100 instead of tasks/batch-1
  2. Adds structured metadata to every result (agent, model, category, timestamps)
  3. Produces a flat cross-agent summary for easy comparison

Deploy:
  modal deploy scripts_batch/deprivacy_modal_runner.py

Trigger (fire-and-forget):
  python scripts_batch/trigger_deprivacy.py --agent claude-code
  python scripts_batch/trigger_deprivacy.py --agent codex
  python scripts_batch/trigger_deprivacy.py --agent gemini-cli
  python scripts_batch/trigger_deprivacy.py --agent aider

Collect results:
  modal run scripts_batch/deprivacy_modal_runner.py --collect <run-tag>

Cross-agent analysis (after all agents finish):
  python scripts_batch/analyze_cross_agent.py results/deprivacy/
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

# ---------------------------------------------------------------------------
# App & volume
# ---------------------------------------------------------------------------
app = modal.App("deprivacy-100-bench")
results_volume = modal.Volume.from_name("deprivacy-100-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image — bakes in updated-deprivacy-100 tasks
# ---------------------------------------------------------------------------
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
    .add_local_dir(
        "tasks",
        "/harbor-bench/tasks",
        copy=True,
    )
    # Harness modules for skill extraction/injection (agent-agnostic)
    .add_local_file("harness/skill_extractor.py", "/harness/skill_extractor.py", copy=True)
    .add_local_file("harness/skill_store.py", "/harness/skill_store.py", copy=True)
    # History module for trace injection
    .add_local_file("scripts_batch/history.py", "/harness/history.py", copy=True)
    # Pre-harvested skills for evolve experiments
    .add_local_dir("evolve-skills", "/evolve-skills", copy=True, ignore=["__pycache__"])
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASKS_ROOT = Path("/harbor-bench/tasks")
DEFAULT_TASKS_DIR = "updated-deprivacy-100"
RESULTS_BASE = Path("/results")
SANDBOX_BASE_IMAGE = "ghcr.io/agent-infra/sandbox:latest"

# Seed tasks used as skill examples — tagged in results for analysis
_SEED_TASKS: set[str] = set()
_seed_manifest = Path("/evolve-skills/_seed_manifest.json")
if _seed_manifest.exists():
    import json as _json
    _SEED_TASKS = {s["task"] for s in _json.loads(_seed_manifest.read_text())}

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


# ---------------------------------------------------------------------------
# Task metadata loader
# ---------------------------------------------------------------------------

def _load_task_metadata(task_dir: Path) -> dict:
    """Parse task.toml for category, tags, timeouts — attached to every result."""
    meta = {"category": "unknown", "tags": [], "timeout_sec": 900.0}
    toml_path = task_dir / "task.toml"
    if not toml_path.exists():
        return meta
    try:
        # Minimal TOML parser (avoids extra dep) — good enough for flat keys
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
                    # Parse ["a", "b", "c"] style
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


def _get_tasks_base(tasks_dir: str = "") -> Path:
    """Resolve tasks directory from name or default."""
    name = tasks_dir or DEFAULT_TASKS_DIR
    return TASKS_ROOT / name


def _discover_tasks(first_n: int = 0, task_names_csv: str = "", tasks_dir: str = "") -> list[dict]:
    """Discover tasks and attach metadata."""
    base = _get_tasks_base(tasks_dir)
    if task_names_csv:
        names = [t.strip() for t in task_names_csv.split(",") if t.strip()]
    else:
        names = sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "instruction.md").exists()
        )
        if first_n > 0:
            names = names[:first_n]

    tasks = []
    for name in names:
        task_dir = base / name
        meta = _load_task_metadata(task_dir)
        tasks.append({"task_name": name, **meta})
    return tasks


# ---------------------------------------------------------------------------
# Output parsers (unchanged from harbor_modal_runner.py)
# ---------------------------------------------------------------------------

def _parse_claude_code_output(stdout: str) -> dict:
    conversation = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = event.get("message", {})
        role = msg.get("role", event.get("type"))
        if role not in ("assistant", "user"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
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
    return {"task_result": task_result, "conversation": conversation, "execution_summary": ""}


def _parse_codex_output(stdout: str) -> dict:
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
        if etype == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text.strip():
                    conversation.append({"role": "assistant", "content": text})
        elif etype == "response_item":
            payload = event.get("payload", {})
            if payload.get("type") != "message":
                continue
            role = payload.get("role", "")
            content = payload.get("content", [])
            if isinstance(content, list):
                parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
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
    return {"task_result": task_result, "conversation": conversation, "execution_summary": ""}


def _parse_generic_output(stdout: str) -> dict:
    return {"task_result": stdout.strip(), "conversation": [], "execution_summary": ""}


def _wrap_codex_instruction(instruction: str) -> str:
    return (
        "Complete the following task immediately without asking any clarifying questions. "
        "Make reasonable assumptions and proceed. "
        "For any visual, interactive, or game output, create a single self-contained HTML file "
        "(no external dependencies) rather than a terminal or desktop application. "
        "After creating any files, print each file's complete contents to stdout "
        "wrapped like this: === FILE: <filename> ===\n<contents>\n=== END FILE ===\n\n"
        "Task:\n" + instruction
    )


AGENT_OUTPUT_PARSERS = {
    "claude-code": _parse_claude_code_output,
    "codex": _parse_codex_output,
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _evaluate(task_name: str, result: dict, tasks_dir: str = "") -> dict:
    test_script = _get_tasks_base(tasks_dir) / task_name / "tests" / "test_task.py"
    if not test_script.exists():
        return {"passed": False, "overall_score": 0.0, "feedback": "no evaluator"}
    spec = importlib.util.spec_from_file_location("test_task", test_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        return mod.test(result)
    except Exception as exc:
        return {"passed": False, "overall_score": 0.0, "feedback": str(exc)}


# ---------------------------------------------------------------------------
# Patched Modal environment (from harbor_modal_runner.py)
# ---------------------------------------------------------------------------

def _make_harbor_env(session_id: str, trial_paths):
    """Create a lightweight sandbox environment that does NOT deploy persistent apps.

    Previous approach used App.lookup(create_if_missing=True) which created one
    deployed app per task, hitting Modal's 200-app limit. This version uses
    Sandbox.create() directly — no persistent app, auto-cleans on terminate.
    """
    import modal as _modal
    from harbor.environments.modal import ModalEnvironment
    from harbor.models.task.config import EnvironmentConfig as TaskEnvConfig
    from harbor.models.trial.paths import EnvironmentPaths

    class _PatchedEnv(ModalEnvironment):
        def _validate_definition(self):
            pass

        async def start(self, force_build: bool) -> None:
            self._image = _modal.Image.from_registry(SANDBOX_BASE_IMAGE)
            # Use Sandbox.create directly — no App.lookup, no persistent deployed app
            self._sandbox = await _modal.Sandbox.create.aio(
                image=self._image,
                cpu=1.0,
                memory=1024,
                timeout=900,
            )
            await self._sandbox.mkdir.aio(str(EnvironmentPaths.agent_dir), parents=True)
            await self._sandbox.mkdir.aio(str(EnvironmentPaths.verifier_dir), parents=True)

        async def stop(self, delete: bool = True) -> None:
            """Terminate the sandbox. No app cleanup needed."""
            try:
                await self._sandbox.terminate.aio()
            except Exception:
                pass

    return _PatchedEnv(
        environment_dir=Path("/tmp"),
        environment_name=session_id,
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=TaskEnvConfig(cpus=1.0, memory_mb=1024),
    )


# ---------------------------------------------------------------------------
# Sandbox I/O helper — Modal API returns str or bytes depending on version
# ---------------------------------------------------------------------------

async def _read_stdout(exec_result) -> str:
    """Read stdout from a sandbox exec result, handling both str and bytes."""
    raw = await exec_result.stdout.read.aio()
    if isinstance(raw, bytes):
        return raw.decode(errors="replace").strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Proxy config injection — write config files into sandbox for CLI agents
# ---------------------------------------------------------------------------

async def _inject_proxy_config(agent_name: str, extra_env: dict, harbor_env, trial_paths):
    """Write proxy config files into the sandbox AFTER agent install, BEFORE run.

    Codex CLI: writes config.toml with a custom model_provider pointing to UniAPI.
    Claude Code: writes .claude/settings.local.json with proxy env overrides.
                 This is needed because Harbor merges AUTH_TOKEN into API_KEY,
                 but Claude Code CLI requires API_KEY="" (empty) and AUTH_TOKEN
                 set separately when using a proxy.
    """
    openai_base = extra_env.get("OPENAI_BASE_URL", "")
    anthropic_base = extra_env.get("ANTHROPIC_BASE_URL", "")
    anthropic_token = extra_env.get("ANTHROPIC_AUTH_TOKEN", "")
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

    elif agent_name == "claude-code" and anthropic_base and anthropic_token:
        # Claude Code with proxy (OpenRouter/UniAPI) requires:
        #   ANTHROPIC_BASE_URL = proxy URL
        #   ANTHROPIC_AUTH_TOKEN = proxy API key
        #   ANTHROPIC_API_KEY = "" (explicitly empty — NOT absent, EMPTY)
        #
        # Harbor's create_run_agent_commands() merges AUTH_TOKEN into API_KEY
        # and strips empty values (line 834), which breaks this.
        #
        # Fix: wrap the `claude` binary so the correct env vars are always set
        # regardless of what Harbor passes. The wrapper intercepts the call
        # and overrides the env before exec-ing the real binary.

        model = model_name or "claude-sonnet-4-20250514"

        wrapper_script = f"""#!/bin/bash
# Proxy wrapper — overrides Harbor's env vars for Claude Code CLI
export ANTHROPIC_API_KEY=""
export ANTHROPIC_AUTH_TOKEN="{anthropic_token}"
export ANTHROPIC_BASE_URL="{anthropic_base}"
export ANTHROPIC_MODEL="{model}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="{model}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="{model}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="{model}"
export CLAUDE_CODE_SUBAGENT_MODEL="{model}"
# Disable features that Bedrock/proxy don't support
export DISABLE_PROMPT_CACHING="1"
export MAX_THINKING_TOKENS="0"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"
# Source nvm so node is on PATH (needed for npm-installed claude)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
exec /root/.local/bin/claude.real "$@"
"""
        # Pin Claude Code to known-working version, then install proxy wrapper.
        # Harbor's install.sh installs latest (2.1.101+) via curl which sends
        # beta flags incompatible with Bedrock-routed proxies like UniAPI.
        # Version 2.1.92 works. We replace the curl-installed binary entirely.
        #
        # Harbor's run command uses: export PATH="$HOME/.local/bin:$PATH"; claude ...
        # So the wrapper MUST be at /root/.local/bin/claude
        pin_version = "2.1.92"
        rename_and_wrap = (
            "if [ -f /root/.local/bin/claude ] && [ ! -f /root/.local/bin/claude.real ]; then "
            # Install node via nvm (sandbox may not have it)
            "  if ! command -v node >/dev/null 2>&1; then "
            "    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
            "    export NVM_DIR=\"$HOME/.nvm\" && . \"$NVM_DIR/nvm.sh\" || true && "
            "    nvm install 22; "
            "  else "
            "    export NVM_DIR=\"$HOME/.nvm\" && [ -s \"$NVM_DIR/nvm.sh\" ] && . \"$NVM_DIR/nvm.sh\" || true; "
            "  fi && "
            # Completely remove the curl-installed claude
            "  rm -f /root/.local/bin/claude && "
            # Install pinned version via npm globally
            f"  npm install -g @anthropic-ai/claude-code@{pin_version} && "
            # Create a claude.real at .local/bin that delegates to npm's claude
            "  NPM_CLAUDE=$(which claude) && "
            "  cat > /root/.local/bin/claude.real << REALEOF\n"
            "#!/bin/bash\n"
            "export NVM_DIR=\"\\$HOME/.nvm\"\n"
            "[ -s \"\\$NVM_DIR/nvm.sh\" ] && . \"\\$NVM_DIR/nvm.sh\"\n"
            "exec \"$NPM_CLAUDE\" \"\\$@\"\n"
            "REALEOF\n"
            "  chmod +x /root/.local/bin/claude.real && "
            # Write proxy wrapper at /root/.local/bin/claude
            "  cat > /root/.local/bin/claude << 'WRAPEOF'\n"
            f"{wrapper_script}"
            "WRAPEOF\n"
            "  chmod +x /root/.local/bin/claude && "
            f"  /root/.local/bin/claude.real --version && "
            f"  echo '[proxy] Claude {pin_version} + wrapper installed'; "
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


# ---------------------------------------------------------------------------
# Core per-task runner
# ---------------------------------------------------------------------------

async def _run_task(
    task_name: str,
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    task_meta: dict,
    learning_config: dict = None,
    tasks_dir: str = "",
) -> dict:
    """
    Run one task end-to-end. Returns an ENRICHED result dict.

    learning_config (optional):
      - history_from: str — run tag to load previous trace from
      - use_skills_from: str — path to skills dir on volume
      - extract_skills: bool — extract skill after evaluation
      - skill_threshold: float — min score for extraction (default 3.0)
      - skill_model: str — LLM model for skill extraction (default gpt-4o-mini)
    tasks_dir: subfolder name under tasks/ (e.g. "batch-1", "updated-deprivacy-100")
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.context import AgentContext
    from harbor.models.agent.name import AgentName
    from harbor.models.trial.paths import TrialPaths

    learning_config = learning_config or {}
    started_at = datetime.now(timezone.utc).isoformat()

    task_dir = _get_tasks_base(tasks_dir) / task_name
    instruction = (task_dir / "instruction.md").read_text().strip()
    instruction += OUTPUT_DIR_INSTRUCTION

    # --- Learning injection: history trace ---
    history_tag = learning_config.get("history_from", "")
    if history_tag:
        sys.path.insert(0, "/harness")
        from history import build_history_injection
        history_text = build_history_injection(RESULTS_BASE, history_tag, task_name)
        if history_text:
            instruction = history_text + "\n" + instruction
            print(f"[{task_name}] Injected history from {history_tag}")
        else:
            print(f"[{task_name}] No history found for {history_tag}")

    # --- Learning injection: skills ---
    skills_dir = learning_config.get("use_skills_from", "")
    if skills_dir:
        sys.path.insert(0, "/harness")
        from skill_store import SkillStore
        store = SkillStore(skills_dir=skills_dir)

        same_cat_only = learning_config.get("same_category_only", False)
        task_category = task_meta.get("category", "")

        if same_cat_only and task_category:
            # Filter: only load skills whose metadata.category matches this task
            all_skills = store.list_skills()
            cat_key = task_category.lower().replace(" & ", "_").replace(" ", "_")
            matched = [
                s for s in all_skills
                if cat_key in [t.lower() for t in (s.get("metadata", {}).get("tags", []))]
                or s.get("metadata", {}).get("category", "").lower() == task_category.lower()
            ]
            if matched:
                skill_bodies = [s["full_body"] for s in matched[:3]]
            else:
                skill_bodies = []
            print(f"[{task_name}] Category filter '{task_category}': {len(matched)} skill(s) matched")
        else:
            # LLM-based search across all skills
            skill_bodies = store.search_skills(task_description=instruction, top_k=3)

        if skill_bodies:
            combined = "\n\n---\n\n".join(skill_bodies)
            instruction = (
                "Here are some relevant strategies from similar past tasks:\n\n"
                f"{combined}\n\n"
                "---\n\n"
                "Now, here is your actual task:\n\n"
                f"{instruction}"
            )
            print(f"[{task_name}] Injected {len(skill_bodies)} skill(s)")
        else:
            print(f"[{task_name}] No relevant skills found")

    if agent_name == "codex":
        instruction = _wrap_codex_instruction(instruction)

    trial_dir = RESULTS_BASE / run_tag / "trials" / task_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    # Modal app names: alphanumeric + dashes + dots + underscores only
    safe_tag = run_tag.replace("/", "-").replace(" ", "-")
    session_id = f"dp-{safe_tag}-{task_name}"[:63].rstrip("-")
    env = _make_harbor_env(session_id, trial_paths)

    # Inject extra_env into os.environ so Harbor's create_run_agent_commands()
    # picks up credentials (it reads from os.environ, not extra_env directly)
    for k, v in agent_kwargs.get("extra_env", {}).items():
        os.environ[k] = v  # use direct set, not setdefault — we want to override

    agent = AgentFactory.create_agent_from_name(
        AgentName(agent_name),
        logs_dir=trial_paths.agent_dir,
        **agent_kwargs,
    )
    context = AgentContext()

    print(f"[{task_name}] Starting sandbox ...")
    try:
        await env.start(force_build=False)
    except Exception as exc:
        print(f"[{task_name}] Sandbox start failed: {exc}")
        return _build_result(
            task_name, agent_name, agent_kwargs, run_tag, task_meta,
            started_at, score=0.0, passed=False,
            error=str(exc), eval_result={},
        )

    try:
        print(f"[{task_name}] Setting up agent ({agent_name}) ...")
        await agent.setup(env)

        # --- Inject proxy config for agents that need it ---
        extra = agent_kwargs.get("extra_env", {})
        await _inject_proxy_config(agent_name, extra, env, trial_paths)

        print(f"[{task_name}] Running agent ...")
        await agent.run(instruction, env, context)
    except Exception as exc:
        print(f"[{task_name}] Agent error: {exc}")
    finally:
        # Collect artifacts from /output/ (primary) and fallback scan of /root /home
        artifacts_dir = RESULTS_BASE / run_tag / "artifacts" / task_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Collect everything from /output/ (the dedicated output dir)
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

                    # Save to artifacts dir on volume (persisted)
                    (artifacts_dir / fname).write_text(content)
                    # Also save to agent dir for evaluation
                    (trial_paths.agent_dir / f"artifact_{fname}").write_text(content)
                    print(f"[{task_name}] Collected: {fpath} ({len(content)} chars)")
                except Exception:
                    pass

            if all_files:
                await results_volume.commit.aio()
        except Exception as e:
            print(f"[{task_name}] Artifact collection error: {e}")

        try:
            await env.stop(delete=True)
        except Exception:
            pass

    # --- Parse agent output ---
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
    print(f"[{task_name}] Parsed task_result: {len(result_dict.get('task_result', ''))} chars")

    # Append artifacts from /output/ to task_result so the judge can evaluate them
    artifact_texts = []
    # Check agent dir artifacts
    agent_artifacts = sorted(trial_paths.agent_dir.glob("artifact_*")) if trial_paths.agent_dir.exists() else []
    # Also check the volume artifacts dir
    volume_artifacts = sorted(artifacts_dir.glob("*")) if artifacts_dir.exists() else []
    print(f"[{task_name}] Agent dir artifacts: {len(agent_artifacts)}, Volume artifacts: {len(volume_artifacts)}")

    # Prefer volume artifacts (always written), fall back to agent dir
    source_files = volume_artifacts if volume_artifacts else agent_artifacts
    for artifact_file in source_files:
        if not artifact_file.is_file():
            continue
        content = artifact_file.read_text()
        fname = artifact_file.name.removeprefix("artifact_")
        artifact_texts.append(f"\n\n=== FILE: {fname} ===\n{content}\n=== END FILE ===")
    if artifact_texts:
        result_dict["task_result"] = (result_dict.get("task_result") or "") + "".join(artifact_texts)
        print(f"[{task_name}] Appended {len(artifact_texts)} artifact(s) to task_result")

    # --- List collected artifacts ---
    collected_artifacts = sorted(
        f.name for f in artifacts_dir.iterdir() if f.is_file()
    ) if artifacts_dir.exists() else []

    # --- Evaluate ---
    eval_result = _evaluate(task_name, result_dict, tasks_dir=tasks_dir)

    score = float(
        eval_result.get("details", {}).get("overall_score", 0)
        or eval_result.get("overall_score", 0)
    )
    passed = bool(eval_result.get("passed", False))

    # --- Learning extraction: store skill if good enough ---
    extracted_skill = None
    if learning_config.get("extract_skills") and score > 0:
        sys.path.insert(0, "/harness")
        from skill_extractor import SkillExtractor
        from skill_store import SkillStore

        threshold = learning_config.get("skill_threshold", 3.0)
        skill_model = learning_config.get("skill_model", "gpt-4o-mini")
        skills_out = str(RESULTS_BASE / run_tag / "skills")

        if score >= threshold:
            extractor = SkillExtractor(model=skill_model)
            config = {"skill_score_threshold": threshold}
            decision = extractor.should_store(result_dict, eval_result, config)

            if decision.get("store"):
                task_data = {"task_name": task_name, "instruction": instruction}
                skill_md = extractor.extract_skill(task_data, result_dict, eval_result)

                store = SkillStore(skills_dir=skills_out)
                metadata = {
                    "score": score,
                    "task_type": decision.get("task_type", ""),
                    "tags": decision.get("tags", []),
                    "agent": agent_name,
                }
                path = store.save_skill(skill_md, task_name, metadata)
                extracted_skill = skill_md
                await results_volume.commit.aio()
                print(f"[{task_name}] Extracted skill -> {path.name}")
            else:
                print(f"[{task_name}] Skill not generalizable: {decision.get('reason', '')}")

    result = _build_result(
        task_name, agent_name, agent_kwargs, run_tag, task_meta,
        started_at, score=score, passed=passed,
        eval_result=eval_result,
        artifacts=collected_artifacts,
        task_result=result_dict.get("task_result", ""),
    )
    if extracted_skill:
        result["extracted_skill"] = extracted_skill
    return result


def _build_result(
    task_name: str,
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    task_meta: dict,
    started_at: str,
    score: float,
    passed: bool,
    eval_result: dict = None,
    error: str = "",
    artifacts: list[str] = None,
    task_result: str = "",
) -> dict:
    """Build the enriched result dict with all metadata for cross-agent analysis."""
    finished_at = datetime.now(timezone.utc).isoformat()
    eval_result = eval_result or {}
    details = eval_result.get("details", {}) or {}
    dimension_scores = details.get("dimension_scores", {}) or {}

    model_name = agent_kwargs.get("model_name", "")
    if not model_name:
        model_name = (agent_kwargs.get("extra_env", {}) or {}).get("LLM_MODEL", "default")

    return {
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
        "votes_used": details.get("votes_used", 1),
        "pass_threshold": details.get("pass_threshold", 3.0),

        # --- Timing ---
        "started_at": started_at,
        "finished_at": finished_at,

        # --- Agent output (truncated for storage, full version in trials/) ---
        "task_result": task_result[:10000] if task_result else "",

        # --- Artifacts ---
        "artifacts": artifacts or [],

        # --- Learning ---
        "is_seed": task_name in _SEED_TASKS,

        # --- Error ---
        "error": error,
    }


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------

@app.function(
    image=harbor_image,
    timeout=3600,
    volumes={str(RESULTS_BASE): results_volume},
    max_containers=20,
)
async def run_task_remote(
    task_name: str,
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    task_meta: dict,
    learning_config: dict = None,
    tasks_dir: str = "",
) -> dict:
    return await _run_task(task_name, agent_name, agent_kwargs, run_tag, task_meta, learning_config, tasks_dir)


@app.function(
    image=harbor_image,
    timeout=14400,  # 4 hours
    volumes={str(RESULTS_BASE): results_volume},
)
async def run_deprivacy_experiment(
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    first_n: int = 0,
    task_names_csv: str = "",
    learning_config: dict = None,
    tasks_dir: str = "",
) -> dict:
    """Orchestrate a full experiment run for one agent on any task set."""
    learning_config = learning_config or {}
    tasks_dir = tasks_dir or DEFAULT_TASKS_DIR
    tasks = _discover_tasks(first_n=first_n, task_names_csv=task_names_csv, tasks_dir=tasks_dir)

    model_name = agent_kwargs.get("model_name", "default")
    learning_desc = ""
    if learning_config.get("history_from"):
        learning_desc += f"  History from: {learning_config['history_from']}\n"
    if learning_config.get("use_skills_from"):
        learning_desc += f"  Skills from: {learning_config['use_skills_from']}\n"
    if learning_config.get("extract_skills"):
        learning_desc += f"  Extract skills: threshold={learning_config.get('skill_threshold', 3.0)}\n"

    print(f"\n{'='*60}")
    print(f"  Experiment: {run_tag}")
    print(f"  Agent: {agent_name}  Model: {model_name}")
    print(f"  Tasks dir: {tasks_dir} ({len(tasks)} tasks)")
    if learning_desc:
        print(learning_desc.rstrip())
    print(f"{'='*60}\n")

    starmap_args = [
        (t["task_name"], agent_name, agent_kwargs, run_tag, t, learning_config, tasks_dir)
        for t in tasks
    ]

    results = []
    async for result in run_task_remote.starmap.aio(starmap_args):
        results.append(result)
        n_done = len(results)
        scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
        print(
            f"  [{n_done}/{len(tasks)}] {result['task']:50s}  "
            f"score={result.get('score', 0):.2f}"
        )
        if scores:
            print(f"    running avg = {sum(scores)/len(scores):.3f}")

    # --- Save per-task results ---
    for r in results:
        task_path = RESULTS_BASE / run_tag / f"{r['task']}.json"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(json.dumps(r, indent=2, default=str))

    # --- Build summary ---
    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    errors = [r for r in results if r.get("error")]

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

    # Per-dimension averages (across all tasks)
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
        "category_summary": category_summary,
        "dimension_averages": dimension_averages,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = RESULTS_BASE / run_tag / "_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    await results_volume.commit.aio()

    print(f"\n{'='*60}")
    print(f"  Experiment Complete: {run_tag}")
    print(f"  Mean score : {summary['mean_score']:.3f}")
    print(f"  Pass rate  : {summary['n_passed']}/{summary['n_tasks']} ({summary['pass_rate_pct']}%)")
    print(f"  Errors     : {summary['n_errors']}")
    print(f"\n  Category breakdown:")
    for cat, info in category_summary.items():
        print(f"    {cat:30s}  n={info['count']:3d}  avg={info['mean_score']:.2f}  pass={info['pass_rate']}%")
    print(f"\n  Dimension averages:")
    for dim, avg in dimension_averages.items():
        print(f"    {dim:30s}  {avg:.2f}")
    print(f"{'='*60}")

    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(collect: str = "", list_runs: bool = False):
    import shutil
    modal_bin = shutil.which("modal") or os.path.expanduser("~/.local/bin/modal")

    if list_runs:
        subprocess.run([modal_bin, "volume", "ls", "deprivacy-100-results"])
        return

    if collect:
        # Derive tasks_dir from run tag: "real118-codex" -> "real118", "synthetic_382-claude" -> "synthetic_382"
        parts = collect.rsplit("-", 1)
        tasks_dir_name = parts[0] if len(parts) > 1 else collect
        out_dir = Path(f"results/{tasks_dir_name}/{collect}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {collect} -> {out_dir}")
        subprocess.run(
            [modal_bin, "volume", "get", "deprivacy-100-results", collect, str(out_dir), "--force"],
            check=True,
        )
        print(f"Done. Results in {out_dir}")
        return

    print("Use scripts_batch/trigger_deprivacy.py to dispatch experiments.")
