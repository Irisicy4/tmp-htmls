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
        "tasks/updated-deprivacy-100",
        "/harbor-bench/tasks/updated-deprivacy-100",
        copy=True,
    )
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASKS_BASE = Path("/harbor-bench/tasks/updated-deprivacy-100")
RESULTS_BASE = Path("/results")
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


def _discover_tasks(first_n: int = 0, task_names_csv: str = "") -> list[dict]:
    """Discover tasks and attach metadata."""
    if task_names_csv:
        names = [t.strip() for t in task_names_csv.split(",") if t.strip()]
    else:
        names = sorted(
            d.name for d in TASKS_BASE.iterdir()
            if d.is_dir() and (d / "instruction.md").exists()
        )
        if first_n > 0:
            names = names[:first_n]

    tasks = []
    for name in names:
        task_dir = TASKS_BASE / name
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

def _evaluate(task_name: str, result: dict) -> dict:
    test_script = TASKS_BASE / task_name / "tests" / "test_task.py"
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
    import modal as _modal
    from harbor.environments.modal import ModalEnvironment
    from harbor.models.task.config import EnvironmentConfig as TaskEnvConfig
    from harbor.models.trial.paths import EnvironmentPaths

    class _PatchedEnv(ModalEnvironment):
        def _validate_definition(self):
            pass

        async def start(self, force_build: bool) -> None:
            self._image = _modal.Image.from_registry(SANDBOX_BASE_IMAGE)
            self._app = await _modal.App.lookup.aio(
                name=self.session_id, create_if_missing=True,
            )
            self._sandbox = await self._create_sandbox(
                gpu_config=None, secrets_config=[], volumes_config={},
            )
            await self._sandbox.mkdir.aio(str(EnvironmentPaths.agent_dir), parents=True)
            await self._sandbox.mkdir.aio(str(EnvironmentPaths.verifier_dir), parents=True)

        async def stop(self, delete: bool = True) -> None:
            """Stop sandbox and delete the ephemeral app to avoid hitting the 200 app limit."""
            try:
                await super().stop(delete=delete)
            except Exception:
                pass
            # Delete the deployed app so it doesn't linger
            try:
                app = await _modal.App.lookup.aio(name=self.session_id)
                await app.stop.aio()
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
    Claude Code: no extra config needed — ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL
                 are passed via env vars by Harbor.
    """
    openai_base = extra_env.get("OPENAI_BASE_URL", "")

    if agent_name == "codex" and openai_base:
        from harbor.models.trial.paths import EnvironmentPaths
        codex_home = EnvironmentPaths.agent_dir.as_posix()

        # Write config.toml with uniapi as a custom provider
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


# ---------------------------------------------------------------------------
# Core per-task runner
# ---------------------------------------------------------------------------

async def _run_task(
    task_name: str,
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    task_meta: dict,
) -> dict:
    """
    Run one task end-to-end. Returns an ENRICHED result dict with:
      - agent, model, category, tags  (for cross-agent analysis)
      - timestamps (started_at, finished_at, elapsed_sec)
      - dimension_scores (flattened from eval details)
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.context import AgentContext
    from harbor.models.agent.name import AgentName
    from harbor.models.trial.paths import TrialPaths

    started_at = datetime.now(timezone.utc).isoformat()

    task_dir = TASKS_BASE / task_name
    instruction = (task_dir / "instruction.md").read_text().strip()
    instruction += OUTPUT_DIR_INSTRUCTION

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
                break

    parser = AGENT_OUTPUT_PARSERS.get(agent_name, _parse_generic_output)
    result_dict = parser(agent_stdout)

    artifact_texts = []
    for artifact_file in sorted(trial_paths.agent_dir.glob("artifact_*")):
        content = artifact_file.read_text()
        fname = artifact_file.name[len("artifact_"):]
        artifact_texts.append(f"\n\n=== FILE: {fname} ===\n{content}\n=== END FILE ===")
    if artifact_texts:
        result_dict["task_result"] = (result_dict.get("task_result") or "") + "".join(artifact_texts)

    # --- List collected artifacts ---
    collected_artifacts = sorted(
        f.name for f in artifacts_dir.iterdir() if f.is_file()
    ) if artifacts_dir.exists() else []

    # --- Evaluate ---
    eval_result = _evaluate(task_name, result_dict)

    score = float(
        eval_result.get("details", {}).get("overall_score", 0)
        or eval_result.get("overall_score", 0)
    )
    passed = bool(eval_result.get("passed", False))

    return _build_result(
        task_name, agent_name, agent_kwargs, run_tag, task_meta,
        started_at, score=score, passed=passed,
        eval_result=eval_result,
        artifacts=collected_artifacts,
    )


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

        # --- Artifacts ---
        "artifacts": artifacts or [],

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
) -> dict:
    return await _run_task(task_name, agent_name, agent_kwargs, run_tag, task_meta)


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
) -> dict:
    """Orchestrate a full deprivacy-100 run for one agent."""
    tasks = _discover_tasks(first_n=first_n, task_names_csv=task_names_csv)

    model_name = agent_kwargs.get("model_name", "default")
    print(f"\n{'='*60}")
    print(f"  Deprivacy-100 Experiment: {run_tag}")
    print(f"  Agent: {agent_name}  Model: {model_name}")
    print(f"  Tasks: {len(tasks)}")
    print(f"{'='*60}\n")

    starmap_args = [
        (t["task_name"], agent_name, agent_kwargs, run_tag, t)
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
        "tasks_dir": "tasks/updated-deprivacy-100",
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
        out_dir = Path(f"results/deprivacy/{collect}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {collect} -> {out_dir}")
        subprocess.run(
            [modal_bin, "volume", "get", "deprivacy-100-results", collect, str(out_dir), "--force"],
            check=True,
        )
        print(f"Done. Results in {out_dir}")
        return

    print("Use scripts_batch/trigger_deprivacy.py to dispatch experiments.")
