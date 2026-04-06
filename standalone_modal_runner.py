"""
Standalone Modal runner — runs skill experiments entirely on Modal.

Two-step workflow:
  1. Deploy:  modal deploy standalone_modal_runner.py
  2. Trigger: python trigger_experiment.py [--tasks-dir tasks/batch-1] [--first-n 3]

The orchestrator runs as a deployed Modal function, so it keeps running
even after the trigger script (and your laptop) disconnects.
Results are saved to a Modal Volume.

You can also use `modal run` for blocking execution (laptop must stay awake):
  modal run standalone_modal_runner.py --tasks-dir tasks/batch-1 --first-n 3

Utilities (always safe to run, no laptop dependency):
  modal run standalone_modal_runner.py --collect <run-tag>
  modal run standalone_modal_runner.py --list-runs

Prerequisites
-------------
1. pip install modal
2. modal setup
3. cp env.template .env   (set OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL)
4. modal secret create openai-secret OPENAI_API_KEY=sk-...
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import modal

# ---------------------------------------------------------------------------
# Modal image — mirrors task Dockerfiles: sandbox base → cocoa-agent → glue
# ---------------------------------------------------------------------------

IGNORE_PATTERNS = [
    ".git", ".git/**",
    "__pycache__", "**/__pycache__",
    "*.pyc", "*.pyo",
    "results", "results/**",
    "jobs", "jobs/**",
    ".venv", ".venv/**",
    "venv", "venv/**",
]

modal_image = (
    modal.Image.from_registry("ghcr.io/agent-infra/sandbox:latest")
    # Patch sandbox startup so supervisord runs in background (not exec)
    .dockerfile_commands(
        "RUN sed -i "
        r"'s|^exec /usr/bin/supervisord -n -c /opt/gem/supervisord.conf$"
        r"|/usr/bin/supervisord -n -c /opt/gem/supervisord.conf \&|' "
        "/opt/gem/entrypoint.sh",
        "RUN sed -i "
        r"'s|^exec /opt/gem/entrypoint.sh$|/opt/gem/entrypoint.sh|' "
        "/opt/gem/run.sh",
        "ENTRYPOINT []",
        "CMD []",
    )
    # Fix broken package deps in the base sandbox image
    .run_commands(
        "/opt/python3.12/bin/pip install --upgrade "
        "typing_extensions sniffio anyio "
        "python-json-logger traitlets jupyter_client "
        "terminado referencing jsonschema pygments "
        "--target /opt/python3.12/lib/python3.12/site-packages/ --quiet"
    )
    # Agent + sandbox dependencies
    .pip_install(
        "openai>=2.6.1",
        "anthropic>=0.75.0",
        "google-genai>=0.5.0",
        "agent-sandbox>=0.0.16",
        "websocket-client>=1.9.0",
        "pyyaml>=6.0.3",
        "requests>=2.32.5",
        "colorama>=0.4.6",
        "numpy>=1.24.0",
        "playwright>=1.44.0",
        "playwright-stealth>=2.0.0",
    )
    # Clone cocoa-agent (same as task Dockerfiles)
    .run_commands(
        "git clone --depth 1 https://github.com/cocoabench/cocoa-agent.git /cocoa-agent-src"
        " && ln -sf /cocoa-agent-src /cocoa-agent"
    )
    # Install Codex CLI (for CodexAgent) and Claude Code CLI (for ClaudeCodeAgent)
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
        " && apt-get install -y nodejs"
        " && npm install -g @openai/codex @anthropic-ai/claude-code"
    )
    # Upload this repo (agent glue, configs, task files)
    .add_local_dir(".", "/harbor-bench", copy=True, ignore=IGNORE_PATTERNS)
    # Copy harness files into /harness/ (agent-agnostic entry point)
    .run_commands(
        "mkdir -p /harness/adapters /harness/configs /harness/agents"
        " && cp /harbor-bench/harness/run_task.py /harness/"
        " && cp /harbor-bench/harness/skill_store.py /harness/"
        " && cp /harbor-bench/harness/skill_extractor.py /harness/"
        " && cp /harbor-bench/harness/adapters/__init__.py /harness/adapters/"
        " && cp /harbor-bench/harness/adapters/cocoa_adapter.py /harness/adapters/"
        " && cp -r /harbor-bench/configs/. /harness/configs/"
        " && cp /harbor-bench/agents/__init__.py /harness/agents/"
        " && cp /harbor-bench/agents/codex_agent.py /harness/agents/"
        " && cp /harbor-bench/agents/claude_code_agent.py /harness/agents/"
    )
    .workdir("/harness")
)

# ---------------------------------------------------------------------------
# Modal app & infrastructure
# ---------------------------------------------------------------------------

app = modal.App("evolve-bench-harbor", image=modal_image)

results_volume = modal.Volume.from_name("evolve-bench-results", create_if_missing=True)

# Create with:  modal secret create openai-secret OPENAI_API_KEY=sk-...
_SECRETS = [modal.Secret.from_name("openai-secret")]


# ---------------------------------------------------------------------------
# Task discovery (runs inside Modal container, reads baked-in repo files)
# ---------------------------------------------------------------------------

def _discover_tasks(
    tasks_dir: str = "tasks/batch-1",
    first_n: int = 0,
    task_names_csv: str = "",
) -> List[Dict[str, Any]]:
    """Discover tasks from baked-in repo files at /harbor-bench/."""
    base = Path(f"/harbor-bench/{tasks_dir}")
    if not base.exists():
        raise FileNotFoundError(f"Tasks dir not found in image: {base}")

    if (base / "instruction.md").exists():
        candidates = [base]
    else:
        candidates = sorted(d for d in base.iterdir() if d.is_dir())

    tasks: List[Dict[str, Any]] = []
    for task_dir in candidates:
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.exists():
            continue
        rel_path = str(task_dir.relative_to(Path("/harbor-bench")))
        tasks.append({
            "task_name": task_dir.name,
            "task_path": rel_path,
            "instruction": instruction_file.read_text().strip(),
        })

    # Filter
    if task_names_csv:
        names = [n.strip() for n in task_names_csv.split(",") if n.strip()]
        tasks = [t for t in tasks if t["task_name"] in names]
    elif first_n > 0:
        tasks = tasks[:first_n]

    return tasks


# ---------------------------------------------------------------------------
# Remote function — one invocation per task
# ---------------------------------------------------------------------------

@app.function(
    secrets=_SECRETS,
    timeout=3600,
    volumes={"/results": results_volume},
    cpu=2,
    memory=4096,
    single_use_containers=True,
    max_containers=10,
)
def run_single_task(
    task: Dict[str, Any],
    config: Dict[str, Any],
    run_tag: str,
) -> Dict[str, Any]:
    """Execute a single task inside a Modal container."""
    import importlib.util
    import subprocess
    import time as _time

    task_name = task["task_name"]

    try:
        sys.path.insert(0, "/harness")

        from run_task import (
            create_agent, apply_env_overrides, run_agent_task, _strip_images,
            maybe_inject_skills, maybe_store_skill, _SANDBOX_AGENTS,
        )

        instruction = task["instruction"]
        agent_type = config.get("agent_type", "cocoa")

        # Logging
        if agent_type == "cocoa":
            from adapters.cocoa_adapter import setup_logging
            setup_logging(config.get("log_level", "INFO"))
        else:
            import logging
            logging.basicConfig(
                level=getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO),
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
        config = apply_env_overrides(config)

        # Start sandbox for browser-based agents
        if agent_type in _SANDBOX_AGENTS:
            setup = subprocess.run(["/bin/bash", "/opt/gem/run.sh"], check=False)
            if setup.returncode != 0:
                print(f"[warn] Sandbox setup exited {setup.returncode} — continuing anyway")

            import requests as _req
            for attempt in range(60):
                try:
                    r = _req.get("http://localhost:8080/v1/ping", timeout=2)
                    if r.status_code == 200:
                        print(f"Sandbox ready after {(attempt + 1) * 2}s")
                        break
                except Exception:
                    pass
                _time.sleep(2)
            else:
                raise RuntimeError("Sandbox did not become ready within 120 seconds")
        else:
            print(f"Agent type '{agent_type}' does not use the sandbox — skipping startup.")

        # Create agent and run task
        agent = create_agent(config)
        task_data = {
            "task_name": task_name,
            "instruction": instruction,
            "description": instruction,
            "task_dir": "/tmp",
            "use_encrypted": False,
        }

        task_data = maybe_inject_skills(task_data, config)

        task_start = _time.time()
        result = run_agent_task(agent, task_data)

        # Run LLM-as-judge verification
        eval_result = None
        task_path = task.get("task_path", task_name)
        test_file = f"/harbor-bench/{task_path}/tests/test_task.py"
        if os.path.exists(test_file):
            try:
                spec = importlib.util.spec_from_file_location("test_task", test_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                eval_result = mod.test(result)
                result["eval"] = eval_result
            except Exception as e:
                print(f"[warn] Verification failed for {task_name}: {e}")
                result["eval"] = {"passed": False, "error": str(e)}
        else:
            print(f"[warn] No test_task.py for {task_name}")

        # Phase 1: store skill if result is good enough
        if eval_result is not None:
            maybe_store_skill(task_data, result, eval_result, config)

        # Capture extracted skill
        skill_path = Path(f"/skills/{task_name}.md")
        if skill_path.exists():
            result["extracted_skill"] = skill_path.read_text()

        # Finalize
        result = _strip_images(result)
        elapsed = _time.time() - task_start
        result["elapsed_seconds"] = round(elapsed, 1)
        result.setdefault("task_name", task_name)
        print(f"[timing] {task_name} finished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    except Exception as e:
        import traceback
        print(f"[error] {task_name}: {e}\n{traceback.format_exc()}")
        result = {
            "task_name": task_name,
            "status": "error",
            "error": str(e),
            "eval": {"passed": False, "error": str(e)},
        }

    # Persist to Modal Volume
    out_path = Path(f"/results/{run_tag}/{task_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    return result


# ---------------------------------------------------------------------------
# Orchestrator — runs on Modal (survives laptop disconnect when deployed)
# ---------------------------------------------------------------------------

@app.function(
    secrets=_SECRETS,
    timeout=14400,  # 4 hours
    volumes={"/results": results_volume},
    cpu=1,
    memory=512,
)
def run_experiment(
    config: Dict[str, Any],
    run_tag: str,
    phase_label: str = "",
    tasks_dir: str = "tasks/batch-1",
    first_n: int = 0,
    task_names_csv: str = "",
    tasks_override: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all tasks in parallel on Modal. Returns summary dict."""
    import time as _time

    # Discover tasks from baked-in image (or use override for Phase 2 with skills)
    if tasks_override:
        tasks = tasks_override
    else:
        tasks = _discover_tasks(tasks_dir, first_n, task_names_csv)

    print(f"=== {phase_label or 'Experiment'}: {len(tasks)} tasks, tag={run_tag} ===")
    wall_start = _time.time()

    pairs = [(task, config, run_tag) for task in tasks]
    results: List[Dict[str, Any]] = list(run_single_task.starmap(pairs))

    # Aggregate
    passed, errors = 0, 0
    scores = []
    skills = {}

    for task, result in zip(tasks, results):
        name = task["task_name"]
        eval_info = result.get("eval", {}) or {}
        status = result.get("status", "unknown")

        if status == "error":
            errors += 1
        elif eval_info.get("passed"):
            passed += 1

        score = (eval_info.get("details", {}) or {}).get("overall_score")
        if score is not None:
            scores.append(float(score))

        skill = result.get("extracted_skill")
        if skill:
            skills[name] = skill

        elapsed_s = result.get("elapsed_seconds")
        elapsed_str = f" [{elapsed_s:.0f}s]" if elapsed_s else ""
        score_str = f" score={score}" if score is not None else ""
        marker = "+" if eval_info.get("passed") else "x"
        print(f"  {marker} {name}{elapsed_str}:{score_str}")

    total = len(tasks)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    wall_elapsed = _time.time() - wall_start

    summary = {
        "phase": phase_label,
        "run_tag": run_tag,
        "total": total,
        "passed": passed,
        "failed": total - passed - errors,
        "errors": errors,
        "pass_rate": round(100 * passed / total, 1) if total else 0,
        "avg_score": round(avg_score, 2),
        "wall_seconds": round(wall_elapsed, 1),
        "skills_extracted": len(skills),
    }

    print(f"\n--- {phase_label or 'Experiment'} Summary ---")
    print(f"  Total: {total}  Passed: {passed}  Failed: {summary['failed']}  Errors: {errors}")
    print(f"  Pass rate: {summary['pass_rate']}%  Avg score: {summary['avg_score']}")
    print(f"  Wall time: {wall_elapsed / 60:.1f} min")
    if skills:
        print(f"  Skills extracted: {len(skills)}")

    summary_path = Path(f"/results/{run_tag}/_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    results_volume.commit()

    return {"summary": summary, "results": results, "skills": skills}


@app.function(
    secrets=_SECRETS,
    timeout=28800,  # 8 hours for full 2-phase experiment
    volumes={"/results": results_volume},
    cpu=1,
    memory=512,
)
def run_skill_experiment(
    phase1_config: Dict[str, Any],
    phase2_config: Dict[str, Any],
    run_tag: str,
    tasks_dir: str = "tasks/batch-1",
    first_n: int = 0,
    task_names_csv: str = "",
) -> Dict[str, Any]:
    """Full skill experiment: Phase 1 → extract skills → Phase 2 → compare.

    Runs entirely on Modal — laptop can disconnect after dispatch.
    Tasks are loaded from the baked-in image (no data from local machine needed).
    """
    from datetime import datetime, timezone

    tasks = _discover_tasks(tasks_dir, first_n, task_names_csv)

    print("=" * 60)
    print(f"  Skill Experiment — {run_tag}")
    print(f"  Tasks: {len(tasks)}")
    print("=" * 60)

    # ---- Phase 1: run tasks, extract skills ----
    phase1_tag = f"{run_tag}/phase1"
    phase1_out = run_experiment.remote(
        phase1_config, phase1_tag, "Phase 1",
        tasks_override=tasks,
    )

    skills = phase1_out.get("skills", {})
    print(f"\n>>> Extracted {len(skills)} skill(s) from Phase 1")

    # Save skills to Volume
    for name, skill_body in skills.items():
        skill_path = Path(f"/results/{run_tag}/skills/{name}.md")
        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(skill_body)
    if skills:
        results_volume.commit()

    # ---- Phase 2: run tasks with skill injection ----
    phase2_config = {**phase2_config, "skills_dir": f"/results/{run_tag}/skills"}
    phase2_tag = f"{run_tag}/phase2"
    phase2_out = run_experiment.remote(
        phase2_config, phase2_tag, "Phase 2",
        tasks_override=tasks,
    )

    # ---- Compare results ----
    p1_summary = phase1_out["summary"]
    p2_summary = phase2_out["summary"]

    comparison_lines = [
        "",
        "=" * 60,
        "  RESULTS COMPARISON",
        "=" * 60,
        "",
        f"{'Metric':<20} {'Phase 1':>10} {'Phase 2':>10} {'Delta':>10}",
        "-" * 52,
    ]

    for key in ["avg_score", "pass_rate", "passed", "errors"]:
        v1 = p1_summary[key]
        v2 = p2_summary[key]
        delta = v2 - v1
        sign = "+" if delta > 0 else ""
        if isinstance(v1, float):
            comparison_lines.append(f"  {key:<18} {v1:>10.2f} {v2:>10.2f} {sign}{delta:>9.2f}")
        else:
            comparison_lines.append(f"  {key:<18} {v1:>10d} {v2:>10d} {sign}{delta:>9d}")

    p1_results = {r["task_name"]: r for r in phase1_out["results"]}
    p2_results = {r["task_name"]: r for r in phase2_out["results"]}
    all_task_names = sorted(set(p1_results) | set(p2_results))

    improved, declined, unchanged = 0, 0, 0
    task_rows = []
    for name in all_task_names:
        s1 = (p1_results.get(name, {}).get("eval", {}) or {}).get("details", {}).get("overall_score", 0) or 0
        s2 = (p2_results.get(name, {}).get("eval", {}) or {}).get("details", {}).get("overall_score", 0) or 0
        delta = float(s2) - float(s1)
        if delta > 0:
            improved += 1
        elif delta < 0:
            declined += 1
        else:
            unchanged += 1
        sign = "+" if delta > 0 else ""
        short_name = name[:38] + ".." if len(name) > 40 else name
        task_rows.append(f"  {short_name:<40} {float(s1):>7.2f} {float(s2):>7.2f} {sign}{delta:>7.2f}")

    comparison_lines.append("")
    comparison_lines.append(f"{'Task':<42} {'Phase 1':>7} {'Phase 2':>7} {'Delta':>8}")
    comparison_lines.append("-" * 66)
    comparison_lines.extend(task_rows)
    comparison_lines.append("-" * 66)
    comparison_lines.append(f"  Improved: {improved}  |  Declined: {declined}  |  Unchanged: {unchanged}")
    comparison_lines.append(f"  Skills extracted: {len(skills)}")

    comparison_text = "\n".join(comparison_lines)
    print(comparison_text)

    # Save comparison to Volume
    comparison = {
        "run_tag": run_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase1": p1_summary,
        "phase2": p2_summary,
        "skills_extracted": len(skills),
        "improved": improved,
        "declined": declined,
        "unchanged": unchanged,
        "comparison_text": comparison_text,
    }
    comp_path = Path(f"/results/{run_tag}/_comparison.json")
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    results_volume.commit()

    return comparison


# ---------------------------------------------------------------------------
# Local entrypoint — for `modal run` (blocking) and utility commands
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    tasks_dir: str = "tasks/batch-1",
    phase: str = "skill-experiment",
    first_n: int = 0,
    task_names: str = "",
    skills_from: str = "",
    run_tag: str = "",
    collect: str = "",
    list_runs: bool = False,
    config: str = "",
    phase1_config: str = "",
    phase2_config: str = "",
):
    """Run experiments on Modal or manage results.

    For fire-and-forget (laptop can disconnect), use trigger_experiment.py instead.
    This entrypoint blocks until the experiment completes.

    --tasks-dir DIR           Task directory (default: tasks/batch-1)
    --phase PHASE             phase1, phase2, single, or skill-experiment (default)
    --first-n N               Only run first N tasks (0 = all)
    --task-names a,b,c        Comma-separated task names to run
    --skills-from TAG         For --phase phase2: run tag containing skills
    --run-tag TAG             Custom run tag (default: auto timestamp)
    --collect TAG             Download results for a run tag
    --list-runs               List all runs on the Volume
    --config PATH             Config JSON for --phase single (e.g. configs/codex-config.json)
    --phase1-config PATH      Override phase1 config (default: configs/skill-phase1.json)
    --phase2-config PATH      Override phase2 config (default: configs/skill-phase2.json)
    """
    from datetime import datetime, timezone

    if list_runs:
        _list_runs()
        return

    if collect:
        _collect_results(collect)
        return

    # Load .env for config overrides
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key and val and key not in os.environ:
                os.environ[key] = val

    # Load configs
    p1_path = phase1_config or "configs/skill-phase1.json"
    p2_path = phase2_config or "configs/skill-phase2.json"
    with open(p1_path) as f:
        phase1_cfg: Dict[str, Any] = json.load(f)
    with open(p2_path) as f:
        phase2_cfg: Dict[str, Any] = json.load(f)

    env_base_url = os.environ.get("LLM_BASE_URL")
    env_model = os.environ.get("LLM_MODEL")
    for cfg in [phase1_cfg, phase2_cfg]:
        if env_base_url:
            cfg.setdefault("controller", {}).setdefault("args", {})["base_url"] = env_base_url
        if env_model:
            cfg.setdefault("controller", {}).setdefault("args", {})["model"] = env_model

    if env_base_url:
        print(f"[.env] LLM_BASE_URL={env_base_url}")
    if env_model:
        print(f"[.env] LLM_MODEL={env_model}")

    tag = run_tag or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    print(f"Run tag:   {tag}")
    print(f"Tasks dir: {tasks_dir}")
    if first_n:
        print(f"First N:   {first_n}")
    print(f"Phase:     {phase}")
    print(f"\nResults will be saved to Volume regardless of laptop state.")
    print(f"Collect with: modal run standalone_modal_runner.py --collect {tag}\n")

    if phase == "skill-experiment":
        result = run_skill_experiment.remote(
            phase1_cfg, phase2_cfg, tag,
            tasks_dir=tasks_dir, first_n=first_n, task_names_csv=task_names,
        )
        print(result.get("comparison_text", ""))

    elif phase == "phase1":
        result = run_experiment.remote(
            phase1_cfg, f"{tag}/phase1", "Phase 1",
            tasks_dir=tasks_dir, first_n=first_n, task_names_csv=task_names,
        )
        s = result["summary"]
        print(f"\nPhase 1 done: {s['passed']}/{s['total']} passed, avg={s['avg_score']}")

    elif phase == "phase2":
        if not skills_from:
            print("[error] --skills-from <run-tag> required for phase2")
            sys.exit(1)
        phase2_cfg["skills_dir"] = f"/results/{skills_from}/skills"
        result = run_experiment.remote(
            phase2_cfg, f"{tag}/phase2", "Phase 2",
            tasks_dir=tasks_dir, first_n=first_n, task_names_csv=task_names,
        )
        s = result["summary"]
        print(f"\nPhase 2 done: {s['passed']}/{s['total']} passed, avg={s['avg_score']}")

    elif phase == "single":
        if not config:
            print("[error] --config <path> required for --phase single")
            sys.exit(1)
        with open(config) as f:
            single_cfg: Dict[str, Any] = json.load(f)
        if env_base_url:
            single_cfg.setdefault("controller", {}).setdefault("args", {})["base_url"] = env_base_url
        # Only apply LLM_MODEL if the config doesn't already specify a model
        existing_model = single_cfg.get("controller", {}).get("args", {}).get("model")
        if env_model and not existing_model:
            single_cfg.setdefault("controller", {}).setdefault("args", {})["model"] = env_model
        result = run_experiment.remote(
            single_cfg, tag, f"Single ({single_cfg.get('agent_type', '?')})",
            tasks_dir=tasks_dir, first_n=first_n, task_names_csv=task_names,
        )
        s = result["summary"]
        print(f"\nDone: {s['passed']}/{s['total']} passed, avg={s['avg_score']}")

    else:
        print(f"[error] Unknown phase: {phase}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Utility: list runs and collect results
# ---------------------------------------------------------------------------

def _list_runs():
    """List all experiment runs on the Volume."""
    print("Runs on evolve-bench-results volume:\n")
    try:
        for entry in results_volume.listdir("/"):
            name = entry.path.strip("/")
            markers = []
            try:
                for sub in results_volume.listdir(f"/{name}"):
                    sub_name = sub.path.rstrip("/").split("/")[-1]
                    if sub_name in ("_summary.json", "_comparison.json", "phase1", "phase2", "skills"):
                        markers.append(sub_name.strip("_").replace(".json", ""))
            except Exception:
                pass
            info = f" [{', '.join(markers)}]" if markers else ""
            print(f"  {name}{info}")
    except Exception as e:
        print(f"[error] {e}")


def _collect_results(run_tag: str):
    """Download results from Modal Volume to local disk."""
    import subprocess
    output_dir = Path(f"results/modal/{run_tag.replace('/', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{run_tag}' → {output_dir}/")

    modal_bin = Path(__file__).parent / ".." / ".." / ".local" / "bin" / "modal"
    if not modal_bin.exists():
        modal_bin = Path(os.path.expanduser("~/.local/bin/modal"))

    result = subprocess.run(
        [str(modal_bin), "volume", "get", "evolve-bench-results", run_tag, str(output_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[warn] modal volume get: {result.stderr.strip()}")
    else:
        files = list(output_dir.rglob("*"))
        print(f"Downloaded {sum(1 for f in files if f.is_file())} file(s) to {output_dir}/")

    # Print summary if available
    for name in ["_comparison.json", "_summary.json"]:
        f = output_dir / name
        if f.exists():
            data = json.loads(f.read_text())
            if "comparison_text" in data:
                print(data["comparison_text"])
            elif "pass_rate" in data:
                print(f"\n  Pass rate: {data['pass_rate']}%  Avg score: {data['avg_score']}")
            break
