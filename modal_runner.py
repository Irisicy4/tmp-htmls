"""
Parallel Modal runner for evolve_bench_harbor.

Runs tasks in parallel on Modal's serverless infrastructure.
Each task executes in its own isolated container with a fresh sandbox.

Prerequisites
-------------
1. Install Modal:          pip install modal
2. Authenticate:           modal setup
3. Add your OpenAI key:    modal secret create openai-secret OPENAI_API_KEY=sk-...
   For UniAPI/custom endpoint, include base URL and model in the secret:
       modal secret create openai-secret OPENAI_API_KEY=sk-... LLM_BASE_URL=https://api.uniapi.io/v1 LLM_MODEL=claude-sonnet-4-20250514

Usage
-----
# Run all tasks:
    modal run modal_runner.py

# First 3 tasks (smoke test):
    modal run modal_runner.py --first-n 3

# Specific tasks:
    modal run modal_runner.py --task-names task-01-im-looking-for-backpack-under,task-02-help-me-search-xiaohongshu-for

# Override config or output:
    modal run modal_runner.py --config configs/skill-phase1.json --output-dir results/my-run
"""

import json
import os
import sys
import time
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
    # Upload this repo (agent glue, configs, task files)
    .add_local_dir(".", "/harbor-bench", copy=True, ignore=IGNORE_PATTERNS)
    # Copy Harbor glue into cocoa-agent (mirrors task Dockerfile COPY commands)
    .run_commands(
        "cp /harbor-bench/agents/cocoa_agent/run_task.py /cocoa-agent-src/"
        " && cp /harbor-bench/agents/cocoa_agent/overlay.py /cocoa-agent-src/"
        " && cp /harbor-bench/agents/cocoa_agent/agents_overlay/__init__.py /cocoa-agent-src/agents/__init__.py"
        " && cp -r /harbor-bench/agents/cocoa_agent/configs/. /cocoa-agent-src/configs/"
        " && cp -r /harbor-bench/configs/. /cocoa-agent-src/configs/"
    )
    .workdir("/cocoa-agent")
)

# ---------------------------------------------------------------------------
# Modal app & infrastructure
# ---------------------------------------------------------------------------

app = modal.App("evolve-bench-harbor", image=modal_image)

results_volume = modal.Volume.from_name("evolve-bench-results", create_if_missing=True)

# Create with:  modal secret create openai-secret OPENAI_API_KEY=sk-...
_SECRETS = [modal.Secret.from_name("openai-secret")]


# ---------------------------------------------------------------------------
# Helpers (run inside Modal containers)
# ---------------------------------------------------------------------------

def _format_execution_summary(
    result: dict,
    max_chars_per_feedback: int = 2000,
    max_total_chars: int = 80000,
) -> str:
    """Format execution_trace for the LLM judge."""
    trace = result.get("execution_trace", [])
    if not trace:
        return ""
    lines = []
    total_chars = 0
    for i, entry in enumerate(trace, 1):
        action = entry.get("action", {})
        feedback = entry.get("feedback", {})
        atype = action.get("action_type", "unknown")
        param_parts = [
            f"  {k}: {str(v)[:500]}{'...' if len(str(v)) > 500 else ''}"
            for k, v in action.items()
            if k not in ("action_type", "tool_call_id")
        ]
        params_block = "\n".join(param_parts) if param_parts else "  (no parameters)"
        msg = feedback.get("message", "")
        if len(msg) > max_chars_per_feedback:
            msg = msg[:max_chars_per_feedback] + f"... [truncated, {len(msg)} total chars]"
        step_text = f"Step {i}: {atype}\n{params_block}\n-> {msg}\n"
        total_chars += len(step_text)
        if total_chars > max_total_chars:
            lines.append(f"... [trace truncated at step {i}/{len(trace)}]")
            break
        lines.append(step_text)
    return "\n".join(lines)


def _strip_images(result: dict) -> dict:
    """Remove base64 screenshots from conversation to reduce result size."""
    for msg in result.get("conversation", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        part["image_url"]["url"] = "[screenshot stripped]"
    return result


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
def run_single_task(task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single task inside a Modal container.

    1. Starts sandbox services (supervisord → nginx + Chromium).
    2. Waits for sandbox health at localhost:8080.
    3. Runs CocoaAgent with skip_docker (sandbox managed here, not by Docker).
    4. Runs LLM-as-judge verification via the task's test_task.py.
    5. Persists result JSON to Modal Volume and returns it.
    """
    import importlib.util
    import subprocess
    import time as _time

    import requests as _req

    sys.path.insert(0, "/cocoa-agent")

    task_name = task["task_name"]
    instruction = task["instruction"]

    # ---- Start sandbox services ----
    setup = subprocess.run(["/bin/bash", "/opt/gem/run.sh"], check=False)
    if setup.returncode != 0:
        print(f"[warn] Sandbox setup exited {setup.returncode} — continuing anyway")

    # ---- Wait for sandbox ----
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

    # ---- Configure agent ----
    import overlay  # noqa: F401 — applies skip_docker monkey-patch

    from agents import CocoaAgent
    from executor.utils import setup_logging

    setup_logging(config.get("log_level", "INFO"))
    config.setdefault("sandbox", {}).update({"skip_docker": True, "docker_port": 8080})

    # Apply env var overrides to config (supports UniAPI / custom endpoints + models)
    env_base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if env_base_url:
        config.setdefault("controller", {}).setdefault("args", {})["base_url"] = env_base_url
        os.environ["OPENAI_BASE_URL"] = env_base_url  # for LLM judge (test_task.py)
    env_model = os.environ.get("LLM_MODEL")
    if env_model:
        config.setdefault("controller", {}).setdefault("args", {})["model"] = env_model

    max_iter = os.environ.get("COCOA_MAX_ITERATIONS")
    if max_iter:
        try:
            config["sandbox"]["max_iterations"] = int(max_iter)
        except ValueError:
            pass

    agent = CocoaAgent(config)
    task_data = {"task_name": task_name, "instruction": instruction, "task_dir": "/tmp"}

    # ---- Run task ----
    result: Dict[str, Any] = {}
    task_start = _time.time()
    agent.setup_environment(task_data)
    try:
        result = agent.run_task(task_data)
    except Exception as exc:
        partial: Dict[str, Any] = {}
        try:
            partial["conversation"] = agent.executor.controller.get_history()
            partial["execution_trace"] = agent.executor.sandbox_client.get_history()
            partial["execution_summary"] = _format_execution_summary(
                {"execution_trace": partial.get("execution_trace", [])}
            )
        except Exception:
            pass
        try:
            if hasattr(agent.executor.controller, "get_cost_stats"):
                partial["api_cost_stats"] = agent.executor.controller.get_cost_stats()
            partial["iterations"] = getattr(agent.executor, "_current_iteration", 0)
        except Exception:
            pass
        result = {**partial, "status": "error", "error": str(exc), "task_name": task_name}
    finally:
        agent.cleanup_environment()

    # ---- Execution summary for LLM judge ----
    if result.get("execution_trace") and not result.get("execution_summary"):
        result["execution_summary"] = _format_execution_summary(result)

    # ---- Run LLM-as-judge verification ----
    test_file = f"/harbor-bench/tasks/{task_name}/tests/test_task.py"
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

    # ---- Finalize ----
    result = _strip_images(result)
    elapsed = _time.time() - task_start
    result["elapsed_seconds"] = round(elapsed, 1)
    result.setdefault("task_name", task_name)
    print(f"[timing] {task_name} finished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ---- Persist to Modal Volume ----
    out_path = Path(f"/results/{task_name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    return result


# ---------------------------------------------------------------------------
# Local entrypoint — loads tasks, fans out to Modal, collects results
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    config: str = "configs/skill-phase1.json",
    tasks_dir: str = "tasks",
    output_dir: str = "results/modal/",
    first_n: int = 0,
    task_names: str = "",
    experiment_name: str = "",
):
    """Load tasks locally, run them in parallel on Modal, save results.

    --first-n N           Only run the first N tasks (0 = all).
    --task-names a,b,c    Comma-separated task directory names.
    --experiment-name X   Label for this run (defaults to timestamp).
    """
    from datetime import datetime, timezone

    # ---- Load config ----
    config_path = Path(config)
    if not config_path.exists():
        print(f"[error] Config not found: {config}")
        sys.exit(1)
    with open(config_path) as f:
        cfg: Dict[str, Any] = json.load(f)

    # ---- Discover tasks ----
    tasks_path = Path(tasks_dir).resolve()
    tasks: List[Dict[str, Any]] = []

    if (tasks_path / "instruction.md").exists():
        # Single task directory
        candidates = [tasks_path]
    else:
        candidates = sorted(d for d in tasks_path.iterdir() if d.is_dir())

    for task_dir in candidates:
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.exists():
            print(f"  [skip] No instruction.md in {task_dir.name}")
            continue
        tasks.append({
            "task_name": task_dir.name,
            "instruction": instruction_file.read_text().strip(),
        })

    if not tasks:
        print(f"[error] No tasks found in {tasks_dir}")
        sys.exit(1)

    # ---- Filter ----
    if task_names:
        names = [n.strip() for n in task_names.split(",") if n.strip()]
        tasks = [t for t in tasks if t["task_name"] in names]
        print(f"[filter] {len(tasks)} task(s) matching: {names}")
    elif first_n > 0:
        tasks = tasks[:first_n]
        print(f"[filter] First {first_n} task(s)")

    print(f"Loaded {len(tasks)} task(s) from {tasks_dir}")
    for t in tasks:
        print(f"  - {t['task_name']}")

    # ---- Dispatch to Modal ----
    print(f"\nDispatching {len(tasks)} task(s) to Modal...")
    wall_start = time.time()
    pairs = [(task, cfg) for task in tasks]
    results: List[Dict[str, Any]] = list(run_single_task.starmap(pairs))

    # ---- Save results locally ----
    os.makedirs(output_dir, exist_ok=True)
    passed, errors = 0, 0

    for task, result in zip(tasks, results):
        name = task["task_name"]
        out_file = Path(output_dir) / f"{name}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        status = result.get("status", "unknown")
        eval_info = result.get("eval", {}) or {}
        task_passed = eval_info.get("passed", False)
        elapsed_s = result.get("elapsed_seconds")
        elapsed_str = f" [{elapsed_s:.0f}s]" if elapsed_s is not None else ""

        if status == "error":
            errors += 1
            print(f"  x {name}{elapsed_str}: ERROR - {result.get('error', '')[:120]}")
        elif task_passed:
            passed += 1
            score = eval_info.get("details", {}).get("overall_score", "?")
            print(f"  + {name}{elapsed_str}: PASSED (score={score})")
        else:
            score = eval_info.get("details", {}).get("overall_score", "?")
            print(f"  x {name}{elapsed_str}: FAILED (score={score})")

    # ---- Summary ----
    wall_elapsed = time.time() - wall_start
    total = len(tasks)
    failed = total - passed - errors
    task_times = [r.get("elapsed_seconds") for r in results if r.get("elapsed_seconds") is not None]
    avg_task_time = sum(task_times) / len(task_times) if task_times else None

    scores = []
    for result in results:
        eval_info = result.get("eval", {}) or {}
        score = eval_info.get("details", {}).get("overall_score")
        if score is not None:
            scores.append(float(score))
    avg_score = sum(scores) / len(scores) if scores else 0.0
    pass_rate = 100 * passed / total if total > 0 else 0.0

    print(f"\n--- Summary ---")
    print(f"Total:     {total}")
    print(f"Passed:    {passed}")
    print(f"Failed:    {failed}")
    print(f"Errors:    {errors}")
    print(f"Pass rate: {pass_rate:.1f}%")
    print(f"Avg score: {avg_score:.2f}")
    print(f"Wall time: {wall_elapsed / 60:.1f} min ({wall_elapsed:.0f}s)")
    if avg_task_time is not None:
        print(f"Avg task:  {avg_task_time / 60:.1f} min ({avg_task_time:.0f}s)")
    print(f"\nResults:   {output_dir}")
    print(f"Volume:    modal volume ls evolve-bench-results")

    # ---- Experiment metadata ----
    exp_name = experiment_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg_summary = dict(cfg)
    cfg_summary.get("controller", {}).get("args", {}).pop("api_key", None)
    experiment_meta = {
        "experiment_name": exp_name,
        "config": cfg_summary,
        "tasks_dir": tasks_dir,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_tasks": total,
        "summary": {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": round(pass_rate, 1),
            "avg_score": round(avg_score, 2),
        },
    }
    meta_path = Path(output_dir) / "_experiment.json"
    with open(meta_path, "w") as f:
        json.dump(experiment_meta, f, indent=2)
    print(f"Metadata:  {meta_path}")
