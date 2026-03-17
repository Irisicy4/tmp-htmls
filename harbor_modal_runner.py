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
    .add_local_dir("tasks/batch-1", "/harbor-bench/tasks/batch-1", copy=True)
)

# ── Constants ─────────────────────────────────────────────────────────────────
TASKS_BASE = Path("/harbor-bench/tasks/batch-1")
RESULTS_BASE = Path("/results")

# Base sandbox image: browser + Python, no agent.  Harbor installs the agent.
SANDBOX_BASE_IMAGE = "ghcr.io/agent-infra/sandbox:latest"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_tasks(tasks_dir: str, first_n: int, task_names_csv: str) -> list[str]:
    """Return ordered task names from TASKS_BASE."""
    if task_names_csv:
        return [t.strip() for t in task_names_csv.split(",") if t.strip()]
    tasks = sorted(
        d.name
        for d in TASKS_BASE.iterdir()
        if d.is_dir() and (d / "instruction.md").exists()
    )
    return tasks[:first_n] if first_n > 0 else tasks


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
        "For any visual, interactive, or game output, create a single self-contained HTML file "
        "(no external dependencies) rather than a terminal or desktop application. "
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


def _evaluate(task_name: str, result: dict, run_tag: str) -> dict:
    """
    Load test_task.py for the given task and call its test() function.
    Returns the full eval_result dict.
    """
    test_script = TASKS_BASE / task_name / "tests" / "test_task.py"
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


# ── PatchedModalEnvironment ────────────────────────────────────────────────────
# Defined as a factory so it's constructed inside the Modal container where
# the harbor SDK is available.

def _make_harbor_env(session_id: str, trial_paths):
    """
    Returns a ModalEnvironment subclass instance that:
    1. Uses ghcr.io/agent-infra/sandbox:latest instead of a per-task Dockerfile.
    2. Uses session_id as the Modal app name to avoid the "__harbor__" locking bug.
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
) -> dict:
    """
    Run one task end-to-end:
    1. Create a Harbor-managed Modal Sandbox with the base sandbox image.
    2. Harbor installs the agent (claude-code, codex, …) in the sandbox.
    3. Agent runs; Harbor captures stdout to local (volume-mounted) logs.
    4. Parse stdout, evaluate with test_task.py, return result dict.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.context import AgentContext
    from harbor.models.agent.name import AgentName
    from harbor.models.trial.paths import TrialPaths

    task_dir = TASKS_BASE / task_name
    instruction = (task_dir / "instruction.md").read_text().strip()

    # For codex: prevent clarifying questions, request file contents in stdout
    if agent_name == "codex":
        instruction = _wrap_codex_instruction(instruction)

    trial_dir = RESULTS_BASE / run_tag / "trials" / task_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    # Modal app names: max 64 chars, alphanumeric + hyphens
    session_id = f"hm-{run_tag}-{task_name}"[:63].rstrip("-")

    env = _make_harbor_env(session_id, trial_paths)

    # Expose API keys to the evaluator (test_task.py) which runs in this process.
    # extra_env is only forwarded to the sandbox by Harbor, not to this container.
    for k, v in agent_kwargs.get("extra_env", {}).items():
        os.environ.setdefault(k, v)

    agent = AgentFactory.create_agent_from_name(
        AgentName(agent_name),
        logs_dir=trial_paths.agent_dir,
        **agent_kwargs,
    )

    context = AgentContext()

    print(f"[{task_name}] Starting sandbox …")
    try:
        await env.start(force_build=False)
    except Exception as exc:
        print(f"[{task_name}] Sandbox start failed: {exc}")
        return {"task": task_name, "score": 0.0, "error": str(exc), "run_tag": run_tag}

    try:
        print(f"[{task_name}] Setting up agent ({agent_name}) …")
        await agent.setup(env)

        print(f"[{task_name}] Running agent …")
        await agent.run(instruction, env, context)
    except Exception as exc:
        print(f"[{task_name}] Agent error: {exc}")
    finally:
        # Collect any files the agent wrote to the sandbox home directory
        try:
            result_obj = await env._sandbox.exec.aio(
                "bash", "-c",
                r"find /root /home -maxdepth 4 \( -name '*.html' -o -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.sh' -o -name '*.json' -o -name '*.txt' \) 2>/dev/null | head -20"
            )
            await result_obj.wait.aio()
            file_list = (await result_obj.stdout.read.aio()).decode(errors="replace").strip()
            if file_list:
                for fpath in file_list.splitlines():
                    fpath = fpath.strip()
                    if not fpath:
                        continue
                    try:
                        cat_obj = await env._sandbox.exec.aio("cat", fpath)
                        await cat_obj.wait.aio()
                        content = (await cat_obj.stdout.read.aio()).decode(errors="replace")
                        fname = fpath.split("/")[-1]
                        (trial_paths.agent_dir / f"artifact_{fname}").write_text(content)
                        print(f"[{task_name}] Collected artifact: {fpath} ({len(content)} chars)")
                    except Exception:
                        pass
        except Exception as collect_exc:
            print(f"[{task_name}] Artifact collection skipped: {collect_exc}")
        try:
            await env.stop(delete=True)
        except Exception:
            pass

    # ── Evaluate ──────────────────────────────────────────────────────────────
    # Harbor captured each exec's stdout to trial_paths.agent_dir/command-{i}/stdout.txt
    # For most agents, command-1 is the actual agent run (command-0 is setup).
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

    eval_result = _evaluate(task_name, result_dict, run_tag)

    score = float(eval_result.get("details", {}).get("overall_score", 0)
                  or eval_result.get("overall_score", 0))
    passed = bool(eval_result.get("passed", False))

    summary = {
        "task": task_name,
        "agent": agent_name,
        "run_tag": run_tag,
        "score": score,
        "passed": passed,
        "feedback": eval_result.get("feedback", ""),
    }
    (trial_dir / "result.json").write_text(json.dumps(summary, indent=2))

    print(f"[{task_name}] Done — score={score:.2f} passed={passed}")
    return summary


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
) -> dict:
    """Run a single task with a Harbor agent. Used by run_harbor_experiment via starmap."""
    return await _run_task(task_name, agent_name, agent_kwargs, run_tag)


@app.function(
    image=harbor_image,
    timeout=14400,
    volumes={str(RESULTS_BASE): results_volume},
)
async def run_harbor_experiment(
    agent_name: str,
    agent_kwargs: dict,
    run_tag: str,
    tasks_dir: str = "tasks/batch-1",
    first_n: int = 0,
    task_names_csv: str = "",
) -> dict:
    """
    Orchestrate a full experiment run.
    Dispatches one run_harbor_task per task (parallel via starmap).
    """
    tasks = _discover_tasks(tasks_dir, first_n, task_names_csv)

    print(f"\n=== Harbor Experiment: {run_tag} ===")
    print(f"Agent : {agent_name}")
    print(f"Tasks : {len(tasks)}")
    if agent_kwargs:
        print(f"Kwargs: {agent_kwargs}")

    results = []
    async for result in run_harbor_task.starmap.aio(
        [(t, agent_name, agent_kwargs, run_tag) for t in tasks]
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
    summary = {
        "run_tag": run_tag,
        "agent": agent_name,
        "agent_kwargs": agent_kwargs,
        "tasks_dir": tasks_dir,
        "n_tasks": len(results),
        "n_passed": sum(1 for r in results if r.get("passed")),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "results": results,
    }

    out_path = RESULTS_BASE / run_tag / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    await results_volume.commit.aio()

    print(f"\n=== Experiment Complete ===")
    print(f"Mean score : {summary['mean_score']:.3f}")
    print(f"Pass rate  : {summary['n_passed']}/{summary['n_tasks']}")
    return summary


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(collect: str = "", list_runs: bool = False):
    """
    modal run harbor_modal_runner.py --collect <run-tag>
    modal run harbor_modal_runner.py --list-runs
    """
    modal_bin = os.path.expanduser("~/.local/bin/modal")

    if list_runs:
        subprocess.run([modal_bin, "volume", "ls", "evolve-bench-results"])
        return

    if collect:
        out_dir = Path(f"results/harbor/{collect}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {collect} → {out_dir}")
        subprocess.run(
            [modal_bin, "volume", "get", "evolve-bench-results", collect, str(out_dir)],
            check=True,
        )
        print(f"Done. Results in {out_dir}")
        return

    print("Use trigger_harbor.py to dispatch experiments.")
    print("  python trigger_harbor.py --agent claude-code --tasks-dir tasks/batch-1")
