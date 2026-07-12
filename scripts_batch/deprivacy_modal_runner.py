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
        "playwright>=1.48",
        "pypdf>=4.0",
    )
    # System deps Chromium needs
    .apt_install(
        "libnss3", "libatk-bridge2.0-0", "libx11-xcb1", "libxcomposite1",
        "libxdamage1", "libxfixes3", "libxrandr2", "libgbm1", "libgtk-3-0",
        "libasound2", "libxkbcommon0", "libpango-1.0-0", "libcairo2",
        "libatspi2.0-0", "fonts-liberation",
    )
    # Install the Chromium browser used by the grounding fetcher
    .run_commands("python -m playwright install chromium")
    # Cache-buster: bump this string to force the tasks/ layer below to
    # re-copy from local (Modal's content-hash on a large nested dir does
    # not always detect edits to deeply-nested task files).
    .run_commands("echo tasks-layer-cachebust-20260713-two-turn")
    .add_local_dir(
        "tasks",
        "/harbor-bench/tasks",
        copy=True,
    )
    # Grounded judging framework — shipped alongside tasks so
    # each task's tests/test_grounded.py can import from it.
    .add_local_dir(
        "agentic_judge",
        "/harbor-bench/agentic_judge",
        copy=True,
        ignore=["__pycache__"],
    )
    # Harness modules for skill extraction/injection (agent-agnostic)
    .add_local_file("harness/skill_extractor.py", "/harness/skill_extractor.py", copy=True)
    .add_local_file("harness/skill_store.py", "/harness/skill_store.py", copy=True)
    .add_local_file("harness/evaluator.py", "/harness/evaluator.py", copy=True)
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


def _load_task_instruction(task_dir: Path) -> str:
    """Prefer instruction.md (the human-edited prompt that may carry
    grounded-judging addenda) over test_task.TASK_INSTRUCTION.  Fall back
    to TASK_INSTRUCTION if instruction.md is missing.

    Heuristic: when instruction.md is at least as long as the AST-extracted
    TASK_INSTRUCTION AND contains the first 40 chars of it (so we know the
    two are about the same task), use instruction.md.  Otherwise stick
    with TASK_INSTRUCTION to avoid surprising other pipelines.
    """
    import ast
    src = (task_dir / "tests" / "test_task.py").read_text()
    ti = None
    tree = ast.parse(src)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "TASK_INSTRUCTION"):
            ti = ast.literal_eval(node.value)
            break
    inst_path = task_dir / "instruction.md"
    if inst_path.is_file():
        inst = inst_path.read_text(encoding="utf-8").rstrip()
        if inst and (ti is None or (len(inst) >= len(ti) and ti.strip()[:40] in inst)):
            return inst
    if ti is not None:
        return ti
    raise RuntimeError(f"neither instruction.md nor TASK_INSTRUCTION found in {task_dir}")


def _discover_tasks(first_n: int = 0, task_names_csv: str = "", tasks_dir: str = "") -> list[dict]:
    """Discover tasks and attach metadata."""
    base = _get_tasks_base(tasks_dir)
    if task_names_csv:
        names = [t.strip() for t in task_names_csv.split(",") if t.strip()]
    else:
        names = sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "tests" / "test_task.py").exists()
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

def _smart_truncate(text: str, cap: int = 60_000) -> str:
    """Cap text at `cap` chars, but preserve any `=== JSON RESULT ===` block
    that lives at the tail so the grader can always parse it.

    If text fits under cap → return as-is.
    Otherwise look for the LAST `=== JSON RESULT ===` ... `=== END JSON ===`
    delimiter pair, keep that intact, and fit as much of the preceding prose
    as the budget allows.
    """
    if not text:
        return ""
    if len(text) <= cap:
        return text
    import re as _re
    # Find the LAST JSON-result block (agents sometimes mention the delim in
    # prose before emitting the real block at the end)
    matches = list(_re.finditer(
        r"=== JSON RESULT ===[\s\S]*?=== END JSON ===", text))
    if not matches:
        # No JSON block in the text — just truncate
        return text[:cap]
    js_start, js_end = matches[-1].span()
    json_block = text[js_start:js_end]
    if len(json_block) >= cap - 100:
        # JSON block alone is huge; keep it
        return json_block
    # Keep a leading prose window + the JSON block, joined with an ellipsis
    head_budget = cap - len(json_block) - 32
    return text[:head_budget] + "\n…[truncated]…\n" + json_block


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
    # Keep the raw stdout (capped) so the grounded judge can read tool
    # calls, fetched-page snippets, intermediate reasoning, and error
    # traces — not just the final agent_message.  Effectiveness trims
    # further at prompt time.
    return {
        "task_result": task_result,
        "conversation": conversation,
        "agent_stdout": stdout[-60_000:] if len(stdout) > 60_000 else stdout,
        "execution_summary": "",
    }


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
    # Keep the raw stdout (capped) so the grounded judge can read tool
    # calls, fetched-page snippets, intermediate reasoning, and error
    # traces — not just the final agent_message.  Effectiveness trims
    # further at prompt time.
    return {
        "task_result": task_result,
        "conversation": conversation,
        "agent_stdout": stdout[-60_000:] if len(stdout) > 60_000 else stdout,
        "execution_summary": "",
    }


def _parse_generic_output(stdout: str) -> dict:
    return {
        "task_result": stdout.strip(),
        "conversation": [],
        "agent_stdout": stdout[-60_000:] if len(stdout) > 60_000 else stdout,
        "execution_summary": "",
    }


def _render_followup_from_spec(task_dir: Path) -> str:
    """Load <task_dir>/tests/test_grounded.py and render the followup-mode
    addendum from its SUMMARY_SCHEMA. Returns "" if the spec or schema
    is missing."""
    spec_path = task_dir / "tests" / "test_grounded.py"
    if not spec_path.is_file():
        return ""
    # The spec does `from agentic_judge.grounded...`; ensure that is
    # importable HERE. This runs during instruction-building, before
    # _evaluate adds the bench root to sys.path — without this, the very
    # first task in a container fails the import, the followup silently
    # returns "", and two-phase/two-turn fall back to single-turn.
    for _p in ("/harbor-bench", str(TASKS_ROOT.parent)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        import importlib.util as _ilu
        mod_name = f"_render_followup_{task_dir.name.replace('-','_')}"
        spec = _ilu.spec_from_file_location(mod_name, spec_path)
        mod = _ilu.module_from_spec(spec)
        sys.modules[mod_name] = mod
        mod.__name__ = mod_name
        spec.loader.exec_module(mod)
        schema = getattr(mod, "SUMMARY_SCHEMA", None)
        if schema is None:
            return ""
        return schema.to_instruction_addendum(mode="followup")
    except Exception as exc:
        print(f"[render_followup] {task_dir.name}: {type(exc).__name__}: {exc}")
        return ""


def _wrap_codex_instruction(instruction: str) -> str:
    return (
        "Complete the following task immediately without asking any clarifying questions. "
        "Make reasonable assumptions and proceed. "
        "For any visual, interactive, or game output, create a single self-contained HTML file "
        "(no external dependencies) rather than a terminal or desktop application. "
        "After creating any files, print each file's complete contents to stdout "
        "wrapped like this: === FILE: <filename> ===\n<contents>\n=== END FILE ===\n\n"
        # JSON-block self-check loop — up to 3 attempts before terminating.
        # The grader hard-penalises missing JSON (-2.0), so this is treated as a
        # blocking exit condition. If the task is infeasible, emit the JSON block
        # with a single `infeasible_reason` key (see instruction.md addendum).
        "BEFORE YOU END YOUR TURN, do this check up to THREE times:\n"
        "  1. Scan your final response.\n"
        "  2. Confirm it contains a JSON block between the EXACT delimiters\n"
        "     === JSON RESULT === and === END JSON === (these literal strings).\n"
        "  3. If the JSON block is missing, append it now and re-check.\n"
        "  4. The JSON must be either a structured answer for the task, OR\n"
        "     `{\"infeasible_reason\": \"<short explanation>\"}` if the task cannot\n"
        "     be completed (site unreachable, login required, data does not exist).\n"
        "Do not end your turn until either the JSON block is present or you have\n"
        "made 3 attempts to add it.\n\n"
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
    task_root = _get_tasks_base(tasks_dir) / task_name
    test_script = task_root / "tests" / "test_task.py"
    grounded_script = task_root / "tests" / "test_grounded.py"
    if not test_script.exists():
        return {"passed": False, "overall_score": 0.0, "feedback": "no evaluator"}

    # Make `import agentic_judge.*` resolvable inside the sandbox.
    # The image bakes agentic_judge/ next to tasks/ at /harbor-bench.
    import sys as _sys
    bench_root = "/harbor-bench"
    if bench_root not in _sys.path:
        _sys.path.insert(0, bench_root)

    # Prefer the grounded judge (LLM-agentic Faithfulness + pending-question
    # loop) when test_grounded.py is present.  Fall back to the v1 LLM-only
    # judge in test_task.py::test otherwise.
    if grounded_script.exists():
        try:
            mod_name = f"test_grounded_{task_name.replace('-', '_')}"
            gspec = importlib.util.spec_from_file_location(
                mod_name, grounded_script)
            gmod = importlib.util.module_from_spec(gspec)
            # IMPORTANT: register before exec so `sys.modules[__name__]`
            # inside grade_with_llm finds the module.
            _sys.modules[mod_name] = gmod
            gmod.__name__ = mod_name
            gspec.loader.exec_module(gmod)
            if hasattr(gmod, "grade_with_llm"):
                out = gmod.grade_with_llm(result)
                # grounded_judge_test returns
                # {passed, feedback, details:{overall_score,...}}
                # — flatten overall_score to top-level for compatibility
                # with downstream tooling that reads result['overall_score'].
                if isinstance(out, dict) and "details" in out:
                    out.setdefault("overall_score",
                                     out["details"].get("overall_score", 0.0))
                    out.setdefault("dimension_scores",
                                     out["details"].get("dimension_scores", {}))
                return out
        except Exception as exc:
            # Surface but fall through to v1 judge so we always get a verdict
            print(f"[grounded judge fell through] {task_name}: "
                  f"{type(exc).__name__}: {exc}")

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


async def _run_codex_turn2(harbor_env, followup: str, extra_env: dict,
                            task_name: str, run_tag: str) -> str:
    """Genuine two-turn: resume the codex session from turn 1 and send the
    followup as a second message. Returns the turn-2 stdout (which carries
    the JSON summary). Codex only — other agents fall back to single-turn.

    The turn-1 session is recorded in $CODEX_HOME/sessions/, so
    `codex exec resume --last` continues that exact conversation with the
    agent's turn-1 answer in context. The chatgpt-mode `codex` wrapper (see
    _inject_proxy_config) self-restores auth.json and allocates a pty on
    each invocation, so no extra auth wiring is needed there; for api mode
    we (re)write auth.json from OPENAI_API_KEY since harbor's turn-1 trap
    deletes it on exit."""
    import shlex as _shlex
    from harbor.models.trial.paths import EnvironmentPaths
    codex_home = EnvironmentPaths.agent_dir.as_posix()
    esc = _shlex.quote(followup)
    auth_mode = (extra_env.get("CODEX_AUTH_MODE") or "").lower()

    if auth_mode == "chatgpt":
        auth_setup = (
            f'if [ -f /opt/codex-chatgpt-auth.json ]; then '
            f'mkdir -p {codex_home!r}; '
            f'cp /opt/codex-chatgpt-auth.json {codex_home + "/auth.json"!r}; '
            f'chmod 600 {codex_home + "/auth.json"!r}; fi; '
        )
    else:
        auth_setup = (
            f'mkdir -p {codex_home!r}; '
            f'printf \'{{"OPENAI_API_KEY": "%s"}}\' "$OPENAI_API_KEY" '
            f'> {codex_home + "/auth.json"!r}; '
        )

    cmd = (
        f'export CODEX_HOME={codex_home!r}; '
        f'{auth_setup}'
        '. ~/.nvm/nvm.sh >/dev/null 2>&1 || true; '
        'codex exec resume --last '
        '--dangerously-bypass-approvals-and-sandbox '
        '--skip-git-repo-check '
        '--json '
        '--enable unified_exec '
        f'-- {esc} 2>&1 </dev/null'
    )
    diag = {"cmd": cmd, "auth_mode": auth_mode, "codex_home": codex_home}
    try:
        # First: does a turn-1 session exist to resume?
        probe = await harbor_env._sandbox.exec.aio(
            "bash", "-lc",
            f'ls -la {codex_home!r}/sessions 2>&1; echo "---"; '
            f'find {codex_home!r}/sessions -name "*.jsonl" 2>&1 | head')
        await probe.wait.aio()
        diag["sessions_probe"] = (await _read_stdout(probe))[:1500]

        res = await harbor_env._sandbox.exec.aio("bash", "-lc", cmd)
        await res.wait.aio()
        out = await _read_stdout(res)
        diag["stdout_len"] = len(out)
        diag["stdout_head"] = out[:2000]
        print(f"[{task_name}] turn-2 (resume) produced {len(out)} chars")
        try:
            _os_p = __import__("pathlib").Path
            dbg = _os_p(str(RESULTS_BASE)) / run_tag / "turn2_debug"
            dbg.mkdir(parents=True, exist_ok=True)
            (dbg / f"{task_name}.json").write_text(__import__("json").dumps(diag, indent=2))
            await results_volume.commit.aio()
        except Exception:
            pass
        return out
    except Exception as e:
        diag["exception"] = f"{type(e).__name__}: {e}"
        print(f"[{task_name}] turn-2 resume failed: {type(e).__name__}: {e}")
        try:
            _os_p = __import__("pathlib").Path
            dbg = _os_p(str(RESULTS_BASE)) / run_tag / "turn2_debug"
            dbg.mkdir(parents=True, exist_ok=True)
            (dbg / f"{task_name}.json").write_text(__import__("json").dumps(diag, indent=2))
            await results_volume.commit.aio()
        except Exception:
            pass
        return ""


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

    if agent_name == "codex":
        from harbor.models.trial.paths import EnvironmentPaths
        codex_home = EnvironmentPaths.agent_dir.as_posix()
        auth_mode = (extra_env.get("CODEX_AUTH_MODE") or "").lower()

        # --- ChatGPT-subscription mode: write auth.json from the Modal Secret
        # and install a codex wrapper that ensures OPENAI_API_KEY can't leak
        # in and shadow the auth.json (harbor's CodexAgent reads
        # OPENAI_API_KEY from os.environ and sets it on the codex subprocess
        # env, which would otherwise force codex into API mode).
        if auth_mode == "chatgpt":
            import os as _os
            auth_blob = _os.environ.get("CODEX_AUTH_JSON", "")
            if not auth_blob:
                print("  [proxy] CODEX_AUTH_MODE=chatgpt but CODEX_AUTH_JSON "
                      "secret is missing — codex will likely fail to authenticate")
            else:
                # 1) Write auth.json to {codex_home}/auth.json
                # 2) Install codex wrapper at /usr/local/bin/codex that unsets
                #    OPENAI_API_KEY/OPENAI_BASE_URL, sets CODEX_HOME, then
                #    exec's the real codex binary.
                # Two writes:
                #   1. /opt/codex-chatgpt-auth.json  — install-time backup the
                #      wrapper restores from at every run.  This is needed
                #      because Harbor's run setup writes its own API-key
                #      auth.json into $CODEX_HOME/auth.json right before the
                #      agent runs (the trap that rm's it on EXIT proves it).
                #   2. $CODEX_HOME/auth.json — initial copy in case codex
                #      gets invoked outside the wrapper.
                # codex is npm-installed via nvm; the parent shell of this
                # exec doesn't have nvm on PATH, so source it before lookup.
                wrapper_install = (
                    "set -e\n"
                    "mkdir -p /opt\n"
                    "cat > /opt/codex-chatgpt-auth.json << 'AUTHJSONEOF'\n"
                    f"{auth_blob}\n"
                    "AUTHJSONEOF\n"
                    "chmod 600 /opt/codex-chatgpt-auth.json\n"
                    f"mkdir -p {codex_home!r}\n"
                    f"cp /opt/codex-chatgpt-auth.json {codex_home + '/auth.json'!r}\n"
                    f"chmod 600 {codex_home + '/auth.json'!r}\n"
                    'export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"\n'
                    '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true\n'
                    'REAL="$(command -v codex 2>/dev/null || true)"\n'
                    'if [ -z "$REAL" ]; then\n'
                    '  for cand in /root/.local/bin/codex /usr/local/bin/codex /usr/bin/codex '
                    '"$NVM_DIR"/versions/node/*/bin/codex; do\n'
                    '    [ -x "$cand" ] && REAL="$cand" && break\n'
                    '  done\n'
                    'fi\n'
                    'if [ -z "$REAL" ]; then\n'
                    '  echo "[proxy] codex binary not found in PATH (searched nvm + /root/.local/bin + /usr/local/bin)"\n'
                    '  exit 0\n'
                    'fi\n'
                    'WRAPPER_DIR="$(dirname "$REAL")"\n'
                    'if [ ! -f "$WRAPPER_DIR/codex.real" ]; then\n'
                    '  mv "$REAL" "$WRAPPER_DIR/codex.real"\n'
                    'fi\n'
                    # Wrapper body in a single-quoted heredoc so $@ etc.
                    # are deferred to runtime.  argv is forwarded verbatim
                    # via pty.spawn (no shell re-parsing — the prompt has
                    # `(no external deps)` parens that break /bin/sh -c).
                    "cat > \"$WRAPPER_DIR/codex\" << 'WRAPEOF'\n"
                    '#!/bin/bash\n'
                    '# chatgpt-mode wrapper for codex.\n'
                    '# 1. Restore chatgpt auth.json from backup (harbor\'s\n'
                    "#    run-setup overwrites $CODEX_HOME/auth.json with an\n"
                    '#    API-key version right before the agent runs).\n'
                    '# 2. Strip API-key envs so codex falls back to auth.json.\n'
                    '# 3. Allocate a pty so isatty(stdin) returns true —\n'
                    '#    codex 0.136 in subscription mode otherwise bails\n'
                    "#    with 'Error: stdin is not a terminal'.\n"
                    'export CODEX_HOME=__CODEX_HOME__\n'
                    'if [ -f /opt/codex-chatgpt-auth.json ]; then\n'
                    '  mkdir -p "$CODEX_HOME"\n'
                    '  cp /opt/codex-chatgpt-auth.json "$CODEX_HOME/auth.json"\n'
                    '  chmod 600 "$CODEX_HOME/auth.json"\n'
                    'fi\n'
                    'unset OPENAI_API_KEY\n'
                    'unset OPENAI_BASE_URL\n'
                    'exec python3 - "__REAL_BIN__" "$@" << \'PYEOF\'\n'
                    'import os, pty, sys\n'
                    'argv = sys.argv[1:]\n'
                    '# Strip flags harbor\'s bundled CodexAgent injects that\n'
                    '# force codex back into API mode:\n'
                    "#   * --model X (every explicit value 400's in chatgpt mode)\n"
                    '#   * -c model_provider="openai-responses"\n'
                    '#   * -c model_providers.openai-responses.*\n'
                    'i, filt = 0, []\n'
                    'while i < len(argv):\n'
                    '    a = argv[i]\n'
                    '    if a == "--model" and i + 1 < len(argv):\n'
                    '        i += 2; continue\n'
                    '    if a == "-c" and i + 1 < len(argv):\n'
                    '        nxt = argv[i+1]\n'
                    '        if nxt.startswith("model_provider") or nxt.startswith("model_providers"):\n'
                    '            i += 2; continue\n'
                    '    filt.append(a); i += 1\n'
                    'rc = pty.spawn(filt)\n'
                    'os.WIFEXITED(rc) and sys.exit(os.WEXITSTATUS(rc))\n'
                    'sys.exit(1)\n'
                    'PYEOF\n'
                    'WRAPEOF\n'
                    # Substitute the two placeholders with real values
                    f"sed -i \"s|__CODEX_HOME__|{codex_home}|g; "
                    "s|__REAL_BIN__|$WRAPPER_DIR/codex.real|g\" "
                    "\"$WRAPPER_DIR/codex\"\n"
                    'chmod +x "$WRAPPER_DIR/codex"\n'
                    'echo "[proxy] codex wrapper installed -> $WRAPPER_DIR/codex (real at $WRAPPER_DIR/codex.real)"\n'
                )
                try:
                    res = await harbor_env._sandbox.exec.aio("bash", "-c", wrapper_install)
                    await res.wait.aio()
                    stdout = await _read_stdout(res)
                    print(f"  [proxy] Wrote codex auth.json -> {codex_home} "
                          f"(ChatGPT subscription mode, {len(auth_blob)} bytes)")
                    if stdout:
                        for line in stdout.splitlines():
                            print(f"  {line}")
                except Exception as e:
                    print(f"  [proxy] Failed to install codex chatgpt wrapper: {e}")

        # --- API-key mode: write config.toml pointing at OPENAI_BASE_URL ---
        elif openai_base:
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

    elif agent_name == "claude-code" and extra_env.get("CLAUDE_AUTH_MODE","").lower() == "subscription":
        # Claude Code subscription auth — same pattern as codex chatgpt subscription.
        # The Modal Secret `claude-code-auth` provides CLAUDE_CODE_CREDENTIALS_JSON
        # (the macOS keychain blob `{"claudeAiOauth": {...}}`). We:
        #   1) Write that JSON to ~/.claude/.credentials.json inside the sandbox
        #      (claude-code CLI reads this file on Linux)
        #   2) Clear ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN
        #      so they don't shadow the OAuth path
        #   3) Use the default Anthropic endpoint (no proxy)
        import os as _os
        creds_blob = _os.environ.get("CLAUDE_CODE_CREDENTIALS_JSON", "")
        if not creds_blob:
            print("  [proxy] CLAUDE_AUTH_MODE=subscription but CLAUDE_CODE_CREDENTIALS_JSON "
                  "is missing — claude will likely fail")
        else:
            # Extract the OAuth accessToken from the keychain blob so we can
            # export it as CLAUDE_CODE_OAUTH_TOKEN (the documented headless-
            # auth env var). Also write the full .credentials.json so claude
            # can refresh the token when it expires.
            import json as _json, shlex as _shlex
            try:
                _parsed = _json.loads(creds_blob)
                access_token = _parsed.get("claudeAiOauth", {}).get("accessToken", "")
            except Exception:
                access_token = ""
            install_cmd = (
                "set -e\n"
                # 1) Write the full credentials.json (both leading-dot and
                #    plain — different claude-code versions look at each)
                "mkdir -p /root/.claude /home/gem/.claude\n"
                "cat > /root/.claude/.credentials.json << 'CREDSEOF'\n"
                f"{creds_blob}\n"
                "CREDSEOF\n"
                "cp /root/.claude/.credentials.json /root/.claude/credentials.json\n"
                "cp /root/.claude/.credentials.json /home/gem/.claude/.credentials.json\n"
                "cp /root/.claude/.credentials.json /home/gem/.claude/credentials.json\n"
                "chmod 600 /root/.claude/*.json /home/gem/.claude/*.json 2>/dev/null || true\n"
                # 2) Install a claude wrapper that exports CLAUDE_CODE_OAUTH_TOKEN
                #    AND unsets the proxy/API env vars before exec'ing the real binary
                "REAL=$(command -v claude 2>/dev/null || true)\n"
                'if [ -z "$REAL" ]; then\n'
                '  for cand in /root/.local/bin/claude /usr/local/bin/claude '
                '"$NVM_DIR"/versions/node/*/bin/claude; do\n'
                '    [ -x "$cand" ] && REAL="$cand" && break\n'
                '  done\n'
                'fi\n'
                'if [ -n "$REAL" ] && [ ! -f "${REAL}.real" ]; then\n'
                '  cp "$REAL" "${REAL}.real"\n'
                "  cat > \"$REAL\" << 'WRAPEOF'\n"
                "#!/bin/bash\n"
                "# Claude subscription wrapper — exports OAuth token, blocks shadowing env vars\n"
                f'export CLAUDE_CODE_OAUTH_TOKEN={_shlex.quote(access_token)}\n'
                'unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL\n'
                'unset ANTHROPIC_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL\n'
                'unset ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL\n'
                'export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"\n'
                '[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true\n'
                'exec __REAL_BIN__ "$@"\n'
                "WRAPEOF\n"
                '  sed -i "s|__REAL_BIN__|${REAL}.real|g" "$REAL"\n'
                '  chmod +x "$REAL"\n'
                '  echo "[proxy] claude wrapper installed -> $REAL (real at $REAL.real)"\n'
                "fi\n"
                f'echo "[proxy] claude subscription creds installed ({len(creds_blob)} bytes, '
                f'token={access_token[:18]}...)"\n'
            )
            try:
                res = await harbor_env._sandbox.exec.aio("bash", "-c", install_cmd)
                await res.wait.aio()
                stdout = await _read_stdout(res)
                print(f"  [proxy] Wrote claude .credentials.json -> /root/.claude/ "
                      f"(subscription mode, {len(creds_blob)} bytes)")
                if stdout:
                    for line in stdout.splitlines():
                        print(f"  {line}")
            except Exception as e:
                print(f"  [proxy] Failed to install claude subscription creds: {e}")

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
    instruction = _load_task_instruction(task_dir).strip()

    # Two-phase prompt mode: strip the inline addendum below the first
    # `---` separator and append a "STEP 2" follow-up that asks the
    # agent to summarise as JSON after answering the brief. The phase-2
    # text is rendered from the task's SUMMARY_SCHEMA so paths stay in
    # sync with the hard-constraint layer. Inline mode keeps the
    # current single-turn behaviour.
    prompt_mode = (agent_kwargs.get("extra_env", {}) or {}).get("PROMPT_MODE", "inline")
    # In genuine two-turn mode the followup is sent as a SEPARATE second
    # message (codex exec resume), so the agent's turn-1 answer is in its
    # conversation history when it produces the JSON. `two_turn_followup`
    # is set here and consumed after agent.run() below.
    two_turn_followup = None
    if prompt_mode == "two-phase":
        brief, _, _addendum = instruction.partition("\n---\n")
        followup = _render_followup_from_spec(task_dir)
        if followup:
            instruction = (
                "STEP 1 — Answer the user's task naturally. Do NOT mention "
                "anything about JSON output, schemas, or grading. Finalise "
                "a complete prose answer before moving to STEP 2.\n\n"
                f"{brief.strip()}\n\n"
                "STEP 2 — After your STEP 1 answer is complete, write a "
                "JSON summary of what you did, without changing the "
                "substance of your STEP 1 answer.\n\n"
                f"{followup}"
            )
            print(f"[{task_name}] PROMPT_MODE=two-phase: brief={len(brief)} chars, "
                  f"followup={len(followup)} chars")
        else:
            print(f"[{task_name}] PROMPT_MODE=two-phase but no SUMMARY_SCHEMA "
                  f"found — falling back to original instruction")
    elif prompt_mode == "two-turn":
        # Turn 1 = the brief alone. Turn 2 = the schema followup, sent as a
        # real second message after the agent answers turn 1.
        brief, _, _addendum = instruction.partition("\n---\n")
        followup = _render_followup_from_spec(task_dir)
        if followup:
            instruction = brief.strip()
            two_turn_followup = followup
            print(f"[{task_name}] PROMPT_MODE=two-turn: turn1(brief)={len(brief)} chars, "
                  f"turn2(followup)={len(followup)} chars")
        else:
            print(f"[{task_name}] PROMPT_MODE=two-turn but no SUMMARY_SCHEMA "
                  f"found — falling back to single-turn brief")

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

        # --- Genuine two-turn: send the schema followup as a second
        # message so the agent's turn-1 answer is in its context. ---
        if two_turn_followup and agent_name == "codex":
            print(f"[{task_name}] Running turn 2 (codex resume) ...")
            turn2_out = await _run_codex_turn2(
                env, two_turn_followup,
                agent_kwargs.get("extra_env", {}) or {}, task_name, run_tag)
            if turn2_out.strip():
                # Persist turn-2 stdout next to turn-1 so the parser can
                # combine them.
                try:
                    (trial_paths.agent_dir / "turn2_stdout.txt").write_text(turn2_out)
                except Exception as _e:
                    print(f"[{task_name}] could not save turn2 stdout: {_e}")
        elif two_turn_followup:
            print(f"[{task_name}] two-turn requested but agent={agent_name} "
                  f"has no resume path — turn-1 answer used as-is")
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

    # --- Two-turn: append the parsed turn-2 output (the schema JSON) LAST,
    # after artifacts, so the extractor's "prefer last JSON block" picks the
    # turn-2 schema summary rather than a turn-1 draft or an artifact copy. ---
    turn2_file = trial_paths.agent_dir / "turn2_stdout.txt"
    if turn2_file.exists():
        turn2_raw = turn2_file.read_text()
        if turn2_raw.strip():
            turn2_parsed = parser(turn2_raw).get("task_result", "") or turn2_raw
            result_dict["task_result"] = (
                (result_dict.get("task_result") or "")
                + "\n\n=== TURN 2 (JSON summary) ===\n"
                + turn2_parsed
            )
            print(f"[{task_name}] Appended turn-2 output LAST ({len(turn2_parsed)} chars)")

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

        # --- Full per-layer judge trace (persisted for audit) ---
        # Every step the judge took, so a curator can see WHY a task
        # scored what it did without re-running the judge:
        #   layer 1  hard_constraint_report  — each predicate + pass/detail
        #   layer 2  faithfulness_report      — per-URL fetch + per-field
        #                                       agent-vs-judge verdicts
        #   layer 3  effectiveness_judge      — rubric dims, reasoning,
        #            effectiveness_judge_prior  pending-question loop
        "judge_trace": {
            "overall_score": details.get("overall_score"),
            "llm_overall_score": details.get("llm_overall_score"),
            "hard_constraint_pass_rate": details.get("hard_constraint_pass_rate"),
            "hard_constraint_report": details.get("hard_constraint_report", []),
            "faithfulness_report": details.get("faithfulness_report", {}),
            "summary_json_parsed": details.get("summary_json_parsed"),
            "summary_json_source": details.get("summary_json_source"),
            "summary_json": details.get("summary_json"),
            "effectiveness_judge": details.get("effectiveness_judge", {}),
            "effectiveness_judge_prior": details.get("effectiveness_judge_prior"),
            "pending_faith_questions_answered": details.get("pending_faith_questions_answered", []),
            "score_components": details.get("score_components", {}),
        },

        # --- Timing ---
        "started_at": started_at,
        "finished_at": finished_at,

        # --- Agent output (capped for storage, full version in trials/) ---
        # Cap is high enough to fit a long analysis + JSON tail; if the agent
        # writes more, _smart_truncate preserves the `=== JSON RESULT ===`
        # block from the tail so the grader can still parse it.
        "task_result": _smart_truncate(task_result, cap=60_000),

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
    # Throttled to 2 concurrent containers — Anthropic claude-code
    # subscription throttles parallel bursts ("Server is temporarily limiting
    # requests · Rate limited" 429s), so we cap concurrency here. For codex
    # the rate-limit is per-account-per-day, so concurrency cap doesn't help
    # there, but 2 is still enough to make codex runs reasonable.
    max_containers=2,
    secrets=[
        modal.Secret.from_name("codex-chatgpt-auth"),
        modal.Secret.from_name("claude-code-auth"),
    ],
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

