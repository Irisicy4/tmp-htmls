"""
Fire-and-forget trigger for the Harbor-inside-Modal experiment runner.

Requires the app to be deployed first:
  modal deploy harbor_modal_runner.py

Then trigger and close your laptop:
  python trigger_harbor.py --agent claude-code
  python trigger_harbor.py --agent claude-code --first-n 3
  python trigger_harbor.py --agent codex --task-names task-01-...,task-05-...

Collect results later (no laptop requirement):
  modal run harbor_modal_runner.py --collect <run-tag>

Supported agents (Harbor agent names):
  claude-code   — Anthropic Claude Code CLI  (needs ANTHROPIC_API_KEY)
  codex         — OpenAI Codex CLI           (needs OPENAI_API_KEY)
  gemini-cli    — Google Gemini CLI          (needs GEMINI_API_KEY)
  aider         — Aider                      (needs OPENAI_API_KEY or ANTHROPIC_API_KEY)
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


# ── Env / credential loading ──────────────────────────────────────────────────

def _load_env(env_file: Path = Path(".env")):
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and val and key not in os.environ:
            os.environ[key] = val


def _detect_uniapi() -> tuple[str, str]:
    """Return (key, base_url) if UniAPI proxy is configured, else ("", "")."""
    key = os.environ.get("UNIAPI_KEY", "")
    base = os.environ.get("UNIAPI_BASE", "")
    if key and not base:
        print("  [warn] UNIAPI_KEY is set but UNIAPI_BASE is missing — UniAPI disabled")
    elif base and not key:
        print("  [warn] UNIAPI_BASE is set but UNIAPI_KEY is missing — UniAPI disabled")
    if key and base:
        return key, base
    return "", ""


_DEFAULT_MODELS = {
    "claude-code": "anthropic/claude-sonnet-4-20250514",
    "codex": "openai/gpt-4.1-mini",
    "gemini-cli": "google/gemini-2.5-flash",
    "aider": "openai/gpt-4.1-mini",
}


def _build_agent_kwargs(agent_name: str, model: str) -> dict:
    """
    Build the agent-specific kwargs dict that harbor_modal_runner passes to
    AgentFactory.create_agent_from_name(..., **agent_kwargs).

    If UNIAPI_KEY is set, automatically derives per-provider keys and base URLs.
    Otherwise falls back to provider-specific env vars (OPENAI_API_KEY, etc.).
    """
    extra_env: dict[str, str] = {}
    uniapi_key, uniapi_base = _detect_uniapi()

    if agent_name == "claude-code":
        key = os.environ.get("ANTHROPIC_API_KEY", "") or uniapi_key
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        if not base_url and uniapi_base:
            base_url = f"{uniapi_base}/claude"
        if key:
            extra_env["ANTHROPIC_API_KEY"] = key
        if base_url:
            extra_env["ANTHROPIC_BASE_URL"] = base_url
            print(f"  [proxy] claude-code -> {base_url}")

    elif agent_name in ("codex", "aider"):
        key = os.environ.get("OPENAI_API_KEY", "") or uniapi_key
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if not base_url and uniapi_base:
            base_url = f"{uniapi_base}/v1"
        if key:
            extra_env["OPENAI_API_KEY"] = key
        if base_url:
            extra_env["OPENAI_BASE_URL"] = base_url
            print(f"  [proxy] {agent_name} -> {base_url}")

    elif agent_name == "gemini-cli":
        key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "") or uniapi_key
        if key:
            extra_env["GEMINI_API_KEY"] = key
        if uniapi_base:
            extra_env["GEMINI_API_BASE_URL"] = f"{uniapi_base}/gemini"
            print(f"  [proxy] gemini-cli -> {uniapi_base}/gemini")

    if agent_name not in ("claude-code", "codex", "aider", "gemini-cli"):
        print(f"  [warn] No credential mapping for agent '{agent_name}' — extra_env will be empty")

    if not model:
        model = _DEFAULT_MODELS.get(agent_name, "")

    # --- Judge credentials (separate from agent) ---
    # Judge (test_task.py) uses openai.OpenAI(), so it always needs
    # OPENAI_API_KEY + optionally OPENAI_BASE_URL.
    judge_key = (
        os.environ.get("JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or uniapi_key
    )
    judge_base = (
        os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or (f"{uniapi_base}/v1" if uniapi_base else "")
    )
    judge_env: dict[str, str] = {}
    if judge_key:
        judge_env["OPENAI_API_KEY"] = judge_key
    if judge_base:
        judge_env["OPENAI_BASE_URL"] = judge_base

    kwargs: dict = {}
    if extra_env:
        kwargs["extra_env"] = extra_env
    if model:
        kwargs["model_name"] = model
    kwargs["judge_env"] = judge_env

    return kwargs


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trigger a Harbor-inside-Modal experiment and disconnect."
    )
    parser.add_argument(
        "--agent",
        default="claude-code",
        choices=["claude-code", "codex", "gemini-cli", "aider", "opencode"],
        help="Harbor agent name (default: claude-code)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name override (e.g. claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--tasks-dir",
        default="tasks/updated-deprivacy-100",
        help="Task directory relative to repo root (default: tasks/updated-deprivacy-100)",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=0,
        help="Run only the first N tasks (0 = all)",
    )
    parser.add_argument(
        "--task-names",
        default="",
        help="Comma-separated task names to run (overrides --first-n)",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Custom run tag (default: auto timestamp)",
    )
    parser.add_argument(
        "--phase",
        default="single",
        choices=["single", "skill-experiment"],
        help="single = one pass; skill-experiment = Phase 1 + Phase 2 (default: single)",
    )
    args = parser.parse_args()

    _load_env()

    tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    agent_kwargs = _build_agent_kwargs(args.agent, args.model)

    # Validate credentials before dispatching
    if not agent_kwargs.get("extra_env"):
        print(f"\n[ERROR] No API credentials resolved for agent '{args.agent}'.")
        print("  Set the appropriate env vars (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, or UNIAPI_KEY + UNIAPI_BASE).")
        sys.exit(1)
    if not agent_kwargs.get("judge_env", {}).get("OPENAI_API_KEY"):
        print(f"\n[ERROR] No judge credentials resolved.")
        print("  The LLM judge requires an OpenAI-compatible API key.")
        print("  Set JUDGE_API_KEY, OPENAI_API_KEY, or UNIAPI_KEY + UNIAPI_BASE.")
        sys.exit(1)

    if args.phase == "skill-experiment":
        fn = modal.Function.from_name("evolve-bench-harbor", "run_harbor_skill_experiment")
        fn.spawn(
            args.agent,
            agent_kwargs,
            tag,
            tasks_dir=args.tasks_dir,
            first_n=args.first_n,
            task_names_csv=args.task_names,
        )
        print(f"\nDispatched Harbor skill experiment (Phase 1 + Phase 2).")
    else:
        fn = modal.Function.from_name("evolve-bench-harbor", "run_harbor_experiment")
        fn.spawn(
            args.agent,
            agent_kwargs,
            tag,
            tasks_dir=args.tasks_dir,
            first_n=args.first_n,
            task_names_csv=args.task_names,
        )
        print(f"\nDispatched Harbor experiment.")

    print(f"  Agent     : {args.agent}")
    print(f"  Phase     : {args.phase}")
    print(f"  Run tag   : {tag}")
    print(f"  Tasks dir : {args.tasks_dir}")
    if args.first_n:
        print(f"  First N   : {args.first_n}")
    if args.task_names:
        print(f"  Tasks     : {args.task_names}")
    print(f"\nYour laptop can disconnect now.")
    print(f"\nCollect results when done:")
    print(f"  modal run harbor_modal_runner.py --collect {tag}")


if __name__ == "__main__":
    main()
