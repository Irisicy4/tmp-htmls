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


def _build_agent_kwargs(agent_name: str, model: str) -> dict:
    """
    Build the agent-specific kwargs dict that harbor_modal_runner passes to
    AgentFactory.create_agent_from_name(..., **agent_kwargs).

    Credentials are injected here so the Modal container picks them up via
    extra_env rather than requiring a correctly configured Modal Secret.
    """
    extra_env: dict[str, str] = {}

    if agent_name == "claude-code":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        if key:
            extra_env["ANTHROPIC_API_KEY"] = key
        if base_url:
            extra_env["ANTHROPIC_BASE_URL"] = base_url

    elif agent_name in ("codex", "aider"):
        key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("LLM_BASE_URL", "")
        if key:
            extra_env["OPENAI_API_KEY"] = key
        if base_url:
            extra_env["OPENAI_BASE_URL"] = base_url

    elif agent_name == "gemini-cli":
        key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if key:
            extra_env["GEMINI_API_KEY"] = key

    # Fall back to LLM_MODEL from .env when no explicit --model given
    if not model:
        model = os.environ.get("LLM_MODEL", "")

    kwargs: dict = {}
    if extra_env:
        kwargs["extra_env"] = extra_env
    if model:
        kwargs["model_name"] = model

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
        default="tasks/batch-1",
        help="Task directory relative to repo root (default: tasks/batch-1)",
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
    args = parser.parse_args()

    _load_env()

    tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    agent_kwargs = _build_agent_kwargs(args.agent, args.model)

    fn = modal.Function.from_name("evolve-bench-harbor", "run_harbor_experiment")
    handle = fn.spawn(
        args.agent,
        agent_kwargs,
        tag,
        tasks_dir=args.tasks_dir,
        first_n=args.first_n,
        task_names_csv=args.task_names,
    )

    print(f"\nDispatched Harbor experiment.")
    print(f"  Agent     : {args.agent}")
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
