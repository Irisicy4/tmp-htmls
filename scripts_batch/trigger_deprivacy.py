"""
Fire-and-forget trigger for deprivacy-100 experiments.

Runs ONE agent at a time. To run all agents, use run_all_agents.sh.

Usage:
  # Single agent:
  python scripts_batch/trigger_deprivacy.py --agent claude-code
  python scripts_batch/trigger_deprivacy.py --agent codex
  python scripts_batch/trigger_deprivacy.py --agent gemini-cli
  python scripts_batch/trigger_deprivacy.py --agent aider

  # Quick smoke test (first 3 tasks):
  python scripts_batch/trigger_deprivacy.py --agent claude-code --first-n 3

  # Custom model override:
  python scripts_batch/trigger_deprivacy.py --agent claude-code --model claude-opus-4-5-20251101

Requires:
  modal deploy scripts_batch/deprivacy_modal_runner.py
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


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
    """Detect if user is using UniAPI and return (api_key, base_domain).

    UniAPI supports native protocols for all providers:
      - OpenAI:  https://api.uniapi.io/v1
      - Claude:  https://api.uniapi.io/claude
      - Gemini:  https://api.uniapi.io/gemini
    """
    key = os.environ.get("UNIAPI_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("LLM_BASE_URL", "")
    if "uniapi" in base or os.environ.get("UNIAPI_KEY", ""):
        return key, "https://api.uniapi.io"
    return "", ""


# Default models for each agent when using UniAPI
_DEFAULT_MODELS = {
    "claude-code": "claude-sonnet-4-20250514",
    "codex": "gpt-4.1-mini",
    "gemini-cli": "gemini-2.5-flash",
    "aider": "gpt-4.1-mini",
}


def _build_agent_kwargs(agent_name: str, model: str) -> dict:
    extra_env: dict[str, str] = {}
    uniapi_key, uniapi_base = _detect_uniapi()

    if agent_name == "claude-code":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        # UniAPI: use its native Claude endpoint
        if not key and uniapi_key:
            key = uniapi_key
            base_url = base_url or f"{uniapi_base}/claude"
        if base_url:
            # Proxy mode: pass AUTH_TOKEN for the settings.local.json injection,
            # and also set ANTHROPIC_API_KEY so Harbor doesn't complain (it gets
            # overridden by settings.local.json inside the sandbox)
            extra_env["ANTHROPIC_AUTH_TOKEN"] = key
            extra_env["ANTHROPIC_API_KEY"] = key  # Harbor needs this to not error
            extra_env["ANTHROPIC_BASE_URL"] = base_url
            # Pass model name to proxy config injector
            resolved_model = model or os.environ.get("LLM_MODEL", "") or _DEFAULT_MODELS.get("claude-code", "")
            extra_env["_CLAUDE_MODEL"] = resolved_model
            print(f"  [proxy] claude-code -> {base_url} (model: {resolved_model})")
        elif key:
            extra_env["ANTHROPIC_API_KEY"] = key

    elif agent_name in ("codex", "aider"):
        key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("LLM_BASE_URL", "")
        if uniapi_key and not base_url:
            base_url = f"{uniapi_base}/v1"
        if uniapi_key and not key:
            key = uniapi_key
        if key:
            extra_env["OPENAI_API_KEY"] = key
        if base_url:
            extra_env["OPENAI_BASE_URL"] = base_url
            print(f"  [proxy] {agent_name} -> {base_url}")

    elif agent_name == "gemini-cli":
        key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if not key and uniapi_key:
            key = uniapi_key
            print(f"  [uniapi] gemini-cli -> {uniapi_base}/gemini")
        if key:
            extra_env["GEMINI_API_KEY"] = key
        if uniapi_base:
            extra_env["GEMINI_API_BASE_URL"] = f"{uniapi_base}/gemini"

    # Model: use explicit --model, or env LLM_MODEL, or default for agent
    if not model:
        model = os.environ.get("LLM_MODEL", "") or _DEFAULT_MODELS.get(agent_name, "")

    # Always pass OPENAI_API_KEY + LLM_BASE_URL for the LLM judge,
    # regardless of which agent is running
    judge_key = os.environ.get("OPENAI_API_KEY", "")
    judge_base = os.environ.get("LLM_BASE_URL", "")
    if judge_key:
        extra_env.setdefault("OPENAI_API_KEY", judge_key)
    if judge_base:
        extra_env.setdefault("OPENAI_BASE_URL", judge_base)

    kwargs: dict = {}
    if extra_env:
        kwargs["extra_env"] = extra_env
    if model:
        kwargs["model_name"] = model
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Trigger a deprivacy-100 experiment on Modal (fire-and-forget)."
    )
    parser.add_argument(
        "--agent", required=True,
        choices=["claude-code", "codex", "gemini-cli", "aider", "opencode"],
    )
    parser.add_argument("--model", default="", help="Model name override")
    parser.add_argument("--first-n", type=int, default=0, help="Run only first N tasks (0 = all 100)")
    parser.add_argument("--task-names", default="", help="Comma-separated task names")
    parser.add_argument("--run-tag", default="", help="Custom run tag (default: auto)")
    args = parser.parse_args()

    _load_env()

    # Tag format: deprivacy/<agent>/<timestamp>
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = args.run_tag or f"deprivacy/{args.agent}/{ts}"
    agent_kwargs = _build_agent_kwargs(args.agent, args.model)

    fn = modal.Function.from_name("deprivacy-100-bench", "run_deprivacy_experiment")
    handle = fn.spawn(
        args.agent,
        agent_kwargs,
        tag,
        first_n=args.first_n,
        task_names_csv=args.task_names,
    )

    print(f"\nDispatched deprivacy-100 experiment.")
    print(f"  Agent     : {args.agent}")
    print(f"  Model     : {args.model or '(default)'}")
    print(f"  Run tag   : {tag}")
    if args.task_names:
        print(f"  Tasks     : {args.task_names}")
    elif args.first_n:
        print(f"  Tasks     : first {args.first_n}")
    else:
        print(f"  Tasks     : all 100")
    print(f"\nYour laptop can disconnect now.")
    print(f"\nCollect results when done:")
    print(f"  modal run scripts_batch/deprivacy_modal_runner.py --collect {tag}")


if __name__ == "__main__":
    main()
