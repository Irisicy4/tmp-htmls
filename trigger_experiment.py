"""
Fire-and-forget trigger for the deployed Modal skill experiment.

Requires the app to be deployed first:
  modal deploy standalone_modal_runner.py

Then trigger and close your laptop:
  python trigger_experiment.py --tasks-dir tasks/batch-1
  python trigger_experiment.py --tasks-dir tasks/batch-1 --first-n 3
  python trigger_experiment.py --tasks-dir tasks/batch-1 --phase phase1

Collect results later (no laptop requirement):
  modal run standalone_modal_runner.py --collect <run-tag>
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


def load_env(env_file: Path = Path(".env")):
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


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    args = cfg.setdefault("controller", {}).setdefault("args", {})
    # Inject all credentials/overrides from .env so the container doesn't
    # need a correctly-configured Modal secret for base_url or api_key.
    env_api_key = os.environ.get("OPENAI_API_KEY")
    env_base_url = os.environ.get("LLM_BASE_URL")
    env_model = os.environ.get("LLM_MODEL")
    if env_api_key:
        args["api_key"] = env_api_key
    if env_base_url:
        args["base_url"] = env_base_url
    if env_model:
        args["model"] = env_model
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Trigger a skill experiment on Modal and disconnect.")
    parser.add_argument("--tasks-dir", default="tasks/batch-1", help="Task directory (default: tasks/batch-1)")
    parser.add_argument("--phase", default="skill-experiment",
                        choices=["skill-experiment", "phase1", "phase2"],
                        help="Which phase to run (default: skill-experiment)")
    parser.add_argument("--first-n", type=int, default=0, help="Only run first N tasks")
    parser.add_argument("--task-names", default="", help="Comma-separated task names to run")
    parser.add_argument("--skills-from", default="", help="Run tag with skills for --phase phase2")
    parser.add_argument("--run-tag", default="", help="Custom run tag (default: auto timestamp)")
    args = parser.parse_args()

    load_env()

    phase1_cfg = load_config("configs/skill-phase1.json")
    phase2_cfg = load_config("configs/skill-phase2.json")

    env_base_url = os.environ.get("LLM_BASE_URL")
    env_model = os.environ.get("LLM_MODEL")
    if env_base_url:
        print(f"[.env] LLM_BASE_URL={env_base_url}")
    if env_model:
        print(f"[.env] LLM_MODEL={env_model}")

    tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if args.phase == "skill-experiment":
        fn = modal.Function.from_name("evolve-bench-harbor", "run_skill_experiment")
        handle = fn.spawn(
            phase1_cfg, phase2_cfg, tag,
            tasks_dir=args.tasks_dir,
            first_n=args.first_n,
            task_names_csv=args.task_names,
        )
        print(f"\nDispatched full skill experiment (Phase 1 + Phase 2).")

    elif args.phase == "phase1":
        fn = modal.Function.from_name("evolve-bench-harbor", "run_experiment")
        handle = fn.spawn(
            phase1_cfg, f"{tag}/phase1", "Phase 1",
            tasks_dir=args.tasks_dir,
            first_n=args.first_n,
            task_names_csv=args.task_names,
        )
        print(f"\nDispatched Phase 1.")

    elif args.phase == "phase2":
        if not args.skills_from:
            print("[error] --skills-from <run-tag> is required for phase2")
            sys.exit(1)
        phase2_cfg["skills_dir"] = f"/results/{args.skills_from}/skills"
        fn = modal.Function.from_name("evolve-bench-harbor", "run_experiment")
        handle = fn.spawn(
            phase2_cfg, f"{tag}/phase2", "Phase 2",
            tasks_dir=args.tasks_dir,
            first_n=args.first_n,
            task_names_csv=args.task_names,
        )
        print(f"\nDispatched Phase 2 (skills from {args.skills_from}).")

    print(f"  Run tag:   {tag}")
    print(f"  Tasks dir: {args.tasks_dir}")
    if args.first_n:
        print(f"  First N:   {args.first_n}")
    print(f"\nYour laptop can disconnect now.")
    print(f"\nCollect results when done:")
    print(f"  modal run standalone_modal_runner.py --collect {tag}")


if __name__ == "__main__":
    main()
