"""Main entry point for the agentic judge."""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

from agentic_judge.core.verifier import verify_task, _parse_domain
from agentic_judge.core.result_writer import write_result


def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml.

    Search order:
    1. Explicitly provided path
    2. agentic_judge/config.yaml next to this file (works in worktrees)
    3. agentic_judge/config.yaml in repo root
    """
    if config_path is None:
        local = Path(__file__).parent / "config.yaml"
        config_path = local if local.exists() else Path(__file__).parent.parent / "agentic_judge" / "config.yaml"  # assumes run from repo root
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_passed_results(result_dir: str) -> list[dict]:
    """Return list of {path, data} dicts for all passing result JSONs.

    Expects result_dir to contain task subdirectories each with a result.json
    (e.g., result_dir/task-01-xxx/result.json).
    """
    results = []
    for json_path in sorted(Path(result_dir).glob("*/result.json")):
        with open(json_path) as f:
            data = json.load(f)
        if data.get("passed"):
            results.append({"path": str(json_path), "data": data})
    return results


def _get_task_domain(task_dir: str, task_name: str) -> str:
    """Read category from task.toml for a given task_name subdirectory."""
    task_toml_path = Path(task_dir) / task_name / "task.toml"
    if not task_toml_path.exists():
        return "Unknown"
    return _parse_domain(task_toml_path.read_text())


def stratified_sample(results: list[dict], task_dir: str, n: int) -> list[dict]:
    """Sample n tasks stratified by domain, proportional allocation."""
    for r in results:
        task_name = r["data"].get("task", r["data"].get("task_name", Path(r["path"]).parent.name))
        r["task_name"] = task_name
        if "domain" not in r:
            r["domain"] = _get_task_domain(task_dir, task_name) if task_dir else "Unknown"

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_domain[r["domain"]].append(r)

    total = len(results)
    sampled: list[dict] = []
    remainder: list[dict] = []

    for domain, items in sorted(by_domain.items()):
        alloc = max(1, round(len(items) / total * n))
        shuffled = list(items)
        random.shuffle(shuffled)
        sampled.extend(shuffled[:alloc])
        remainder.extend(shuffled[alloc:])

    random.shuffle(remainder)
    while len(sampled) < n and remainder:
        sampled.append(remainder.pop())

    return sampled[:n]


def _resolve_task_dir(config: dict, dataset: str, task_name: str) -> str:
    """Return absolute path to a task's directory."""
    repo_root = Path(__file__).parent.parent
    task_base = repo_root / config["task_dirs"][dataset]
    return str(task_base / task_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic judge verification")
    parser.add_argument(
        "--dataset",
        choices=["original", "synthetic", "both", "real118"],
        default="both",
        help="Which dataset to verify",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Override result dir from config (path to trials directory with task subdirs)",
    )
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Override task dir from config (path to tasks root directory)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of tasks to sample (default from config)",
    )
    parser.add_argument(
        "--task-names",
        nargs="+",
        default=None,
        help="Verify specific tasks by name instead of sampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which tasks would be verified; do not run",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: agentic_judge/config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(__file__).parent.parent  # assumes run from repo root
    sample_size = args.sample_size or config["sampling"]["total_sample_size"]

    if args.dataset == "both":
        datasets = ["original", "synthetic"]
    else:
        datasets = [args.dataset]

    all_tasks: list[dict] = []
    for dataset in datasets:
        if args.result_dir and len(datasets) == 1:
            result_dir = args.result_dir
        else:
            result_dir = str(repo_root / config["result_dirs"][dataset])
        passed = load_passed_results(result_dir)
        if args.task_dir and len(datasets) == 1:
            task_base = args.task_dir
        else:
            task_base = str(repo_root / config["task_dirs"][dataset])
        for r in passed:
            task_name = r["data"].get("task", r["data"].get("task_name", Path(r["path"]).parent.name))
            r["task_name"] = task_name
            r["domain"] = _get_task_domain(task_base, task_name)
            r["dataset"] = dataset
        all_tasks.extend(passed)

    if args.task_names:
        tasks_to_run = [t for t in all_tasks if t["task_name"] in args.task_names]
        if not tasks_to_run:
            print(f"No matching tasks found for: {args.task_names}", file=sys.stderr)
            sys.exit(1)
    else:
        tasks_to_run = stratified_sample(all_tasks, "", sample_size)

    if args.dry_run:
        print(f"\nDry run — would verify {len(tasks_to_run)} tasks:\n")
        for t in tasks_to_run:
            print(f"  [{t['dataset']}] {t['task_name']} (domain: {t['domain']})")
        return

    results_summary = []
    for t in tasks_to_run:
        if args.task_dir and len(datasets) == 1:
            task_dir = str(Path(args.task_dir) / t["task_name"])
        else:
            task_dir = _resolve_task_dir(config, t["dataset"], t["task_name"])
        output_dir = str(repo_root / config["verification_output"][t["dataset"]])
        print(f"Verifying [{t['dataset']}] {t['task_name']} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            verification = verify_task(
                result_json_path=t["path"],
                task_dir=task_dir,
                model=config["agent"]["model"],
                timeout_seconds=config["agent"]["timeout_seconds"],
            )
            verification["dataset"] = t["dataset"]
            write_result(verification, output_dir)
            elapsed = round(time.time() - t0, 1)
            status = (
                "verified" if verification["agentic_verified"] is True
                else "NOT verified" if verification["agentic_verified"] is False
                else "unverifiable"
            )
            print(f"{status} ({elapsed}s)")
            results_summary.append(verification)
        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"\n{'Task':<50} {'Category':<20} {'Static':>7} {'Agentic':>10} {'Method':<20}")
    print("-" * 115)
    for r in results_summary:
        static = "PASS" if r["static_judge_passed"] else "FAIL"
        agentic = (
            "VERIFIED" if r["agentic_verified"] is True
            else "FAILED" if r["agentic_verified"] is False
            else "NULL"
        )
        print(
            f"{r['task_name']:<50} {r['category']:<20} {static:>7} {agentic:>10} "
            f"{r['verification_method']:<20}"
        )
    print(f"\nTotal: {len(results_summary)} tasks processed")


if __name__ == "__main__":
    main()
