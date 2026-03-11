#!/usr/bin/env python3
"""
Extract skills from Harbor experiment results (Phase 1 post-processing).

Reads agent result.json + verifier reward.json from a Harbor output directory,
runs skill extraction on tasks that scored above the threshold, and saves
skill .md files to the output directory.

Usage:
    python scripts/extract_skills.py results/modal/<run-dir>
    python scripts/extract_skills.py results/modal/<run-dir> --output skills/ --threshold 2.0
"""

import argparse
import json
import sys
from pathlib import Path

# Add harness to path so we can import skill utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "harness"))

from skill_extractor import SkillExtractor
from skill_store import SkillStore


def find_tasks(results_dir: Path) -> list[dict]:
    """Find all task results in a Harbor output directory.

    Harbor structure: <run>/task-name__hash/agent/result.json + verifier/reward.json
    Also supports flat structure: <dir>/task-name.json (standalone runner output).
    """
    tasks = []

    # Harbor structure: look for agent/result.json directories
    for agent_result in sorted(results_dir.rglob("agent/result.json")):
        task_dir = agent_result.parent.parent
        verifier_reward = task_dir / "verifier" / "reward.json"

        # Load agent result
        try:
            result = json.loads(agent_result.read_text())
        except Exception as e:
            print(f"  [skip] Can't read {agent_result}: {e}")
            continue

        # Load verifier eval
        eval_result = None
        if verifier_reward.exists():
            try:
                eval_result = json.loads(verifier_reward.read_text())
            except Exception:
                pass

        # If no verifier output, check if eval is inline (standalone runner)
        if eval_result is None and "eval" in result:
            eval_result = result["eval"]

        task_name = result.get("task_name", task_dir.name.split("__")[0])
        tasks.append({
            "task_name": task_name,
            "result": result,
            "eval_result": eval_result,
            "source": str(agent_result),
        })

    # Flat structure: look for *.json files with task results
    if not tasks:
        for json_file in sorted(results_dir.glob("*.json")):
            if json_file.name.startswith("_"):
                continue
            try:
                result = json.loads(json_file.read_text())
            except Exception:
                continue
            if "execution_trace" not in result and "task_result" not in result:
                continue
            eval_result = result.get("eval")
            task_name = result.get("task_name", json_file.stem)
            tasks.append({
                "task_name": task_name,
                "result": result,
                "eval_result": eval_result,
                "source": str(json_file),
            })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Extract skills from Harbor results")
    parser.add_argument("results_dir", help="Harbor output directory (e.g. results/modal/<run-dir>)")
    parser.add_argument("--output", "-o", default="skills/", help="Output directory for skill .md files")
    parser.add_argument("--threshold", "-t", type=float, default=2.0,
                        help="Minimum score to consider for skill extraction")
    parser.add_argument("--model", "-m", default="gpt-4o-mini", help="Model for skill extraction LLM calls")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be extracted without writing")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[error] Results directory not found: {results_dir}")
        sys.exit(1)

    tasks = find_tasks(results_dir)
    if not tasks:
        print(f"[error] No task results found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(tasks)} task result(s) in {results_dir}")

    extractor = SkillExtractor(model=args.model)
    config = {"skill_score_threshold": args.threshold}

    if not args.dry_run:
        store = SkillStore(skills_dir=args.output)

    extracted = 0
    skipped = 0

    for task in tasks:
        task_name = task["task_name"]
        result = task["result"]
        eval_result = task["eval_result"]

        if eval_result is None:
            print(f"  [skip] {task_name}: no eval result")
            skipped += 1
            continue

        # Normalize eval_result: Harbor reward.json has "details" at top level
        # while inline eval has "details" nested
        if "details" not in eval_result and "overall_score" in eval_result:
            eval_result = {"details": eval_result, "passed": eval_result.get("passed", False)}

        score = (eval_result.get("details", {}) or {}).get("overall_score")
        if score is None:
            print(f"  [skip] {task_name}: no score")
            skipped += 1
            continue

        print(f"  {task_name}: score={score}", end="")

        if float(score) < args.threshold:
            print(f" (below threshold {args.threshold})")
            skipped += 1
            continue

        # Ask LLM if this is generalizable
        decision = extractor.should_store(result, eval_result, config)
        if not decision.get("store"):
            print(f" — not storing: {decision.get('reason', 'unknown')}")
            skipped += 1
            continue

        print(f" — extracting skill: {decision.get('reason', '')}")

        if args.dry_run:
            extracted += 1
            continue

        # Extract and save skill
        task_data = {"task_name": task_name, "instruction": result.get("instruction", "")}
        skill_md = extractor.extract_skill(task_data, result, eval_result)

        metadata = {
            "score": score,
            "task_type": decision.get("task_type", ""),
            "tags": decision.get("tags", []),
        }
        path = store.save_skill(skill_md, task_name, metadata)
        print(f"    Saved: {path}")
        extracted += 1

    print(f"\n--- Summary ---")
    print(f"Total tasks:  {len(tasks)}")
    print(f"Extracted:    {extracted}")
    print(f"Skipped:      {skipped}")
    if not args.dry_run and extracted > 0:
        print(f"Skills dir:   {args.output}")
        print(f"\nNext: run ./scripts/sync-harness.sh to sync skills to task environments")


if __name__ == "__main__":
    main()
