"""Re-run the judge locally for all tasks with all-None dimension scores."""
import json
import os
import importlib.util
from pathlib import Path


def load_env(env_file=Path(".env")):
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
    if not os.environ.get("OPENAI_BASE_URL") and os.environ.get("LLM_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["LLM_BASE_URL"]
        print(f"Bridged OPENAI_BASE_URL -> {os.environ['OPENAI_BASE_URL']}")


def is_judge_failed(data):
    if data.get("score", -1) != 0.0:
        return False
    ds = data.get("dimension_scores", {})
    return bool(ds) and all(v is None for v in ds.values())


def load_test(task_name, tasks_root):
    test_path = tasks_root / task_name / "tests" / "test_task.py"
    if not test_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("test_task", test_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    load_env()

    tasks_root = Path("tasks/real_118")
    result_dirs = [
        Path("results/real118/real118-claude/real118-claude"),
        Path("results/real118/real118-codex/real118-codex"),
    ]

    total = updated = 0
    errors = []

    for result_dir in result_dirs:
        print(f"\n{'='*60}")
        print(f"{result_dir}")
        print("="*60)
        for fpath in sorted(result_dir.glob("task-*.json")):
            if "_summary" in fpath.name:
                continue
            with open(fpath) as f:
                data = json.load(f)
            if not is_judge_failed(data):
                continue
            task_name = data["task"]
            total += 1
            print(f"[{task_name}] judging...", flush=True)

            mod = load_test(task_name, tasks_root)
            if not mod:
                print("  ERROR: no test script")
                errors.append((task_name, "no test script"))
                continue

            try:
                eval_result = mod.test(data)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                errors.append((task_name, str(e)))
                continue

            details = eval_result.get("details", {}) or {}
            score = float(details.get("overall_score", 0) or eval_result.get("overall_score", 0))
            passed = bool(eval_result.get("passed", False))
            dim_scores = details.get("dimension_scores", {})

            print(f"  -> score={score}  passed={passed}", flush=True)

            data["score"] = score
            data["passed"] = passed
            data["dimension_scores"] = dim_scores
            data["feedback"] = eval_result.get("feedback", "")
            data["evidence_summary"] = details.get("evidence_summary", "")
            data["dimension_reasoning"] = details.get("dimension_reasoning", {})
            data["votes_used"] = details.get("votes_used", 1)

            with open(fpath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            updated += 1

    print(f"\nDone: {updated}/{total} updated, {len(errors)} errors")
    for t, e in errors:
        print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
