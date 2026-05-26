#!/usr/bin/env python3
"""
CLI verifier for Harbor's tests/test.sh model.

Loads the task's tests/test_task.py, calls its test() function N times against
the agent's parsed result, and writes:

    <output-dir>/reward.json     - aggregate scores (Harbor's grading file)
    <output-dir>/judge_runs.json - per-run raw judge results
    <output-dir>/feedback.txt    - human-readable summary

reward.json schema (flat, Harbor-readable):
    overall_score        : mean across runs
    overall_score_std    : stdev across runs (only if N > 1)
    <dim>_mean / <dim>_std : per-dimension stats
    n_judge_runs         : how many judge calls were made
    passed               : 1.0 if mean >= 3.0 else 0.0

Usage (inside the task environment):
    python3 /harness/run_judge.py \\
        --tests-dir /tests \\
        --agent-result /logs/agent/agent_result.json \\
        --output-dir /logs/verifier \\
        --n-runs "${JUDGE_RUNS:-1}"
"""
import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path


PASS_THRESHOLD = 3.0


def _load_test_module(tests_dir: Path):
    """Load tests/test_task.py as a uniquely-named module."""
    test_task_py = tests_dir / "test_task.py"
    if not test_task_py.is_file():
        raise FileNotFoundError(f"{test_task_py} not found")
    mod_name = f"task_test_{tests_dir.parent.name}"
    spec = importlib.util.spec_from_file_location(mod_name, test_task_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _aggregate(runs: list[dict], dimension_weights: dict | None) -> dict:
    """Build the flat reward.json dict from a list of test() return values."""
    reward: dict = {"n_judge_runs": len(runs)}

    scores = []
    for r in runs:
        if not isinstance(r, dict):
            continue
        details = r.get("details") or {}
        s = details.get("overall_score")
        if s is None:
            continue
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            pass

    if not scores:
        reward["overall_score"] = 0.0
        reward["passed"] = 0.0
        return reward

    reward["overall_score"] = round(statistics.mean(scores), 3)
    if len(scores) > 1:
        reward["overall_score_std"] = round(statistics.stdev(scores), 3)

    for dim in (dimension_weights or {}):
        vals = []
        for r in runs:
            if not isinstance(r, dict):
                continue
            ds = (r.get("details") or {}).get("dimension_scores") or {}
            v = ds.get(dim)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        if vals:
            reward[f"{dim}_mean"] = round(statistics.mean(vals), 3)
            if len(vals) > 1:
                reward[f"{dim}_std"] = round(statistics.stdev(vals), 3)

    reward["passed"] = 1.0 if reward["overall_score"] >= PASS_THRESHOLD else 0.0
    return reward


def main():
    parser = argparse.ArgumentParser(description="Run the LLM judge for a Harbor task.")
    parser.add_argument("--tests-dir", required=True, help="Path to the task's tests/ dir (containing test_task.py)")
    parser.add_argument("--agent-result", required=True, help="Path to a JSON file with the parsed agent output")
    parser.add_argument("--output-dir", required=True, help="Directory to write reward.json, judge_runs.json, feedback.txt")
    parser.add_argument("--n-runs", type=int, default=1, help="Number of judge calls (for variance studies)")
    parser.add_argument("--model", default=None, help="Judge model. Overrides LLM_MODEL.")
    parser.add_argument("--temperature", type=float, default=None, help="Judge temperature. Overrides LLM_TEMPERATURE.")
    args = parser.parse_args()

    if args.model is not None:
        os.environ["LLM_MODEL"] = args.model
    if args.temperature is not None:
        os.environ["LLM_TEMPERATURE"] = str(args.temperature)

    sys.path.insert(0, "/harness")

    tests_dir = Path(args.tests_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_result_path = Path(args.agent_result)
    if not agent_result_path.is_file():
        print(f"ERROR: agent result {agent_result_path} not found", file=sys.stderr)
        # Still write a zero reward so Harbor sees something.
        (out_dir / "reward.json").write_text(json.dumps({"overall_score": 0.0, "passed": 0.0, "error": "agent_result missing"}))
        sys.exit(2)
    agent_result = json.loads(agent_result_path.read_text())

    tt = _load_test_module(tests_dir)
    if not hasattr(tt, "test"):
        msg = f"{tests_dir}/test_task.py has no test() function"
        print(f"ERROR: {msg}", file=sys.stderr)
        (out_dir / "reward.json").write_text(json.dumps({"overall_score": 0.0, "passed": 0.0, "error": msg}))
        sys.exit(2)

    runs = []
    for i in range(args.n_runs):
        try:
            res = tt.test(agent_result)
        except Exception as exc:
            res = {
                "passed": False,
                "feedback": f"Judge call crashed: {exc}",
                "details": {"overall_score": 0, "error": str(exc)},
            }
        runs.append(res)
        score = (res.get("details") or {}).get("overall_score", "?")
        print(f"[judge {i + 1}/{args.n_runs}] score={score}")

    reward = _aggregate(runs, getattr(tt, "DIMENSION_WEIGHTS", None))
    (out_dir / "reward.json").write_text(json.dumps(reward, indent=2))
    (out_dir / "judge_runs.json").write_text(json.dumps(runs, indent=2, default=str))

    feedback_lines = [f"Aggregate (n={args.n_runs}): overall_score={reward.get('overall_score', 0):.3f}"]
    if "overall_score_std" in reward:
        feedback_lines.append(f"  std={reward['overall_score_std']:.3f}")
    feedback_lines.append("")
    for i, r in enumerate(runs):
        feedback_lines.append(f"--- Judge run {i + 1} ---")
        feedback_lines.append(r.get("feedback", "") or "")
        feedback_lines.append("")
    (out_dir / "feedback.txt").write_text("\n".join(feedback_lines))

    print(f"Wrote {out_dir}/reward.json: {json.dumps(reward)}")


if __name__ == "__main__":
    main()
