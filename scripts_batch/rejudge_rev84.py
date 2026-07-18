#!/usr/bin/env python3
"""Re-judge all 84 rev84 results locally with the fixed runner
(schema-guided verify_url_keys now actually invoked). Agents are NOT
re-run — we re-grade the stored task_result of each task and update
scores in place, preserving the pre-regrade score."""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))

for line in (REPO / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
if not os.environ.get("OPENAI_BASE_URL") and os.environ.get("LLM_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["LLM_BASE_URL"]

REV = pathlib.Path("/Users/icy/Desktop/act/2020604-evolvebench/_local_only/run_results_revise84")
SPEC_ROOT = REPO / "tasks" / "real_118_revise_84"
WORKERS = 4


def rejudge_one(task: str) -> dict:
    t0 = time.time()
    d = REV / "by_task" / task
    spec_path = SPEC_ROOT / task / "tests" / "test_grounded.py"
    if not spec_path.is_file():
        return {"task": task, "error": "no spec"}
    mod_name = f"rj_{task.replace('-','_')}"
    sp = importlib.util.spec_from_file_location(mod_name, spec_path)
    mod = importlib.util.module_from_spec(sp)
    sys.modules[mod_name] = mod
    mod.__name__ = mod_name
    sp.loader.exec_module(mod)

    result = json.load((d / "modal_verdict.json").open())
    old_score = result.get("score")
    try:
        v = mod.grade_with_llm(result)
    except Exception as e:
        return {"task": task, "error": f"{type(e).__name__}: {str(e)[:150]}"}

    details = v.get("details", {})
    new_score = details.get("overall_score")
    passed = v.get("passed")
    feedback = v.get("feedback", "") or ""
    json_ok = "Summary JSON parseable: True" in feedback

    # Update files in place
    m2 = dict(result)
    m2["score"] = new_score
    m2["passed"] = passed
    m2["feedback"] = feedback
    m2["dimension_scores"] = details.get("dimension_scores", {})
    # Persist the full per-layer judge trace for audit.
    m2["judge_trace"] = {
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
    }
    (d / "modal_verdict.json").write_text(json.dumps(m2, indent=2))
    (d / "feedback.txt").write_text(feedback)
    # Also drop a standalone, human-readable trace file per task.
    (d / "judge_trace.json").write_text(json.dumps(m2["judge_trace"], indent=2, ensure_ascii=False))

    old_v = json.load((d / "verdict.json").open()) if (d / "verdict.json").exists() else {}
    (d / "verdict.json").write_text(json.dumps({
        **old_v,
        "run_tag": old_v.get("run_tag", "?") + "+regrounded",
        "final_score": new_score,
        "passed": passed,
        "dim_scores": details.get("dimension_scores", {}),
        "json_parsed": json_ok,
        "score_before_regrounding": old_score,
        "faith_matched": details.get("faithfulness_report", {}).get("matched"),
        "faith_total": details.get("faithfulness_report", {}).get("total"),
    }, indent=2))
    return {"task": task, "old": old_score, "new": new_score,
            "dt": round(time.time() - t0, 1)}


def main():
    tasks = sorted([p.name for p in (REV / "by_task").iterdir() if p.is_dir()])
    print(f"re-judging {len(tasks)} tasks with {WORKERS} workers", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(rejudge_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"[{i}/{len(tasks)}] ✗ {r['task']}: {r['error']}", flush=True)
            else:
                delta = (r["new"] - r["old"]) if isinstance(r["old"], (int, float)) and isinstance(r["new"], (int, float)) else None
                ds = f"{delta:+.2f}" if delta is not None else "?"
                print(f"[{i}/{len(tasks)}] ✓ {r['task']}: {r['old']} → {r['new']} (Δ{ds}, {r['dt']}s)", flush=True)

    ok = [r for r in results if "error" not in r and isinstance(r.get("new"), (int, float))]
    scored = [r["new"] for r in ok]
    deltas = [r["new"] - r["old"] for r in ok if isinstance(r.get("old"), (int, float))]
    print(f"\ndone: {len(ok)}/{len(tasks)} re-judged")
    if scored:
        print(f"mean: {statistics.mean(scored):.2f}   meanΔ vs pre-regrade: {statistics.mean(deltas):+.2f}")

    # Refresh _summary.json
    all_scored = []; all_pass = 0; all_json = 0
    for d in (REV / "by_task").iterdir():
        if not d.is_dir(): continue
        v = json.load((d / "verdict.json").open())
        s = v.get("final_score")
        if isinstance(s, (int, float)):
            all_scored.append(s)
            if v.get("passed"): all_pass += 1
            if v.get("json_parsed"): all_json += 1
    (REV / "_summary.json").write_text(json.dumps({
        "n_tasks": len(all_scored),
        "mean_score": round(statistics.mean(all_scored), 2),
        "n_passed": all_pass,
        "n_json_parsed": all_json,
        "regrounded": True,
    }, indent=2))
    print(f"final: n={len(all_scored)} mean={statistics.mean(all_scored):.2f} pass={all_pass} json={all_json}")


if __name__ == "__main__":
    main()
