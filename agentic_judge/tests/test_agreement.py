import json
import os
import tempfile
from pathlib import Path

from agentic_judge.analysis.agreement import compute_agreement, _load_verification_files


def _write_verification(output_dir: str, task_name: str, **kwargs) -> None:
    data = {
        "task_name": task_name,
        "category": kwargs.get("category", "Shopping"),
        "dataset": kwargs.get("dataset", "original"),
        "static_judge_passed": kwargs.get("static_judge_passed", True),
        "static_judge_score": kwargs.get("static_judge_score", 3.5),
        "static_judge_dimensions": {},
        "agentic_verified": kwargs.get("agentic_verified", True),
        "agentic_finding": "Test finding.",
        "null_reason": kwargs.get("null_reason", None),
        "verification_method": kwargs.get("verification_method", "browser_navigation"),
        "deliverable_url": "https://example.com",
        "time_seconds": 10.0,
        "timestamp": "2026-04-04T00:00:00+00:00",
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{task_name}_agentic_verification.json"), "w") as f:
        json.dump(data, f)


def test_load_verification_files(tmp_path):
    out = str(tmp_path)
    _write_verification(out, "task-01")
    _write_verification(out, "task-02")
    (tmp_path / "other.json").write_text("{}")
    files = _load_verification_files(out)
    assert len(files) == 2


def test_compute_agreement_basic(tmp_path):
    orig = str(tmp_path / "original")
    synth = str(tmp_path / "synthetic")

    _write_verification(orig, "task-01", agentic_verified=True, static_judge_passed=True)
    _write_verification(orig, "task-02", agentic_verified=False, static_judge_passed=True,
                        verification_method="browser_navigation")
    _write_verification(synth, "task-03", agentic_verified=None, null_reason="no_urls",
                        verification_method="unverifiable")

    report = compute_agreement(orig, synth)

    assert report["total_sampled"] == 3
    assert report["unverifiable_count"] == 1
    assert report["verifiable_count"] == 2
    assert abs(report["unverifiable_pct"] - 33.33) < 0.1
    # catch_rate: task-02 static passed, agentic False → caught (1/2 = 50%)
    assert report["catch_rate"] == 50.0
    # agreement: task-01 agrees (T+T), task-02 disagrees (T+F) → 1/2 = 50%
    assert report["agreement_rate"] == 50.0


def test_compute_agreement_unverifiable_breakdown(tmp_path):
    orig = str(tmp_path / "original")
    synth = str(tmp_path / "synthetic")
    os.makedirs(synth, exist_ok=True)

    _write_verification(orig, "task-01", agentic_verified=None, null_reason="no_urls",
                        verification_method="unverifiable")
    _write_verification(orig, "task-02", agentic_verified=None, null_reason="navigation_error",
                        verification_method="url_check")
    _write_verification(orig, "task-03", agentic_verified=None, null_reason="gpt_uncertain",
                        verification_method="browser_navigation")

    report = compute_agreement(orig, synth)

    breakdown = report["unverifiable_by_reason"]
    assert breakdown["no_urls"]["count"] == 1
    assert breakdown["navigation_error"]["count"] == 1
    assert breakdown["gpt_uncertain"]["count"] == 1


def test_compute_agreement_joint_pass_rate(tmp_path):
    orig = str(tmp_path / "original")
    synth = str(tmp_path / "synthetic")
    os.makedirs(synth, exist_ok=True)

    _write_verification(orig, "task-01", agentic_verified=True, static_judge_passed=True)
    _write_verification(orig, "task-02", agentic_verified=True, static_judge_passed=True)
    _write_verification(orig, "task-03", agentic_verified=False, static_judge_passed=True)

    report = compute_agreement(orig, synth)
    assert abs(report["joint_pass_rate"] - 66.67) < 0.1


def test_compute_agreement_with_static_pass_rate(tmp_path):
    orig = str(tmp_path / "original")
    synth = str(tmp_path / "synthetic")
    orig_results = str(tmp_path / "orig_results")
    synth_results = str(tmp_path / "synth_results")
    os.makedirs(orig_results, exist_ok=True)
    os.makedirs(synth_results, exist_ok=True)

    _write_verification(orig, "task-01", agentic_verified=True)

    # Write fake static result JSONs in the new nested trial structure
    for i, passed in enumerate([True, True, False, False], 1):
        task_dir = os.path.join(orig_results, f"task-{i:02d}")
        os.makedirs(task_dir)
        with open(os.path.join(task_dir, "result.json"), "w") as f:
            json.dump({"task": f"task-{i:02d}", "passed": passed}, f)

    report = compute_agreement(orig, synth, orig_results, synth_results)
    assert report["static_pass_rate_original"] == 50.0
    assert report["static_pass_rate_synthetic"] == 0.0
