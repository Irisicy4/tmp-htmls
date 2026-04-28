"""Compute static-vs-agentic agreement statistics and write agreement_report.json."""

import glob
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _load_verification_files(directory: str) -> list[dict]:
    """Load all *_agentic_verification.json files from a directory."""
    pattern = os.path.join(directory, "*_agentic_verification.json")
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            records.append(json.load(f))
    return records


def _pct(numerator: int, denominator: int) -> float:
    """Return percentage rounded to 2 decimal places; 0.0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def _count_static_results(result_dir: str) -> tuple[int, int]:
    """Return (passed_count, total_count) from a static judge trials directory.

    Expects result_dir to contain task subdirectories each with a result.json
    (e.g., result_dir/task-01-xxx/result.json).
    """
    passed = total = 0
    for path in sorted(Path(result_dir).glob("*/result.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            total += 1
            if data.get("passed"):
                passed += 1
        except (json.JSONDecodeError, KeyError):
            continue
    return passed, total


def compute_agreement(
    original_dir: str,
    synthetic_dir: str,
    original_result_dir: Optional[str] = None,
    synthetic_result_dir: Optional[str] = None,
) -> dict:
    """Read all verification JSONs and compute agreement statistics.

    Args:
        original_dir: Path to agentic_judge/results/original/
        synthetic_dir: Path to agentic_judge/results/synthetic/
        original_result_dir: Optional path to results/modal-updated/ for static pass rate.
        synthetic_result_dir: Optional path to results/synthetic/ for static pass rate.

    Returns:
        Agreement report dict (also written to agentic_judge/results/agreement_report.json).
    """
    records = _load_verification_files(original_dir) + _load_verification_files(synthetic_dir)

    total_sampled = len(records)

    verifiable = [r for r in records if r["agentic_verified"] is not None]
    unverifiable = [r for r in records if r["agentic_verified"] is None]

    verifiable_count = len(verifiable)
    unverifiable_count = len(unverifiable)

    reason_counts: dict[str, int] = defaultdict(int)
    for r in unverifiable:
        reason = r.get("null_reason") or "unknown"
        reason_counts[reason] += 1

    unverifiable_by_reason: dict[str, dict] = {}
    for reason, count in reason_counts.items():
        unverifiable_by_reason[reason] = {
            "count": count,
            "pct_of_unverifiable": _pct(count, unverifiable_count),
            "pct_of_total": _pct(count, total_sampled),
        }

    agreed = sum(
        1 for r in verifiable
        if r["static_judge_passed"] == bool(r["agentic_verified"])
    )
    static_passed_verifiable = [r for r in verifiable if r["static_judge_passed"]]
    caught = sum(1 for r in static_passed_verifiable if r["agentic_verified"] is False)
    joint_passed = sum(
        1 for r in verifiable if r["static_judge_passed"] and r["agentic_verified"] is True
    )

    agreement_rate = _pct(agreed, verifiable_count)
    catch_rate = _pct(caught, len(static_passed_verifiable))
    joint_pass_rate = _pct(joint_passed, verifiable_count)
    agentic_verification_rate = _pct(
        sum(1 for r in verifiable if r["agentic_verified"] is True),
        verifiable_count,
    )

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in verifiable:
        by_category[r.get("category", "Unknown")].append(r)

    category_breakdown: dict[str, dict] = {}
    for cat, cat_records in sorted(by_category.items()):
        cat_verifiable = len(cat_records)
        cat_passed = sum(1 for r in cat_records if r["static_judge_passed"])
        cat_agentic_passed = sum(1 for r in cat_records if r["agentic_verified"] is True)
        cat_caught = sum(
            1 for r in cat_records if r["static_judge_passed"] and r["agentic_verified"] is False
        )
        cat_agreed = sum(
            1 for r in cat_records if r["static_judge_passed"] == bool(r["agentic_verified"])
        )
        category_breakdown[cat] = {
            "n_verifiable": cat_verifiable,
            "static_pass_rate": _pct(cat_passed, cat_verifiable),
            "agentic_pass_rate": _pct(cat_agentic_passed, cat_verifiable),
            "agreement_rate": _pct(cat_agreed, cat_verifiable),
            "catch_rate": _pct(cat_caught, cat_passed),
        }

    report: dict = {
        "total_sampled": total_sampled,
        "verifiable_count": verifiable_count,
        "unverifiable_count": unverifiable_count,
        "unverifiable_pct": _pct(unverifiable_count, total_sampled),
        "unverifiable_by_reason": unverifiable_by_reason,
        "agreement_rate": agreement_rate,
        "catch_rate": catch_rate,
        "joint_pass_rate": joint_pass_rate,
        "agentic_verification_rate": agentic_verification_rate,
        "category_breakdown": category_breakdown,
    }

    # Optionally include static pass rates from full result directories
    if original_result_dir is not None:
        orig_passed, orig_total = _count_static_results(original_result_dir)
        report["static_pass_rate_original"] = _pct(orig_passed, orig_total)
    if synthetic_result_dir is not None:
        synth_passed, synth_total = _count_static_results(synthetic_result_dir)
        report["static_pass_rate_synthetic"] = _pct(synth_passed, synth_total)

    # Write report JSON next to the results directories
    report_path = Path(original_dir).parent / "agreement_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Agreement report written to {report_path}")
    return report


def _resolve_repo_root() -> Path:
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], text=True
        ).strip()
        return Path(common_dir).resolve().parent
    except subprocess.CalledProcessError:
        return Path(__file__).parent.parent.parent


def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Compute static-vs-agentic agreement statistics")
    parser.add_argument(
        "--dataset",
        choices=["original", "synthetic", "real118", "all"],
        default="all",
        help="Which dataset(s) to include (default: all)",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        config_path = repo_root / "agentic_judge" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    aj_results = repo_root / "agentic_judge" / "results"

    datasets = ["original", "synthetic", "real118"] if args.dataset == "all" else [args.dataset]

    records = []
    for dataset in datasets:
        d = aj_results / dataset
        if d.exists():
            records += _load_verification_files(str(d))

    if not records:
        print("No verification files found for the requested datasets.")
        return

    result_dirs = {
        ds: str(repo_root / config["result_dirs"][ds])
        for ds in datasets
        if ds in config.get("result_dirs", {})
    }

    # Reuse compute_agreement logic but feed pre-loaded records
    # Write a merged report to agentic_judge/results/agreement_report.json
    total_sampled = len(records)
    verifiable = [r for r in records if r["agentic_verified"] is not None]
    unverifiable = [r for r in records if r["agentic_verified"] is None]

    from collections import defaultdict
    reason_counts: dict[str, int] = defaultdict(int)
    for r in unverifiable:
        reason_counts[r.get("null_reason") or "unknown"] += 1

    unverifiable_by_reason = {
        reason: {
            "count": count,
            "pct_of_unverifiable": _pct(count, len(unverifiable)),
            "pct_of_total": _pct(count, total_sampled),
        }
        for reason, count in reason_counts.items()
    }

    verifiable_count = len(verifiable)
    unverifiable_count = len(unverifiable)
    agreed = sum(1 for r in verifiable if r["static_judge_passed"] == bool(r["agentic_verified"]))
    static_passed = [r for r in verifiable if r["static_judge_passed"]]
    caught = sum(1 for r in static_passed if r["agentic_verified"] is False)
    joint_passed = sum(1 for r in verifiable if r["static_judge_passed"] and r["agentic_verified"] is True)

    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in verifiable:
        by_category[r.get("category", "Unknown")].append(r)

    category_breakdown = {}
    for cat, cat_records in sorted(by_category.items()):
        n = len(cat_records)
        cat_passed = sum(1 for r in cat_records if r["static_judge_passed"])
        cat_agentic = sum(1 for r in cat_records if r["agentic_verified"] is True)
        cat_caught = sum(1 for r in cat_records if r["static_judge_passed"] and r["agentic_verified"] is False)
        cat_agreed = sum(1 for r in cat_records if r["static_judge_passed"] == bool(r["agentic_verified"]))
        category_breakdown[cat] = {
            "n_verifiable": n,
            "static_pass_rate": _pct(cat_passed, n),
            "agentic_pass_rate": _pct(cat_agentic, n),
            "agreement_rate": _pct(cat_agreed, n),
            "catch_rate": _pct(cat_caught, cat_passed),
        }

    report = {
        "datasets": datasets,
        "total_sampled": total_sampled,
        "verifiable_count": verifiable_count,
        "unverifiable_count": unverifiable_count,
        "unverifiable_pct": _pct(unverifiable_count, total_sampled),
        "unverifiable_by_reason": unverifiable_by_reason,
        "agreement_rate": _pct(agreed, verifiable_count),
        "catch_rate": _pct(caught, len(static_passed)),
        "joint_pass_rate": _pct(joint_passed, verifiable_count),
        "agentic_verification_rate": _pct(
            sum(1 for r in verifiable if r["agentic_verified"] is True), verifiable_count
        ),
        "category_breakdown": category_breakdown,
    }

    for ds, rdir in result_dirs.items():
        if Path(rdir).exists():
            p, t = _count_static_results(rdir)
            report[f"static_pass_rate_{ds}"] = _pct(p, t)

    report_path = aj_results / "agreement_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Agreement report written to {report_path}")


if __name__ == "__main__":
    main()
