"""Generate paper-ready summary table from agreement_report.json."""

import json
import subprocess
import sys
from pathlib import Path

HEADLINE_TEMPLATE = """\
=== Agentic Judge Headline Numbers ===
  Overall agreement rate:          {agreement_rate:.2f}%
  Plausible-but-wrong catch rate:  {catch_rate:.2f}%
  Joint pass rate:                 {joint_pass_rate:.2f}%
  Unverifiable tasks:              {unverifiable_pct:.2f}%
"""

_COL = "{:<25} {:>10} {:>18} {:>18} {:>12} {:>12}"
_SEP = "-" * 99


def format_report(report: dict) -> str:
    lines = []

    lines.append("\n=== Per-Category Breakdown ===\n")
    lines.append(_COL.format("Category", "N verified", "Static pass%", "Agentic pass%", "Agreement%", "Catch%"))
    lines.append(_SEP)

    breakdown = report.get("category_breakdown", {})
    overall_n = report["verifiable_count"]
    for cat in sorted(breakdown):
        row = breakdown[cat]
        lines.append(_COL.format(
            cat,
            row["n_verifiable"],
            f"{row['static_pass_rate']:.1f}%",
            f"{row['agentic_pass_rate']:.1f}%",
            f"{row['agreement_rate']:.1f}%",
            f"{row['catch_rate']:.1f}%",
        ))

    lines.append(_SEP)
    lines.append(_COL.format(
        "Overall",
        overall_n,
        "—",
        f"{report['agentic_verification_rate']:.1f}%",
        f"{report['agreement_rate']:.1f}%",
        f"{report['catch_rate']:.1f}%",
    ))

    lines.append("\n")
    lines.append(HEADLINE_TEMPLATE.format(
        agreement_rate=report["agreement_rate"],
        catch_rate=report["catch_rate"],
        joint_pass_rate=report["joint_pass_rate"],
        unverifiable_pct=report["unverifiable_pct"],
    ))

    lines.append("=== Unverifiable Tasks Breakdown ===")
    lines.append(
        f"  Total unverifiable: {report['unverifiable_count']} / {report['total_sampled']} "
        f"({report['unverifiable_pct']:.2f}%)\n"
    )
    for reason, stats in sorted(report.get("unverifiable_by_reason", {}).items()):
        lines.append(
            f"  {reason:<22}  count={stats['count']}  "
            f"({stats['pct_of_unverifiable']:.1f}% of unverifiable, "
            f"{stats['pct_of_total']:.1f}% of total)"
        )
    lines.append("")

    # Static pass rates if present
    if "static_pass_rate_original" in report or "static_pass_rate_synthetic" in report:
        lines.append("=== Static Judge Pass Rates (full datasets) ===")
        if "static_pass_rate_original" in report:
            lines.append(f"  Original dataset:   {report['static_pass_rate_original']:.2f}%")
        if "static_pass_rate_synthetic" in report:
            lines.append(f"  Synthetic dataset:  {report['static_pass_rate_synthetic']:.2f}%")
        lines.append("")

    return "\n".join(lines)


def _resolve_report_path() -> Path:
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], text=True
        ).strip()
        repo_root = Path(common_dir).resolve().parent
    except subprocess.CalledProcessError:
        repo_root = Path(__file__).parent.parent.parent
    return repo_root / "agentic_judge" / "results" / "agreement_report.json"


def main() -> None:
    report_path = _resolve_report_path()
    if not report_path.exists():
        print(f"Error: {report_path} not found. Run analysis/agreement.py first.", file=sys.stderr)
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    print(format_report(report))


if __name__ == "__main__":
    main()
