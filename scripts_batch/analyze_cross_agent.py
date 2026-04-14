"""
Cross-agent analysis for deprivacy-100 experiments.

Reads all agent results from a base directory and produces:
  1. Console summary table (agent x metric)
  2. cross_agent_summary.json — machine-readable comparison
  3. per_task_matrix.csv     — task x agent score matrix (for heatmaps, pivots)

Usage:
  # After collecting all results:
  modal run scripts_batch/deprivacy_modal_runner.py --collect deprivacy/claude-code/20260330-...
  modal run scripts_batch/deprivacy_modal_runner.py --collect deprivacy/codex/20260330-...
  modal run scripts_batch/deprivacy_modal_runner.py --collect deprivacy/gemini-cli/20260330-...
  modal run scripts_batch/deprivacy_modal_runner.py --collect deprivacy/aider/20260330-...

  # Then analyze:
  python scripts_batch/analyze_cross_agent.py results/deprivacy/

The script auto-discovers agent runs by finding _summary.json files recursively.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_all_runs(base_dir: Path) -> list[dict]:
    """Find all _summary.json files and their sibling per-task JSONs."""
    runs = []
    for summary_path in sorted(base_dir.rglob("_summary.json")):
        summary = json.loads(summary_path.read_text())
        run_dir = summary_path.parent

        # Load per-task results
        tasks = []
        for f in sorted(run_dir.glob("task-*.json")):
            tasks.append(json.loads(f.read_text()))

        runs.append({
            "summary": summary,
            "tasks": tasks,
            "run_dir": str(run_dir),
        })
    return runs


def build_task_matrix(runs: list[dict]) -> tuple[list[str], list[str], dict]:
    """Build task x agent score matrix.

    Returns:
        agents: sorted agent labels ("agent/model")
        task_names: sorted task names
        matrix: {task_name: {agent_label: score}}
    """
    agents = set()
    task_names = set()
    matrix: dict[str, dict[str, float]] = defaultdict(dict)

    for run in runs:
        s = run["summary"]
        agent_label = f"{s['agent']}/{s.get('model', 'default')}"
        agents.add(agent_label)

        for task in run["tasks"]:
            name = task.get("task", task.get("task_name", "unknown"))
            task_names.add(name)
            matrix[name][agent_label] = task.get("score", 0.0)

    return sorted(agents), sorted(task_names), dict(matrix)


def print_summary_table(runs: list[dict]):
    """Print a console table comparing agents."""
    print("\n" + "=" * 90)
    print("  CROSS-AGENT COMPARISON: deprivacy-100")
    print("=" * 90)

    header = f"{'Agent':<20} {'Model':<25} {'Score':>7} {'Pass%':>7} {'Pass':>5} {'Err':>5} {'Tasks':>5}"
    print(header)
    print("-" * 90)

    for run in sorted(runs, key=lambda r: r["summary"].get("mean_score", 0), reverse=True):
        s = run["summary"]
        print(
            f"{s['agent']:<20} {s.get('model', 'default'):<25} "
            f"{s['mean_score']:>7.3f} {s.get('pass_rate_pct', 0):>6.1f}% "
            f"{s['n_passed']:>5} {s.get('n_errors', 0):>5} {s['n_tasks']:>5}"
        )

    # --- Category breakdown across agents ---
    print(f"\n{'='*90}")
    print("  PER-CATEGORY BREAKDOWN")
    print(f"{'='*90}")

    # Collect all categories
    all_cats = set()
    for run in runs:
        all_cats.update(run["summary"].get("category_summary", {}).keys())

    cat_header = f"{'Category':<30}"
    for run in sorted(runs, key=lambda r: r["summary"]["agent"]):
        label = run["summary"]["agent"][:12]
        cat_header += f" {label:>12}"
    print(cat_header)
    print("-" * (30 + 13 * len(runs)))

    for cat in sorted(all_cats):
        row = f"{cat:<30}"
        for run in sorted(runs, key=lambda r: r["summary"]["agent"]):
            info = run["summary"].get("category_summary", {}).get(cat, {})
            avg = info.get("mean_score", 0)
            row += f" {avg:>12.2f}"
        print(row)

    # --- Dimension breakdown across agents ---
    print(f"\n{'='*90}")
    print("  PER-DIMENSION AVERAGES")
    print(f"{'='*90}")

    all_dims = set()
    for run in runs:
        all_dims.update(run["summary"].get("dimension_averages", {}).keys())

    dim_header = f"{'Dimension':<35}"
    for run in sorted(runs, key=lambda r: r["summary"]["agent"]):
        label = run["summary"]["agent"][:12]
        dim_header += f" {label:>12}"
    print(dim_header)
    print("-" * (35 + 13 * len(runs)))

    for dim in sorted(all_dims):
        row = f"{dim:<35}"
        for run in sorted(runs, key=lambda r: r["summary"]["agent"]):
            avg = run["summary"].get("dimension_averages", {}).get(dim, 0)
            row += f" {avg:>12.2f}"
        print(row)

    # --- Head-to-head wins ---
    print(f"\n{'='*90}")
    print("  HEAD-TO-HEAD (task-level wins)")
    print(f"{'='*90}")

    agents_sorted = sorted(runs, key=lambda r: r["summary"]["agent"])
    _, task_names, matrix = build_task_matrix(runs)

    for i, run_a in enumerate(agents_sorted):
        for run_b in agents_sorted[i + 1:]:
            a_label = f"{run_a['summary']['agent']}/{run_a['summary'].get('model', 'default')}"
            b_label = f"{run_b['summary']['agent']}/{run_b['summary'].get('model', 'default')}"
            a_wins, b_wins, ties = 0, 0, 0
            for task_name in task_names:
                sa = matrix.get(task_name, {}).get(a_label, 0)
                sb = matrix.get(task_name, {}).get(b_label, 0)
                if sa > sb:
                    a_wins += 1
                elif sb > sa:
                    b_wins += 1
                else:
                    ties += 1
            print(f"  {a_label} vs {b_label}:  {a_wins} / {b_wins} / {ties} (wins/losses/ties)")


def save_cross_agent_summary(runs: list[dict], out_dir: Path):
    """Save machine-readable cross-agent comparison."""
    comparison = {
        "agents": [],
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }

    agents, task_names, matrix = build_task_matrix(runs)

    for run in sorted(runs, key=lambda r: r["summary"].get("mean_score", 0), reverse=True):
        s = run["summary"]
        comparison["agents"].append({
            "agent": s["agent"],
            "model": s.get("model", "default"),
            "run_tag": s["run_tag"],
            "mean_score": s["mean_score"],
            "pass_rate_pct": s.get("pass_rate_pct", 0),
            "n_passed": s["n_passed"],
            "n_errors": s.get("n_errors", 0),
            "n_tasks": s["n_tasks"],
            "category_summary": s.get("category_summary", {}),
            "dimension_averages": s.get("dimension_averages", {}),
        })

    comparison["task_matrix"] = {
        "agents": agents,
        "tasks": {name: matrix.get(name, {}) for name in task_names},
    }

    out_path = out_dir / "cross_agent_summary.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved: {out_path}")


def save_task_matrix_csv(runs: list[dict], out_dir: Path):
    """Save task x agent score matrix as CSV for spreadsheets / heatmaps."""
    agents, task_names, matrix = build_task_matrix(runs)

    out_path = out_dir / "per_task_matrix.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Find category for each task (from any run that has it)
        task_cats = {}
        for run in runs:
            for t in run["tasks"]:
                name = t.get("task", t.get("task_name"))
                if name and name not in task_cats:
                    task_cats[name] = t.get("category", "unknown")

        writer.writerow(["task", "category"] + agents)
        for task_name in task_names:
            cat = task_cats.get(task_name, "unknown")
            row = [task_name, cat]
            for agent in agents:
                row.append(matrix.get(task_name, {}).get(agent, ""))
            writer.writerow(row)

    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts_batch/analyze_cross_agent.py <results-base-dir>")
        print("  e.g. python scripts_batch/analyze_cross_agent.py results/deprivacy/")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        sys.exit(1)

    runs = load_all_runs(base_dir)
    if not runs:
        print(f"No _summary.json files found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(runs)} experiment run(s)")

    print_summary_table(runs)
    save_cross_agent_summary(runs, base_dir)
    save_task_matrix_csv(runs, base_dir)

    print(f"\nDone. Files saved to {base_dir}/")


if __name__ == "__main__":
    main()
