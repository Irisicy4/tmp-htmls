#!/bin/bash
# Run the full skill experiment on Harbor: Phase 1 → extract skills → Phase 2 → compare.
#
# Phase 1: Run tasks with store_skills=true (no skill injection)
# Extract: Pull skills from Phase 1 results using LLM evaluation
# Phase 2: Rerun SAME tasks with use_skills=true (skills injected into prompts)
# Compare: Show score deltas between phases
#
# Usage:
#   ./run_skill_experiment.sh                                    # all tasks
#   ./run_skill_experiment.sh tasks/batch-1/task-01-im-looking-for-*     # single task
#   ./run_skill_experiment.sh tasks --max-iter 15                # limit iterations
#
# Environment variables (set in .env or export before running):
#   OPENAI_API_KEY        Required
#   LLM_BASE_URL          Optional — proxy endpoint
#   LLM_MODEL             Optional — model override
#   SKILL_THRESHOLD       Optional — min score for skill extraction (default: 2.0)
#   SKILL_MODEL           Optional — model for skill extraction (default: gpt-4o-mini)

set -euo pipefail
cd "$(dirname "$0")"

# Load .env
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "$1" ]; then
    echo "Usage: ./run_skill_experiment.sh <task_path> [--max-iter N] [--threshold N]"
    echo "  e.g. ./run_skill_experiment.sh tasks/batch-1"
    exit 1
fi
TASK_PATH="$1"
MAX_ITER="${COCOA_MAX_ITERATIONS:-50}"
SKILL_THRESHOLD="${SKILL_THRESHOLD:-2.0}"
SKILL_MODEL="${SKILL_MODEL:-gpt-4o-mini}"
TIMESTAMP=$(date -u +"%Y-%m-%d__%H-%M-%S")

# Parse optional flags
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --threshold) SKILL_THRESHOLD="$2"; shift 2 ;;
    --skill-model) SKILL_MODEL="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo "  Skill Experiment — ${TIMESTAMP}"
echo "  Tasks:     ${TASK_PATH}"
echo "  Max iter:  ${MAX_ITER}"
echo "  Threshold: ${SKILL_THRESHOLD}"
echo "============================================"

# Clean skills from previous experiments
echo ""
echo "Clearing old skills..."
rm -f skills/*.md
mkdir -p skills
touch skills/.keep

# ---- Phase 1: run tasks, no skills ----
echo ""
echo ">>> PHASE 1: running tasks (no skills)..."
COCOA_CONFIG=/harness/configs/skill-phase1.json \
COCOA_MAX_ITERATIONS="$MAX_ITER" \
  ./harbor_runner.sh "$TASK_PATH"

# Find the Phase 1 output directory (most recent)
PHASE1_DIR=$(ls -td results/modal/*/ 2>/dev/null | head -1)
if [ -z "$PHASE1_DIR" ]; then
  echo "[error] No Phase 1 results found"
  exit 1
fi
echo ""
echo "Phase 1 results: ${PHASE1_DIR}"

# ---- Extract skills from Phase 1 ----
echo ""
echo ">>> EXTRACTING SKILLS from Phase 1 results..."
python3 scripts/extract_skills.py "$PHASE1_DIR" \
  --output skills/ \
  --threshold "$SKILL_THRESHOLD" \
  --model "$SKILL_MODEL"

SKILL_COUNT=$(ls skills/*.md 2>/dev/null | wc -l | tr -d ' ')
if [ "$SKILL_COUNT" -eq 0 ]; then
  echo ""
  echo "[warning] No skills extracted. Phase 2 will run without skills."
fi
echo ""
echo "Skills extracted: ${SKILL_COUNT}"

# ---- Phase 2: rerun SAME tasks with skills ----
echo ""
echo ">>> PHASE 2: rerunning tasks (with ${SKILL_COUNT} skill(s))..."
COCOA_CONFIG=/harness/configs/skill-phase2.json \
COCOA_MAX_ITERATIONS="$MAX_ITER" \
  ./harbor_runner.sh "$TASK_PATH"

PHASE2_DIR=$(ls -td results/modal/*/ 2>/dev/null | head -1)
echo ""
echo "Phase 2 results: ${PHASE2_DIR}"

# ---- Compare results ----
echo ""
echo "============================================"
echo "  RESULTS COMPARISON"
echo "============================================"

python3 - "$PHASE1_DIR" "$PHASE2_DIR" <<'PYEOF'
import json, sys
from pathlib import Path


def load_scores(run_dir: str) -> dict[str, dict]:
    """Load per-task scores from a Harbor run directory."""
    scores = {}
    run = Path(run_dir)
    for reward_file in sorted(run.rglob("verifier/reward.json")):
        task_dir = reward_file.parent.parent
        # Extract task name from directory (strip __hash suffix)
        task_slug = task_dir.name.split("__")[0]
        try:
            data = json.loads(reward_file.read_text())
            scores[task_slug] = {
                "overall_score": data.get("overall_score", 0),
                "generic_score": data.get("generic_score", 0),
                "passed": data.get("passed", False),
                "reward": data.get("reward", 0),
            }
        except Exception:
            pass
    return scores


phase1_dir = sys.argv[1]
phase2_dir = sys.argv[2]

p1 = load_scores(phase1_dir)
p2 = load_scores(phase2_dir)

all_tasks = sorted(set(p1.keys()) | set(p2.keys()))

if not all_tasks:
    print("No task results found to compare.")
    sys.exit(1)

# Per-task comparison
print(f"\n{'Task':<45} {'Phase 1':>8} {'Phase 2':>8} {'Delta':>8}")
print("-" * 72)

deltas = []
for task in all_tasks:
    s1 = p1.get(task, {}).get("overall_score", 0)
    s2 = p2.get(task, {}).get("overall_score", 0)
    delta = s2 - s1
    deltas.append(delta)
    sign = "+" if delta > 0 else ""
    marker = " *" if abs(delta) >= 0.5 else ""
    # Truncate long task names
    name = task[:43] + ".." if len(task) > 45 else task
    print(f"  {name:<43} {s1:>8.2f} {s2:>8.2f} {sign}{delta:>7.2f}{marker}")

# Aggregate stats
p1_scores = [p1[t]["overall_score"] for t in all_tasks if t in p1]
p2_scores = [p2[t]["overall_score"] for t in all_tasks if t in p2]
p1_passed = sum(1 for t in all_tasks if p1.get(t, {}).get("passed", False))
p2_passed = sum(1 for t in all_tasks if p2.get(t, {}).get("passed", False))

avg1 = sum(p1_scores) / len(p1_scores) if p1_scores else 0
avg2 = sum(p2_scores) / len(p2_scores) if p2_scores else 0
avg_delta = avg2 - avg1

print("-" * 72)
print(f"  {'Avg Score':<43} {avg1:>8.2f} {avg2:>8.2f} {'+' if avg_delta > 0 else ''}{avg_delta:>7.2f}")
print(f"  {'Passed':<43} {p1_passed:>8d} {p2_passed:>8d} {'+' if p2_passed - p1_passed > 0 else ''}{p2_passed - p1_passed:>7d}")
print(f"  {'Total tasks':<43} {len(p1):>8d} {len(p2):>8d}")

print()
improved = sum(1 for d in deltas if d > 0)
declined = sum(1 for d in deltas if d < 0)
unchanged = sum(1 for d in deltas if d == 0)
print(f"  Improved: {improved}  |  Declined: {declined}  |  Unchanged: {unchanged}")

if avg_delta > 0:
    print(f"\n  Skills helped: avg score +{avg_delta:.2f}")
elif avg_delta < 0:
    print(f"\n  Skills did not help: avg score {avg_delta:.2f}")
else:
    print(f"\n  No change in avg score")

print(f"\n  Phase 1: {phase1_dir}")
print(f"  Phase 2: {phase2_dir}")
PYEOF
