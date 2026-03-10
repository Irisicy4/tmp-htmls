#!/bin/bash
# Run any Harbor task with CocoaAgent.
#
# Usage:
#   ./run.sh [task_path] [output_dir]
#
# Examples:
#   ./run.sh                                    # task-01, local docker
#   ENV=modal ./run.sh                          # run on Modal (cloud)
#   ./run.sh tasks/... results/my-run
#
# Prerequisites:
#   Set API key: export OPENAI_API_KEY=sk-...
#   (No pre-built image — Harbor builds from task's environment/Dockerfile)
#
# Harbor passes OPENAI_API_KEY into the container when set.

set -e
cd "$(dirname "$0")"

# Load .env if present (Modal + OpenAI keys; .env is gitignored)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY must be set (agent and LLM judge need it)"
    echo "  export OPENAI_API_KEY=sk-your-key"
    exit 1
fi

# Use skill-phase1 config for evolve_bench parity (same as modal run_skill_phase1.sh)
export COCOA_CONFIG="${COCOA_CONFIG:-/cocoa-agent/configs/skill-phase1.json}"
export COCOA_MAX_ITERATIONS="${COCOA_MAX_ITERATIONS:-10}"

TASK_PATH="${1:-tasks/task-01-im-looking-for-backpack-under}"
OUTPUT_DIR="${2:-results/skill-phase1-test}"
AGENT="${AGENT:-agents.cocoa_agent:CocoaHarborAgent}"
MODEL="${MODEL:-openai/gpt-4.1-mini}"
ENV="${ENV:-docker}"

AK_ARGS=()
[ -n "$OPENAI_API_KEY" ] && AK_ARGS+=(--ak "OPENAI_API_KEY=$OPENAI_API_KEY")
[ -n "$LLM_BASE_URL" ] && AK_ARGS+=(--ak "LLM_BASE_URL=$LLM_BASE_URL" --ak "OPENAI_BASE_URL=$LLM_BASE_URL")
[ -n "$LLM_MODEL" ] && AK_ARGS+=(--ak "LLM_MODEL=$LLM_MODEL")
[ -n "$COCOA_MAX_ITERATIONS" ] && AK_ARGS+=(--ak "COCOA_MAX_ITERATIONS=$COCOA_MAX_ITERATIONS")

if [ "$ENV" = "modal" ]; then
  ./scripts/prepare-modal-context.sh "$TASK_PATH"
fi

echo "=== Harbor run: task=$TASK_PATH agent=$AGENT model=$MODEL env=$ENV ==="
harbor run \
  -p "$TASK_PATH" \
  --agent-import-path "$AGENT" \
  -m "$MODEL" \
  -e "$ENV" \
  -o "$OUTPUT_DIR" \
  "${AK_ARGS[@]}"

# Print rubric scores from verifier (Harbor captures to verifier/test-stdout.txt but doesn't show it)
LATEST_RUN=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
  STDOUT=$(find "$LATEST_RUN" -path "*/verifier/test-stdout.txt" 2>/dev/null | head -1)
  if [ -n "$STDOUT" ] && [ -f "$STDOUT" ]; then
    echo ""
    echo "=== Rubric Scores ==="
    cat "$STDOUT"
  fi
fi
