#!/bin/bash
# Run Harbor tasks on Modal (cloud, parallel) or Docker (local).
#
# Usage:
#   ./harbor_runner.sh [task_path] [output_dir]
#
# Examples:
#   ./harbor_runner.sh tasks                                            # all tasks on Modal (parallel)
#   ./harbor_runner.sh tasks/batch-1/task-03-busca-las-mejores-casas-rurales    # single task on Modal
#   ./harbor_runner.sh tasks results/my-run                             # custom output dir
#   ENV=docker ./harbor_runner.sh                                       # single task, local Docker
#
# Environment variables (set in .env or export before running):
#   ENV                   modal (default) or docker
#   OPENAI_API_KEY        Required — API key (or UniAPI key for non-OpenAI models)
#   LLM_BASE_URL          Optional — proxy endpoint (e.g. https://api.uniapi.io/v1)
#   LLM_MODEL             Optional — model override (default: gpt-4.1-mini from config)
#   COCOA_MAX_ITERATIONS  Optional — max agent iterations (default: 50)
#   MODAL_SECRET          Optional — Modal secret name (default: openai-secret)
#
# Prerequisites:
#   cp env.template .env   # set OPENAI_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL)
#   # For Modal: modal secret create openai-secret OPENAI_API_KEY=sk-...

set -e
cd "$(dirname "$0")"

# Load .env
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY must be set. Create .env from env.template."
    exit 1
fi

ENV="${ENV:-modal}"
export COCOA_CONFIG="${COCOA_CONFIG:-/harness/configs/skill-phase1.json}"
export COCOA_MAX_ITERATIONS="${COCOA_MAX_ITERATIONS:-50}"
MODAL_SECRET="${MODAL_SECRET:-openai-secret}"

if [ -z "$1" ]; then
    echo "Usage: ./harbor_runner.sh <task_path> [output_dir]"
    echo "  e.g. ./harbor_runner.sh tasks/batch-1"
    echo "       ./harbor_runner.sh tasks/batch-1/task-01-im-looking-for-backpack-under"
    exit 1
fi
TASK_PATH="$1"
OUTPUT_DIR="${2:-results/$ENV}"
AGENT="${AGENT:-agents.cocoa_harbor_agent:CocoaHarborAgent}"
MODEL="${MODEL:-openai/gpt-4.1-mini}"

# Agent kwargs (env vars passed into the container)
AK_ARGS=()
[ -n "$OPENAI_API_KEY" ] && AK_ARGS+=(--ak "OPENAI_API_KEY=$OPENAI_API_KEY")
[ -n "$LLM_BASE_URL" ] && AK_ARGS+=(--ak "LLM_BASE_URL=$LLM_BASE_URL" --ak "OPENAI_BASE_URL=$LLM_BASE_URL")
[ -n "$LLM_MODEL" ] && AK_ARGS+=(--ak "LLM_MODEL=$LLM_MODEL")
[ -n "$COCOA_MAX_ITERATIONS" ] && AK_ARGS+=(--ak "COCOA_MAX_ITERATIONS=$COCOA_MAX_ITERATIONS")

# Modal-specific: mount secrets
EK_ARGS=()
if [ "$ENV" = "modal" ]; then
    EK_ARGS+=(--ek "secrets=[\"$MODAL_SECRET\"]")
fi

# Sync harness + config files into task environments (Modal build context)
./scripts/sync-harness.sh

echo "=== Harbor run: env=$ENV task=$TASK_PATH model=$MODEL ==="
harbor run \
  -p "$TASK_PATH" \
  --agent-import-path "$AGENT" \
  -m "$MODEL" \
  -e "$ENV" \
  -o "$OUTPUT_DIR" \
  "${AK_ARGS[@]}" \
  "${EK_ARGS[@]}"

# Print rubric scores from verifier
LATEST_RUN=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_RUN" ]; then
  STDOUT=$(find "$LATEST_RUN" -path "*/verifier/test-stdout.txt" 2>/dev/null | head -1)
  if [ -n "$STDOUT" ] && [ -f "$STDOUT" ]; then
    echo ""
    echo "=== Rubric Scores ==="
    cat "$STDOUT"
  fi
fi
