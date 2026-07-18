#!/bin/bash
# Thin wrapper around `harbor run` that wires up agent credentials from .env.
#
# Usage:
#   ./harbor_runner.sh -e modal  -a codex       -m gpt-4.1-mini             -p tasks/real_118/task-11-...
#   ./harbor_runner.sh -e modal  -a claude-code -m claude-sonnet-4-20250514 -p tasks/real_118
#   ./harbor_runner.sh -e docker -a codex       -m gpt-4.1-mini             -p tasks/real_118/task-11-...
#
# Forwards all args to `harbor run`. Uses UNIAPI_KEY as an OpenAI-compatible
# proxy if no provider-specific keys are set.
set -euo pipefail

if [ -f .env ]; then
    set -a; . .env; set +a
fi

# Sync harness/, configs/, skills/ into every task's environment/ so the
# Docker build context is complete (Modal drops empty dirs).
if [ -x scripts/sync-harness.sh ]; then
    ./scripts/sync-harness.sh
fi

# OpenAI-compatible endpoint for codex / OpenAI-shaped agents.
if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${UNIAPI_KEY:-}" ]; then
    export OPENAI_API_KEY="$UNIAPI_KEY"
fi
if [ -z "${OPENAI_BASE_URL:-}" ] && [ -n "${UNIAPI_BASE:-}" ]; then
    export OPENAI_BASE_URL="${UNIAPI_BASE%/}/v1"
fi

# Task verifier (test.sh) reads LLM_BASE_URL via task.toml env mapping.
if [ -z "${LLM_BASE_URL:-}" ] && [ -n "${OPENAI_BASE_URL:-}" ]; then
    export LLM_BASE_URL="$OPENAI_BASE_URL"
fi

# Anthropic endpoint for claude-code.
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -n "${UNIAPI_KEY:-}" ]; then
    export ANTHROPIC_API_KEY="$UNIAPI_KEY"
fi
if [ -z "${ANTHROPIC_BASE_URL:-}" ] && [ -n "${UNIAPI_BASE:-}" ]; then
    export ANTHROPIC_BASE_URL="${UNIAPI_BASE%/}/claude"
fi

exec harbor run "$@"
