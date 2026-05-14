#!/bin/bash
# Sync harness/, configs/, and skills/ into every task's environment/ directory.
#
# Harbor Modal uses environment/ as the Docker build context, so
# task environments must contain copies of all harness, config, and skill files.
# Called automatically by harbor_runner.sh before every run.
#
# Usage:
#   ./scripts/sync-harness.sh

set -e
cd "$(dirname "$0")/.."

HARNESS_FILES=(
    harness/run_task.py
    harness/skill_store.py
    harness/skill_extractor.py
    harness/evaluator.py
)

ADAPTER_FILES=(
    harness/adapters/__init__.py
    harness/adapters/cocoa_adapter.py
)

CONFIG_FILES=(
    configs/harbor-config.json
    configs/skill-phase1.json
    configs/skill-phase2.json
)

# Ensure skills directory exists (empty is fine for Phase 1)
mkdir -p skills

count=0
for env_dir in tasks/*/task-*/environment; do
    [ -d "$env_dir" ] || continue

    # Remove stale files from old layout
    rm -f "$env_dir/overlay.py" "$env_dir/agents__init__.py" 2>/dev/null || true
    rm -f "$env_dir/harbor-config.json" "$env_dir/skill-phase1.json" "$env_dir/skill-phase2.json" 2>/dev/null || true

    for src in "${HARNESS_FILES[@]}"; do
        cp "$src" "$env_dir/$(basename "$src")"
    done

    # Sync adapters into a subdirectory
    mkdir -p "$env_dir/adapters"
    for src in "${ADAPTER_FILES[@]}"; do
        cp "$src" "$env_dir/adapters/$(basename "$src")"
    done

    # Sync configs into a subdirectory
    mkdir -p "$env_dir/configs"
    for src in "${CONFIG_FILES[@]}"; do
        cp "$src" "$env_dir/configs/$(basename "$src")"
    done

    # Sync skills directory (has .keep placeholder for Phase 1, populated for Phase 2)
    mkdir -p "$env_dir/skills"
    rm -f "$env_dir/skills/"*.md "$env_dir/skills/.keep" 2>/dev/null || true
    cp skills/.keep "$env_dir/skills/"
    if ls skills/*.md &>/dev/null; then
        cp skills/*.md "$env_dir/skills/"
    fi

    # Ensure environment/data/ exists with a placeholder so the Dockerfile's
    # `COPY data/ /app/data/` succeeds even when the task has no input files.
    # Tasks with real input data keep their files; the .keep is just a placeholder.
    mkdir -p "$env_dir/data"
    [ -f "$env_dir/data/.keep" ] || touch "$env_dir/data/.keep"

    count=$((count + 1))
done

skill_count=$(ls skills/*.md 2>/dev/null | wc -l | tr -d ' ')
echo "Synced harness + configs + ${skill_count} skill(s) to $count task environments."
