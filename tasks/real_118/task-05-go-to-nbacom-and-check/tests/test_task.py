"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_05_nba.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_05_nba"


TASK_INSTRUCTION = 'go to nba.com and check for Jayson Tatum\'s current 3-point status\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `source_url` (string) — Canonical nba.com URL where the stats were read.\n- `player_name` (string) — Player name as shown on the page (should be \'Jayson Tatum\').\n- `season` (string) — Season label (e.g. \'2024-25\').\n- `current_3pt_stats.attempts` (number) — 3PT attempts (season total or per-game).\n- `current_3pt_stats.makes` (number) — 3PT makes (matching the attempts basis).\n- `current_3pt_stats.pct` (number 0-1 or 0-100) — 3PT percentage.\n- `current_3pt_stats.basis` (string) — \'season-total\' OR \'per-game\'.\n- `data_freshness` (string) — When you fetched it (ISO date or \'as of YYYY-MM-DD\').\n\n**Optional but graded if present:**\n\n- `games_played` (integer) — Season games played.\n- `last_game_3pt` (object) — {date, makes, attempts} for last game played.\n- `career_context` (string) — Where this season ranks vs his career average.\n- `milestone_progress` (string) — All-time 3PT record progress, if relevant.\n- `alternate_sources_checked` (list[string]) — Other source URLs you double-checked (e.g. basketball-reference).\n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "source_url": "https://www.nba.com/player/1628369/jayson-tatum",\n  "player_name": "Jayson Tatum",\n  "season": "2024-25",\n  "current_3pt_stats": {\n    "attempts": 8.6,\n    "makes": 3.0,\n    "pct": 0.349,\n    "basis": "per-game"\n  },\n  "data_freshness": "as of 2026-06-01"\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
