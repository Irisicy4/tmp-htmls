"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_65_tuscany.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_65_tuscany"


TASK_INSTRUCTION = 'Find recommended dinner restaurants in Tuscany, Italy that are open on December 26th and 27th.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `location` (string) — Should contain \'Tuscany\'.\n- `meal_type` (string) — Should be \'dinner\'.\n- `date_range` (list[string]) — [\'2025-12-26\',\'2025-12-27\'] or similar (must include both days).\n- `restaurants` (list[object]) — Recommended dinner restaurants.\n- `restaurants[].name` (string) — \n- `restaurants[].city` (string) — Town within Tuscany.\n- `restaurants[].source_url` (string) — TripAdvisor/Google Maps/TheFork URL.\n- `restaurants[].open_dec_26` (boolean) — Open dinner Dec 26?\n- `restaurants[].open_dec_27` (boolean) — Open dinner Dec 27?\n- `restaurants[].verification_method` (string) — How you verified the hours.\n\n**Optional but graded if present:**\n\n- `restaurants[].cuisine` (string) — \n- `restaurants[].price_band` (string) — $/$$/$$$/$$$$.\n- `restaurants[].rating` (number 0-5) — \n- `restaurants[].reservation_url` (string) — \n- `sources_used` (list[string]) — All discovery platforms consulted.\n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "location": "Tuscany, Italy",\n  "meal_type": "dinner",\n  "date_range": [\n    "2025-12-26",\n    "2025-12-27"\n  ],\n  "restaurants": [\n    {\n      "name": "Trattoria ...",\n      "city": "Florence",\n      "source_url": "https://www.tripadvisor.com/...",\n      "open_dec_26": true,\n      "open_dec_27": true,\n      "verification_method": "Site hours page says \'open daily incl. holidays\'"\n    }\n  ]\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
