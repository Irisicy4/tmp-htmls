"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_15_douban.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_15_douban"


TASK_INSTRUCTION = 'Help me get all 2025 publication info from https://m.douban.com/subject_collection/ECNA7Y7GA and save as an Excel file.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `source_url` (string) — Exact Douban subject_collection URL used.\n- `output_file.path` (string) — Absolute path to the .xlsx file you created.\n- `output_file.format` (string) — \'xlsx\' or \'xls\'.\n- `output_file.sheet_name` (string) — Sheet name used.\n- `output_file.row_count` (integer) — Number of data rows (excluding header).\n- `output_file.column_headers` (list[string]) — Column headers in order.\n- `publications` (list[object]) — All 2025 publications, structured.\n- `publications[].title` (string) — Title.\n- `publications[].year` (integer) — Publication year — must be 2025.\n\n**Optional but graded if present:**\n\n- `publications[].author` (string) — Author.\n- `publications[].rating` (number 0-10) — Douban rating.\n- `publications[].genre` (string) — \n- `publications[].url` (string) — Per-item URL on Douban.\n- `filter_methodology` (string) — How you confirmed only 2025 items were kept.\n- `total_items_seen` (integer) — Items seen on the page before year-filtering.\n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "source_url": "https://m.douban.com/subject_collection/ECNA7Y7GA",\n  "output_file": {\n    "path": "/output/douban_2025.xlsx",\n    "format": "xlsx",\n    "sheet_name": "publications",\n    "row_count": 14,\n    "column_headers": [\n      "title",\n      "year",\n      "author",\n      "rating"\n    ]\n  },\n  "publications": [\n    {\n      "title": "...",\n      "year": 2025,\n      "author": "..."\n    }\n  ]\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
