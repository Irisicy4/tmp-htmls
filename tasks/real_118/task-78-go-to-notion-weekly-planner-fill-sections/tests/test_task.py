"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_78_notion.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_78_notion"


TASK_INSTRUCTION = 'Go to https://www.notion.so/templates/weekly-planner and open the Weekly Planner template. Fill in all the empty sections of the planner with appropriate placeholder data — including goals, daily tasks, notes, and any other incomplete fields — to make the page fully complete.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `template_url` (string) — Should be the Notion weekly planner template URL.\n- `sections_filled` (list[object]) — Each section you filled in.\n- `sections_filled[].name` (string) — Section label as shown on the planner.\n- `sections_filled[].content_preview` (string) — First ~120 chars of the content you filled in.\n- `sections_filled[].placeholder_or_real` (string) — \'placeholder\' (fake but realistic) or \'real\'.\n- `completion_evidence` (string) — URL to your filled copy OR path to a screenshot.\n\n**Optional but graded if present:**\n\n- `sections_total` (integer) — Sections found on the template, filled or not.\n- `sections_skipped` (list[string]) — Names of sections you couldn\'t access/fill.\n- `editing_method` (string) — How you edited (duplicate template + edit, fork, etc.).\n- `notion_account_used` (boolean) — Did you sign into a Notion account?\n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "template_url": "https://www.notion.so/templates/weekly-planner",\n  "sections_filled": [\n    {\n      "name": "Goals",\n      "content_preview": "1. Ship Q3 OKR ...",\n      "placeholder_or_real": "placeholder"\n    },\n    {\n      "name": "Monday tasks",\n      "content_preview": "Standup @ 9, ...",\n      "placeholder_or_real": "placeholder"\n    }\n  ],\n  "completion_evidence": "/output/notion_filled.png"\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
