"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_01_backpack.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_01_backpack"


TASK_INSTRUCTION = 'im looking for a backpack under $75 that has all the features of this one: https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 optionsand summarize their key features and prices, comparing them to the original.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `reference_product.url` (string) — Canonical Amazon URL of the original backpack.\n- `reference_product.name` (string) — Product name as shown on the page.\n- `reference_product.features` (list[string]) — Key features pulled from the original listing.\n- `options` (list[object]) — 3-5 alternative backpacks.\n- `options[].name` (string) — Product name.\n- `options[].url` (string) — Canonical product URL (Amazon, manufacturer, etc.).\n- `options[].price` (number (USD)) — Selling price in USD.\n- `options[].features` (list[string]) — 3+ feature bullets.\n- `options[].comparison_notes` (string) — How it compares to the original (pros and cons).\n\n**Optional but graded if present:**\n\n- `options[].asin` (string) — Amazon ASIN if applicable.\n- `options[].rating` (number 1-5) — Review rating.\n- `options[].review_count` (integer) — Number of reviews.\n- `options[].verified_in_stock` (boolean) — True if you saw stock indicator.\n- `reference_product.price_estimate` (number (USD)) — Your estimate of the original\'s MSRP.\n- `recommendation` (string) — Your top pick and why.\n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "reference_product": {\n    "url": "https://www.amazon.com/dp/B09YRC9Y3G",\n    "name": "...",\n    "features": [\n      "...",\n      "..."\n    ]\n  },\n  "options": [\n    {\n      "name": "MATEIN ...",\n      "url": "https://www.amazon.com/dp/B07...",\n      "price": 64.99,\n      "features": [\n        "40L expandable",\n        "USB port"\n      ],\n      "comparison_notes": "..."\n    }\n  ]\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
