"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_32_amazon_jp.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_32_amazon_jp"


TASK_INSTRUCTION = 'Help me choose the perfect birthday gift for my 28-year-old male friend. He likes gaming, computers and gadgets. The budget should be 10,000-15,000 yen, and it should be something you can buy on Amazon. Pick 3 different options and add them to your shopping cart.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `platform` (string) — Should be \'amazon.co.jp\'.\n- `recipient_profile` (object) — Snapshot of the recipient used.\n- `recipient_profile.age` (integer) — \n- `recipient_profile.interests` (list[string]) — \n- `options` (list[object]) — Exactly 3 chosen items.\n- `options[].name` (string) — Product name.\n- `options[].url` (string) — Amazon.co.jp product URL.\n- `options[].asin` (string) — Amazon ASIN.\n- `options[].price_jpy` (integer) — Price in JPY (yen, no symbol).\n- `options[].category` (string) — Category fit (\'gaming\'|\'computers\'|\'gadgets\').\n- `options[].fit_notes` (string) — Why it fits the profile.\n- `options[].cart_action` (string) — \'added\'|\'attempted\'|\'not_added\' — be honest.\n- `cart_url_or_screenshot` (string) — Cart URL or path to cart screenshot.\n\n**Optional but graded if present:**\n\n- `options[].rating` (number 1-5) — Review rating.\n- `options[].review_count` (integer) — \n- `options[].prime_eligible` (boolean) — \n- `options[].delivery_estimate` (string) — \n- `budget_used_jpy` (integer) — Sum of the 3 prices.\n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "platform": "amazon.co.jp",\n  "recipient_profile": {\n    "age": 28,\n    "interests": [\n      "gaming",\n      "computers",\n      "gadgets"\n    ]\n  },\n  "options": [\n    {\n      "name": "...",\n      "url": "https://www.amazon.co.jp/dp/B0...",\n      "asin": "B0XXXX",\n      "price_jpy": 12800,\n      "category": "gaming",\n      "fit_notes": "...",\n      "cart_action": "added"\n    }\n  ],\n  "cart_url_or_screenshot": "https://www.amazon.co.jp/gp/cart/view.html"\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
