"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_104_redfin.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_104_redfin"


TASK_INSTRUCTION = 'Go to Redfin (https://www.redfin.com/) and search for single-family homes for sale in Austin, TX with 3+ bedrooms, 2+ bathrooms, and a price between $400,000 and $600,000. Identify the five listings with the most days on market. For each property, record: address, list price, days on market, square footage, price per square foot, and the listing agent\'s name. Then look up each property\'s assessed value and tax history on the Travis County Appraisal District website (https://www.traviscad.org/). Finally, calculate the list-price-to-assessed-value ratio for each property and produce a buyer\'s market analysis table with columns: Address, List Price, Assessed Value, L/A Ratio, Days on Market, Price/sqft, and a brief "opportunity signal" note (e.g., "listed below assessed value" or "overpriced vs assessment").\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `filters_applied.city` (string) — Should be \'Austin\'.\n- `filters_applied.beds_min` (integer) — \n- `filters_applied.baths_min` (integer) — \n- `filters_applied.price_min` (integer) — \n- `filters_applied.price_max` (integer) — \n- `filters_applied.property_type` (string) — \'single-family\'.\n- `listings` (list[object]) — 5 listings, most-days-on-market first.\n- `listings[].address` (string) — \n- `listings[].list_price` (integer) — USD.\n- `listings[].days_on_market` (integer) — \n- `listings[].sqft` (integer) — \n- `listings[].price_per_sqft` (number) — \n- `listings[].agent_name` (string) — \n- `listings[].redfin_url` (string) — \n- `listings[].traviscad_url` (string) — \n- `listings[].assessed_value` (integer) — From TravisCAD.\n- `listings[].la_ratio` (number) — list_price / assessed_value.\n- `listings[].opportunity_signal` (string) — \n\n**Optional but graded if present:**\n\n- `listings[].year_built` (integer) — \n- `listings[].tax_history` (list[object]) — Per-year tax payments.\n- `listings[].photos_url` (string) — \n- `methodology_notes` (string) — \n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "filters_applied": {\n    "city": "Austin",\n    "beds_min": 3,\n    "baths_min": 2,\n    "price_min": 400000,\n    "price_max": 600000,\n    "property_type": "single-family"\n  },\n  "listings": [\n    {\n      "address": "1234 Oak St, Austin, TX",\n      "list_price": 549000,\n      "days_on_market": 142,\n      "sqft": 2100,\n      "price_per_sqft": 261.4,\n      "agent_name": "Jane Doe",\n      "redfin_url": "https://www.redfin.com/TX/Austin/1234-Oak-St-78704/home/...",\n      "traviscad_url": "https://search.traviscad.org/Property/View/...",\n      "assessed_value": 575000,\n      "la_ratio": 0.955,\n      "opportunity_signal": "listed below assessed value"\n    }\n  ]\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
