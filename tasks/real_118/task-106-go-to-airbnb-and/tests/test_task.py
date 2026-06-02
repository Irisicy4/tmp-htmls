"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_106_airbnb.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_106_airbnb"


TASK_INSTRUCTION = 'Go to Airbnb (https://www.airbnb.com/) and search for entire-home listings in Lisbon, Portugal with a capacity of at least 4 guests. Find three active listings with 10+ reviews and record: listing title, nightly rate, occupancy-implied availability (count available nights in the next 30 days from the calendar), number of reviews, and average review score. Then search for the same property type on VRBO (https://www.vrbo.com/) in the same city and find two comparable listings. Finally, visit AirDNA\'s free market overview (https://www.airdna.co/vacation-rental-data/app/pt/lisbon/lisbon/overview) to note the published average daily rate and occupancy rate for Lisbon. Using all gathered data, produce a vacation rental ROI analysis table comparing each listing on: Platform, Nightly Rate, Estimated Monthly Revenue (nightly rate × occupancy × 30), Review Score, and an ROI feasibility comment.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `airbnb_listings` (list[object]) — Exactly 3 Airbnb listings.\n- `airbnb_listings[].title` (string) — \n- `airbnb_listings[].url` (string) — Canonical airbnb.com URL.\n- `airbnb_listings[].nightly_rate_usd` (number) — USD/night.\n- `airbnb_listings[].available_nights_30d` (integer) — Next-30-day available nights.\n- `airbnb_listings[].review_count` (integer) — ≥10.\n- `airbnb_listings[].review_score` (number) — Out of 5 (Airbnb) or 10 (VRBO).\n- `vrbo_listings` (list[object]) — Exactly 2 VRBO listings.\n- `vrbo_listings[].title` (string) — \n- `vrbo_listings[].url` (string) — Canonical vrbo.com URL.\n- `vrbo_listings[].nightly_rate_usd` (number) — \n- `airdna.avg_daily_rate_usd` (number) — \n- `airdna.occupancy_rate` (number 0-1) — Decimal fraction.\n- `airdna.source_url` (string) — \n- `revenue_table` (list[object]) — ROI rows for the 5 listings.\n- `revenue_table[].platform` (string) — \'airbnb\'|\'vrbo\'.\n- `revenue_table[].monthly_revenue_usd` (number) — \n- `revenue_table[].roi_comment` (string) — \n\n**Optional but graded if present:**\n\n- `airbnb_listings[].guests_capacity` (integer) — \n- `airbnb_listings[].entire_home` (boolean) — \n- `vrbo_listings[].review_count` (integer) — \n- `vrbo_listings[].review_score` (number) — \n- `methodology_notes` (string) — \n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n\n**Example shape (values illustrative, not literal):**\n\n```json\n{\n  "airbnb_listings": [],\n  "vrbo_listings": [],\n  "airdna": {\n    "avg_daily_rate_usd": 162,\n    "occupancy_rate": 0.72,\n    "source_url": "https://www.airdna.co/vacation-rental-data/app/pt/lisbon/lisbon/overview"\n  },\n  "revenue_table": []\n}\n```\n'


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
