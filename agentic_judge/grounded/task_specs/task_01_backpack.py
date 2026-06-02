"""Spec for task-01-im-looking-for-backpack-under (Shopping)."""
from ..framework.constraints import (
    HardConstraint, AllValuesLE, ListLengthInRange, AllURLsMatch,
    FieldPresent, JSONSchemaConforms, Custom,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "I'm looking for a backpack under $75 that has all the features of this one: "
    "https://www.amazon.com/dp/B09YRC9Y3G  Please do some research and find 3-5 "
    "options and summarize their key features and prices, comparing them to the original."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("reference_product.url",       "string",          "Canonical Amazon URL of the original backpack."),
        RequiredField("reference_product.name",      "string",          "Product name as shown on the page."),
        RequiredField("reference_product.features",  "list[string]",    "Key features pulled from the original listing."),
        RequiredField("options",                     "list[object]",    "3-5 alternative backpacks."),
        RequiredField("options[].name",              "string",          "Product name."),
        RequiredField("options[].url",               "string",          "Canonical product URL (Amazon, manufacturer, etc.)."),
        RequiredField("options[].price",             "number (USD)",    "Selling price in USD."),
        RequiredField("options[].features",          "list[string]",    "3+ feature bullets."),
        RequiredField("options[].comparison_notes",  "string",          "How it compares to the original (pros and cons)."),
    ],
    optional=[
        OptionalField("options[].asin",              "string",          "Amazon ASIN if applicable."),
        OptionalField("options[].rating",            "number 1-5",      "Review rating."),
        OptionalField("options[].review_count",      "integer",         "Number of reviews."),
        OptionalField("options[].verified_in_stock", "boolean",         "True if you saw stock indicator."),
        OptionalField("reference_product.price_estimate","number (USD)","Your estimate of the original's MSRP."),
        OptionalField("recommendation",              "string",          "Your top pick and why."),
    ],
    examples={
        "reference_product": {"url": "https://www.amazon.com/dp/B09YRC9Y3G", "name": "...", "features": ["...", "..."]},
        "options": [
            {"name": "MATEIN ...", "url": "https://www.amazon.com/dp/B07...",
             "price": 64.99, "features": ["40L expandable", "USB port"], "comparison_notes": "..."}
        ],
    },
)

# --- hard constraints --------------------------------------------------------
HARD_CONSTRAINTS: list[HardConstraint] = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ListLengthInRange("options", 3, 5, name="quantity_3_to_5"),
    AllValuesLE("options", "price", 75.0, name="all_prices_under_75_usd"),
    AllURLsMatch("options", "url", r"^https?://", name="every_option_has_url"),
    FieldPresent("reference_product.url", name="reference_url_present"),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    """Verify the reference URL plus a sample of option URLs and prices."""
    checks: list[dict] = []
    ref = (summary.get("reference_product") or {}).get("url")
    if ref:
        checks.append({"url": ref, "claim": "backpack"})  # weak check; site often blocks fetch
    for i, opt in enumerate(summary.get("options") or []):
        url = (opt or {}).get("url")
        if not url:
            continue
        # Confirm the page mentions the agent's claimed name
        claim = (opt.get("name") or "").split()[0] if opt.get("name") else ""
        checks.append({"url": url, "claim": claim})
        if i >= 2:  # cap at first 3 to limit fetch budget
            break
    return checks


DIMENSIONS = ["constraint_satisfaction", "result_specificity", "comparison_quality", "task_completeness"]
DIMENSION_WEIGHTS = {
    "constraint_satisfaction": 0.35,
    "result_specificity":      0.25,
    "comparison_quality":      0.25,
    "task_completeness":       0.15,
}

TASK_RUBRIC = """A. Constraint Satisfaction (0.35)
  5 — All 3-5 options explicitly priced ≤ $75; quantity 3-5.
  4 — All ≤ $75 but 1 option missing a price.
  3 — Quantity 3-5; one price ambiguous; none clearly > $75.
  2 — Quantity outside 3-5, OR 1 option > $75.
  1 — Multiple over budget OR task abandoned.

B. Result Specificity (0.25)
  5 — All 3-5 options have name + price + verifiable identifier (URL/ASIN/brand+model).
  4 — Most options (3+) have name+price; 1-2 lack identifier.
  3 — Named but missing prices/identifiers for 2+; distinct products.
  2 — Generic descriptions.
  1 — No specific products.

C. Comparison Quality (0.25)
  5 — Each option compared to original on 2+ specific features; structured and actionable.
  4 — Most options compared on ≥1 feature; uneven.
  3 — General comparison summary; shallow.
  2 — Comparison language without actual feature comparison.
  1 — No comparison.

D. Task Completeness (0.15)
  5 — All four components for all options: features, price, comparison, original researched.
  4 — Most components, 1 minor gap.
  3 — 2-3 components consistently; one missing.
  2 — Only 1-2 components.
  1 — Task not addressed.
"""
