"""Spec for task-32-help-me-choose-the-perfect (Shopping, Amazon JP, JPY)."""
from ..framework.constraints import (
    JSONSchemaConforms, ListLengthInRange, AllValuesBetween,
    AllURLsMatch, ContainsAllSubstrings, EqualsValue, FieldPresent,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Help me choose the perfect birthday gift for my 28-year-old male friend. "
    "He likes gaming, computers and gadgets. The budget should be ¥10,000-¥15,000, "
    "and it should be something you can buy on Amazon. Pick 3 different options and "
    "add them to your shopping cart."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("platform",               "string",       "Should be 'amazon.co.jp'."),
        RequiredField("recipient_profile",      "object",       "Snapshot of the recipient used."),
        RequiredField("recipient_profile.age",  "integer",      ""),
        RequiredField("recipient_profile.interests", "list[string]", ""),
        RequiredField("options",                "list[object]", "Exactly 3 chosen items."),
        RequiredField("options[].name",         "string",       "Product name."),
        RequiredField("options[].url",          "string",       "Amazon.co.jp product URL."),
        RequiredField("options[].asin",         "string",       "Amazon ASIN."),
        RequiredField("options[].price_jpy",    "integer",      "Price in JPY (yen, no symbol)."),
        RequiredField("options[].category",     "string",       "Category fit ('gaming'|'computers'|'gadgets')."),
        RequiredField("options[].fit_notes",    "string",       "Why it fits the profile."),
        RequiredField("options[].cart_action",  "string",       "'added'|'attempted'|'not_added' — be honest."),
        RequiredField("cart_url_or_screenshot", "string",       "Cart URL or path to cart screenshot."),
    ],
    optional=[
        OptionalField("options[].rating",       "number 1-5",    "Review rating."),
        OptionalField("options[].review_count", "integer",       ""),
        OptionalField("options[].prime_eligible", "boolean",     ""),
        OptionalField("options[].delivery_estimate", "string",   ""),
        OptionalField("budget_used_jpy",        "integer",       "Sum of the 3 prices."),
    ],
    examples={
        "platform": "amazon.co.jp",
        "recipient_profile": {"age": 28, "interests": ["gaming", "computers", "gadgets"]},
        "options": [
            {"name": "...", "url": "https://www.amazon.co.jp/dp/B0...", "asin": "B0XXXX",
             "price_jpy": 12800, "category": "gaming", "fit_notes": "...", "cart_action": "added"}
        ],
        "cart_url_or_screenshot": "https://www.amazon.co.jp/gp/cart/view.html",
    },
)


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    EqualsValue("platform", "amazon.co.jp", case_insensitive=True, name="platform_is_amazon_jp"),
    ListLengthInRange("options", 3, 3, name="exactly_3_options"),
    AllValuesBetween("options", "price_jpy", 10000, 15000, name="all_prices_in_10k_15k_jpy"),
    AllURLsMatch("options", "url", r"amazon\.co\.jp", name="all_urls_amazon_jp"),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for opt in (summary.get("options") or [])[:3]:
        url = (opt or {}).get("url")
        if url:
            # Verify ASIN appears in page text
            asin = (opt.get("asin") or "")
            out.append({"url": url, "claim": asin or (opt.get("name") or "")[:20]})
    return out


DIMENSIONS = ["amazon_search_execution", "budget_interest_fit", "cart_addition", "option_diversity"]
DIMENSION_WEIGHTS = {
    "amazon_search_execution": 0.20,
    "budget_interest_fit":     0.30,
    "cart_addition":           0.35,
    "option_diversity":        0.15,
}

TASK_RUBRIC = """A. Amazon Search Execution (0.20)
  5 — Amazon.co.jp with real product pages reached.
  4 — Amazon.co.jp but indirect.
  3 — Amazon other locale.
  2 — Not Amazon.
  1 — No site.

B. Budget & Interest Fit (0.30)
  5 — All 3 in 10,000-15,000 JPY AND match gaming/computers/gadgets.
  4 — All in budget, weak fit on 1.
  3 — 2 of 3 fit.
  2 — 1 fits.
  1 — None fit.

C. Cart Addition (0.35)
  5 — All 3 added to cart with confirmation (URL/screenshot).
  4 — All 3 added but no confirmation visible.
  3 — 1-2 added.
  2 — Claimed but unverified.
  1 — Not attempted.

D. Option Diversity (0.15)
  5 — 3 distinct categories OR 3 distinct product types.
  4 — Some overlap.
  3 — 2 similar items.
  2 — All similar.
  1 — Duplicates.
"""
