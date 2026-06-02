"""Spec for task-106-go-to-airbnb-and (Travel)."""
from ..framework.constraints import (
    JSONSchemaConforms, ListLengthInRange, AllURLsMatch, FieldPresent, Custom,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Go to Airbnb (https://www.airbnb.com/) and search for entire-home listings "
    "in Lisbon, Portugal with a capacity of at least 4 guests. Find three active "
    "listings with 10+ reviews and record: listing title, nightly rate, "
    "occupancy-implied availability (count available nights in the next 30 days "
    "from the calendar), number of reviews, and average review score. Then "
    "search for the same property type on VRBO (https://www.vrbo.com/) in the "
    "same city and find two comparable listings. Finally, visit AirDNA's free "
    "market overview "
    "(https://www.airdna.co/vacation-rental-data/app/pt/lisbon/lisbon/overview) "
    "to note the published average daily rate and occupancy rate for Lisbon. "
    "Using all gathered data, produce a vacation rental ROI analysis table "
    "comparing each listing on: Platform, Nightly Rate, Estimated Monthly Revenue "
    "(nightly rate × occupancy × 30), Review Score, and an ROI feasibility comment."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("airbnb_listings",                "list[object]", "Exactly 3 Airbnb listings."),
        RequiredField("airbnb_listings[].title",        "string",       ""),
        RequiredField("airbnb_listings[].url",          "string",       "Canonical airbnb.com URL."),
        RequiredField("airbnb_listings[].nightly_rate_usd","number",    "USD/night."),
        RequiredField("airbnb_listings[].available_nights_30d","integer","Next-30-day available nights."),
        RequiredField("airbnb_listings[].review_count", "integer",      "≥10."),
        RequiredField("airbnb_listings[].review_score", "number",       "Out of 5 (Airbnb) or 10 (VRBO)."),
        RequiredField("vrbo_listings",                  "list[object]", "Exactly 2 VRBO listings."),
        RequiredField("vrbo_listings[].title",          "string",       ""),
        RequiredField("vrbo_listings[].url",            "string",       "Canonical vrbo.com URL."),
        RequiredField("vrbo_listings[].nightly_rate_usd","number",      ""),
        RequiredField("airdna.avg_daily_rate_usd",      "number",       ""),
        RequiredField("airdna.occupancy_rate",          "number 0-1",   "Decimal fraction."),
        RequiredField("airdna.source_url",              "string",       ""),
        RequiredField("revenue_table",                  "list[object]", "ROI rows for the 5 listings."),
        RequiredField("revenue_table[].platform",       "string",       "'airbnb'|'vrbo'."),
        RequiredField("revenue_table[].monthly_revenue_usd","number",   ""),
        RequiredField("revenue_table[].roi_comment",    "string",       ""),
    ],
    optional=[
        OptionalField("airbnb_listings[].guests_capacity","integer",    ""),
        OptionalField("airbnb_listings[].entire_home",  "boolean",      ""),
        OptionalField("vrbo_listings[].review_count",   "integer",      ""),
        OptionalField("vrbo_listings[].review_score",   "number",       ""),
        OptionalField("methodology_notes",              "string",       ""),
    ],
    examples={
        "airbnb_listings": [],
        "vrbo_listings": [],
        "airdna": {"avg_daily_rate_usd": 162, "occupancy_rate": 0.72,
                     "source_url": "https://www.airdna.co/vacation-rental-data/app/pt/lisbon/lisbon/overview"},
        "revenue_table": [],
    },
)


def _all_reviews_ge_10(summary, ctx):
    bad = [i for i, l in enumerate(summary.get("airbnb_listings") or [])
            if (l.get("review_count") or 0) < 10]
    if bad:
        return False, f"items {bad} have <10 reviews"
    return True, "all Airbnb listings have ≥10 reviews"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ListLengthInRange("airbnb_listings", 3, 3, name="exactly_3_airbnb"),
    ListLengthInRange("vrbo_listings", 2, 2, name="exactly_2_vrbo"),
    AllURLsMatch("airbnb_listings", "url", r"airbnb\.", name="airbnb_urls_canonical"),
    AllURLsMatch("vrbo_listings", "url", r"vrbo\.", name="vrbo_urls_canonical"),
    Custom("airbnb_reviews_ge_10", _all_reviews_ge_10),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for l in (summary.get("airbnb_listings") or [])[:3]:
        if (u := l.get("url")):
            out.append({"url": u, "claim": (l.get("title") or "")[:25]})
    for l in (summary.get("vrbo_listings") or [])[:2]:
        if (u := l.get("url")):
            out.append({"url": u, "claim": (l.get("title") or "")[:25]})
    if (a := (summary.get("airdna") or {}).get("source_url")):
        out.append({"url": a, "claim": "Lisbon"})
    return out


DIMENSIONS = ["listing_retrieval", "listing_data_quality", "airdna_market_data", "revenue_calc_and_table"]
DIMENSION_WEIGHTS = {d: 0.25 for d in DIMENSIONS}

TASK_RUBRIC = """A. Listing Retrieval (0.25)
  5 — 3 Airbnb + 2 VRBO listings matching filters.
  4 — Counts match, one filter weak.
  3 — Counts off by 1.
  2 — Wrong city.
  1 — No listings.

B. Listing Data Quality (0.25)
  5 — Every listing has title/rate/availability/reviews/score.
  4 — 1-2 fields missing.
  3 — Several missing.
  2 — Half missing.
  1 — Most missing.

C. AirDNA Market Data (0.25)
  5 — ADR + occupancy + source URL all from AirDNA Lisbon overview.
  4 — Numbers present, source noted.
  3 — Estimated, no source.
  2 — Approximate.
  1 — Missing.

D. Revenue Calc + Table (0.25)
  5 — Monthly revenue = nightly_rate × occupancy × 30; ROI comments per row.
  4 — Calc correct, comments thin.
  3 — Calc approximate.
  2 — Missing.
  1 — None.
"""
