"""Spec for task-104-go-to-redfin-search (Real Estate)."""
from ..framework.constraints import (
    JSONSchemaConforms, ListLengthInRange, AllURLsMatch,
    AllValuesBetween, ContainsAllSubstrings, FieldPresent, Custom,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Go to Redfin (https://www.redfin.com/) and search for single-family homes "
    "for sale in Austin, TX with 3+ bedrooms, 2+ bathrooms, and a price between "
    "$400,000 and $600,000.  Identify the five listings with the most days on "
    "market.  For each property, record: address, list price, days on market, "
    "square footage, price per square foot, and the listing agent's name.  Then "
    "look up each property's assessed value and tax history on the Travis County "
    "Appraisal District website (https://www.traviscad.org/).  Finally, calculate "
    "the list-price-to-assessed-value ratio for each property and produce a "
    "buyer's market analysis table with columns: Address, List Price, Assessed "
    "Value, L/A Ratio, Days on Market, Price/sqft, and a brief 'opportunity "
    "signal' note (e.g. 'listed below assessed value' or 'overpriced vs assessment')."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("filters_applied.city",         "string",       "Should be 'Austin'."),
        RequiredField("filters_applied.beds_min",     "integer",      ""),
        RequiredField("filters_applied.baths_min",    "integer",      ""),
        RequiredField("filters_applied.price_min",    "integer",      ""),
        RequiredField("filters_applied.price_max",    "integer",      ""),
        RequiredField("filters_applied.property_type","string",       "'single-family'."),
        RequiredField("listings",                     "list[object]", "5 listings, most-days-on-market first."),
        RequiredField("listings[].address",           "string",       ""),
        RequiredField("listings[].list_price",        "integer",      "USD."),
        RequiredField("listings[].days_on_market",    "integer",      ""),
        RequiredField("listings[].sqft",              "integer",      ""),
        RequiredField("listings[].price_per_sqft",    "number",       ""),
        RequiredField("listings[].agent_name",        "string",       ""),
        RequiredField("listings[].redfin_url",        "string",       ""),
        RequiredField("listings[].traviscad_url",     "string",       ""),
        RequiredField("listings[].assessed_value",    "integer",      "From TravisCAD."),
        RequiredField("listings[].la_ratio",          "number",       "list_price / assessed_value."),
        RequiredField("listings[].opportunity_signal","string",       ""),
    ],
    optional=[
        OptionalField("listings[].year_built",        "integer",      ""),
        OptionalField("listings[].tax_history",       "list[object]", "Per-year tax payments."),
        OptionalField("listings[].photos_url",        "string",       ""),
        OptionalField("methodology_notes",            "string",       ""),
    ],
    examples={
        "filters_applied": {"city": "Austin", "beds_min": 3, "baths_min": 2,
                             "price_min": 400000, "price_max": 600000,
                             "property_type": "single-family"},
        "listings": [{
            "address": "1234 Oak St, Austin, TX",
            "list_price": 549000, "days_on_market": 142, "sqft": 2100,
            "price_per_sqft": 261.4, "agent_name": "Jane Doe",
            "redfin_url": "https://www.redfin.com/TX/Austin/1234-Oak-St-78704/home/...",
            "traviscad_url": "https://search.traviscad.org/Property/View/...",
            "assessed_value": 575000, "la_ratio": 0.955,
            "opportunity_signal": "listed below assessed value",
        }],
    },
)


def _la_ratio_consistent(summary, ctx):
    bad = []
    for i, l in enumerate(summary.get("listings") or []):
        lp, av, lar = l.get("list_price"), l.get("assessed_value"), l.get("la_ratio")
        if None in (lp, av, lar) or av == 0:
            continue
        expected = float(lp) / float(av)
        if abs(expected - float(lar)) > 0.05:
            bad.append(f"item[{i}] computed={expected:.3f}, reported={lar:.3f}")
    if bad:
        return False, f"L/A ratio off: {bad[:3]}"
    return True, "all L/A ratios within ±0.05 of list/assessed"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ContainsAllSubstrings("filters_applied.city", ["austin"], name="city_is_austin"),
    ListLengthInRange("listings", 5, 5, name="exactly_5_listings"),
    AllValuesBetween("listings", "list_price", 400000, 600000, name="all_in_400k_600k"),
    AllURLsMatch("listings", "redfin_url", r"redfin\.com", name="redfin_urls_canonical"),
    AllURLsMatch("listings", "traviscad_url", r"traviscad\.(org|com)|search\.traviscad",
                  name="traviscad_urls_present"),
    Custom("la_ratio_matches_inputs", _la_ratio_consistent),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for i, l in enumerate((summary.get("listings") or [])[:5]):
        ru = l.get("redfin_url")
        if ru:
            out.append({"url": ru, "claim": (l.get("address") or "")[:25]})
    return out


DIMENSIONS = ["redfin_search", "property_data_completeness", "assessed_value_crossref", "la_ratio_and_signals"]
DIMENSION_WEIGHTS = {
    "redfin_search":              0.25,
    "property_data_completeness": 0.25,
    "assessed_value_crossref":    0.30,
    "la_ratio_and_signals":       0.20,
}

TASK_RUBRIC = """A. Redfin Search (0.25)
  5 — Filters applied (3+BR/2+BA/400-600k/SFH/Austin); top-5 most DOM identified.
  4 — Filters applied, ordering slightly off.
  3 — Most filters applied.
  2 — Generic search.
  1 — No Redfin search.

B. Property Data Completeness (0.25)
  5 — Every listing has address/price/DOM/sqft/$/sqft/agent.
  4 — 1 field missing on some.
  3 — Several missing.
  2 — Half missing.
  1 — Mostly missing.

C. Assessed Value Crossref (0.30)
  5 — Every listing crossed with TravisCAD assessed value + tax history.
  4 — All assessed values; tax history brief.
  3 — Most listings.
  2 — Few.
  1 — None.

D. L/A Ratio + Signals (0.20)
  5 — Ratios correct + actionable opportunity signals per listing.
  4 — Ratios correct, signals generic.
  3 — Ratios approximate.
  2 — Ratios missing.
  1 — No analysis.
"""
