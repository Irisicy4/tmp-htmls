"""Spec for task-65-find-recommended-dinner-restaurants-in (Travel)."""
from ..framework.constraints import (
    JSONSchemaConforms, ContainsAllSubstrings, ListLengthInRange,
    FieldPresent, Custom,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Find recommended dinner restaurants in Tuscany, Italy that are open on "
    "December 26th and 27th."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("location",                       "string",       "Should contain 'Tuscany'."),
        RequiredField("meal_type",                      "string",       "Should be 'dinner'."),
        RequiredField("date_range",                     "list[string]", "['2025-12-26','2025-12-27'] or similar (must include both days)."),
        RequiredField("restaurants",                    "list[object]", "Recommended dinner restaurants."),
        RequiredField("restaurants[].name",             "string",       ""),
        RequiredField("restaurants[].city",             "string",       "Town within Tuscany."),
        RequiredField("restaurants[].source_url",       "string",       "TripAdvisor/Google Maps/TheFork URL."),
        RequiredField("restaurants[].open_dec_26",      "boolean",      "Open dinner Dec 26?"),
        RequiredField("restaurants[].open_dec_27",      "boolean",      "Open dinner Dec 27?"),
        RequiredField("restaurants[].verification_method","string",     "How you verified the hours."),
    ],
    optional=[
        OptionalField("restaurants[].cuisine",          "string",       ""),
        OptionalField("restaurants[].price_band",       "string",       "$/$$/$$$/$$$$."),
        OptionalField("restaurants[].rating",           "number 0-5",   ""),
        OptionalField("restaurants[].reservation_url",  "string",       ""),
        OptionalField("sources_used",                   "list[string]", "All discovery platforms consulted."),
    ],
    examples={
        "location": "Tuscany, Italy", "meal_type": "dinner",
        "date_range": ["2025-12-26", "2025-12-27"],
        "restaurants": [
            {"name": "Trattoria ...", "city": "Florence",
             "source_url": "https://www.tripadvisor.com/...",
             "open_dec_26": True, "open_dec_27": True,
             "verification_method": "Site hours page says 'open daily incl. holidays'"}
        ],
    },
)


def _both_dates_present(summary, ctx):
    dr = summary.get("date_range") or []
    s = " ".join(str(x) for x in dr).lower()
    have_26 = "26" in s and ("dec" in s or "12-26" in s or "12/26" in s)
    have_27 = "27" in s and ("dec" in s or "12-27" in s or "12/27" in s)
    if have_26 and have_27:
        return True, "date_range covers both Dec 26 AND Dec 27"
    return False, f"date_range={dr!r} missing one of Dec 26 or Dec 27"


def _restaurants_open_both(summary, ctx):
    rs = summary.get("restaurants") or []
    bad = [i for i, r in enumerate(rs)
           if not (r.get("open_dec_26") and r.get("open_dec_27"))]
    if bad:
        return False, f"items {bad} not confirmed open on BOTH days"
    return True, f"all {len(rs)} restaurants confirmed open on both days"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ContainsAllSubstrings("location", ["tuscany"], name="location_in_tuscany"),
    ContainsAllSubstrings("meal_type", ["dinner"], name="meal_type_dinner"),
    Custom("date_range_covers_dec_26_and_27", _both_dates_present),
    Custom("all_restaurants_open_both_days", _restaurants_open_both),
    ListLengthInRange("restaurants", 1, 20, name="at_least_one_restaurant"),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for r in (summary.get("restaurants") or [])[:5]:
        if (u := r.get("source_url")):
            out.append({"url": u, "claim": (r.get("name") or "")[:25]})
    return out


DIMENSIONS = ["search_execution", "date_verification", "recommendation_quality", "dinner_focus"]
DIMENSION_WEIGHTS = {
    "search_execution":       0.2,
    "date_verification":      0.35,
    "recommendation_quality": 0.3,
    "dinner_focus":           0.15,
}

TASK_RUBRIC = """A. Search Execution (0.20)
  5 — Used a credible platform (TripAdvisor, Google Maps, TheFork) for every result.
  4 — Mostly credible.
  3 — Mixed sources.
  2 — Blogs/reviews only.
  1 — No source.

B. Date Verification (0.35)
  5 — Each restaurant verified open BOTH Dec 26 AND Dec 27 with evidence.
  4 — Open on both, evidence weaker.
  3 — One day verified, the other assumed.
  2 — Dates not addressed.
  1 — Dates ignored.

C. Recommendation Quality (0.30)
  5 — Diverse picks (towns/cuisines/price bands); 4+ recommendations.
  4 — Solid; 3 recommendations.
  3 — Adequate; less diversity.
  2 — Generic.
  1 — Off-topic.

D. Dinner Focus (0.15)
  5 — Explicitly confirmed dinner hours.
  4 — Implied dinner.
  3 — Mixed.
  2 — Lunch-only mistakenly included.
  1 — Wrong meal.
"""
