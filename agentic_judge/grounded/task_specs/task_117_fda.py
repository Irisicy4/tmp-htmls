"""Spec for task-117-go-to-the-fdas (Medical / Clinical)."""
from ..framework.constraints import (
    JSONSchemaConforms, ListLengthInRange, ContainsAllSubstrings,
    AllURLsMatch, FieldPresent, Custom,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = (
    "Go to the FDA's drug database Drugs@FDA "
    "(https://www.accessdata.fda.gov/scripts/cder/daf/) and look up the approval "
    "history, approved indications, and prescribing information for three GLP-1 "
    "receptor agonists: semaglutide (Ozempic/Wegovy), liraglutide (Victoza/Saxenda), "
    "and tirzepatide (Mounjaro/Zepbound).  For each drug, record: brand names, "
    "approval dates for each indication, manufacturer, and the boxed warning (if "
    "any) from the prescribing information.  Then go to Drugs.com "
    "(https://www.drugs.com/) and look up the most common adverse effects and "
    "significant drug interactions for each.  Finally, check the FDA MedWatch "
    "Adverse Event Reporting System (FAERS) public dashboard "
    "(https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-"
    "system-faers/faers-public-dashboard) for the total number of adverse event "
    "reports for each drug.  Compile a clinical risk summary table with columns: "
    "Drug, Brand Names, Indication, Approval Date, Boxed Warning, Top 3 Adverse "
    "Effects, Key Drug Interactions, FAERS Report Count."
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("drugs",                             "list[object]", "Exactly 3 GLP-1 RAs."),
        RequiredField("drugs[].generic_name",              "string",       "semaglutide / liraglutide / tirzepatide."),
        RequiredField("drugs[].brand_names",               "list[string]", ""),
        RequiredField("drugs[].indications",               "list[object]", "Per-indication approval."),
        RequiredField("drugs[].indications[].name",        "string",       ""),
        RequiredField("drugs[].indications[].approval_date","string",      "ISO YYYY-MM-DD."),
        RequiredField("drugs[].manufacturer",              "string",       ""),
        RequiredField("drugs[].boxed_warning",             "string",       "Verbatim or paraphrased; empty string if none."),
        RequiredField("drugs[].adverse_effects_top3",      "list[string]", ""),
        RequiredField("drugs[].key_drug_interactions",     "list[string]", ""),
        RequiredField("drugs[].faers_report_count",        "integer",      ""),
        RequiredField("drugs[].sources_cited",             "list[string]", "URLs you read for this row."),
    ],
    optional=[
        OptionalField("drugs[].fda_application_number",    "string",       ""),
        OptionalField("drugs[].drugscom_url",              "string",       ""),
        OptionalField("drugs[].faers_dashboard_url",       "string",       ""),
        OptionalField("methodology_notes",                 "string",       ""),
    ],
)


def _three_glp1s(summary, ctx):
    expected = {"semaglutide", "liraglutide", "tirzepatide"}
    found = {(d.get("generic_name") or "").lower() for d in (summary.get("drugs") or [])}
    miss = expected - found
    if miss:
        return False, f"missing: {sorted(miss)}"
    return True, "all three drugs present"


def _sources_per_row(summary, ctx):
    bad = []
    for i, d in enumerate(summary.get("drugs") or []):
        srcs = d.get("sources_cited") or []
        if len(srcs) < 2:
            bad.append(f"drug[{i}] cites only {len(srcs)} sources")
    if bad:
        return False, "; ".join(bad)
    return True, "every drug cites ≥2 sources"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ListLengthInRange("drugs", 3, 3, name="exactly_3_drugs"),
    Custom("all_three_glp1_drugs_present", _three_glp1s),
    Custom("each_drug_has_two_plus_sources", _sources_per_row),
]


def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for d in (summary.get("drugs") or [])[:3]:
        for u in (d.get("sources_cited") or [])[:2]:
            out.append({"url": u, "claim": (d.get("generic_name") or "")[:15]})
    return out


DIMENSIONS = ["fda_navigation", "fda_data_accuracy", "drugscom_faers_data", "output_table"]
DIMENSION_WEIGHTS = {
    "fda_navigation":      0.25,
    "fda_data_accuracy":   0.30,
    "drugscom_faers_data": 0.25,
    "output_table":        0.20,
}

TASK_RUBRIC = """A. FDA Navigation (0.25)
  5 — Drugs@FDA reached for all three drugs.
  4 — Reached for most; one partial.
  3 — Reached but only one drug.
  2 — Indirect (search instead of Drugs@FDA).
  1 — No FDA.

B. FDA Data Accuracy (0.30)
  5 — Brand names + indications + approval dates + manufacturer + boxed warning all verifiable.
  4 — Minor inaccuracy on one field.
  3 — Several fields wrong.
  2 — Most fields wrong.
  1 — Fabricated.

C. Drugs.com + FAERS Data (0.25)
  5 — Top-3 adverse effects + key interactions per drug + FAERS counts.
  4 — Mostly complete.
  3 — Some drugs missing data.
  2 — Few drugs.
  1 — None.

D. Output Table (0.20)
  5 — Eight-column table for all three drugs.
  4 — One column missing.
  3 — Several columns missing.
  2 — Unstructured prose.
  1 — None.
"""
