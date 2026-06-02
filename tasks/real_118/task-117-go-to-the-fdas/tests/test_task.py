"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
task_117_fda.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "task_117_fda"


TASK_INSTRUCTION = "Go to the FDA's drug database Drugs@FDA (https://www.accessdata.fda.gov/scripts/cder/daf/) and look up the approval history, approved indications, and prescribing information for three GLP-1 receptor agonists: semaglutide (Ozempic/Wegovy), liraglutide (Victoza/Saxenda), and tirzepatide (Mounjaro/Zepbound). For each drug, record: brand names, approval dates for each indication, manufacturer, and the boxed warning (if any) from the prescribing information. Then go to Drugs.com (https://www.drugs.com/) and look up the most common adverse effects and significant drug interactions for each. Finally, check the FDA MedWatch Adverse Event Reporting System (FAERS) public dashboard (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/faers-public-dashboard) for the total number of adverse event reports for each drug. Compile a clinical risk summary table with columns: Drug, Brand Names, Indication, Approval Date, Boxed Warning, Top 3 Adverse Effects, Key Drug Interactions, FAERS Report Count.\n<!-- agentic_judge.grounded addendum -->\n\n---\n\n## Required output format\n\nAfter completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.\n\n**Required fields:**\n\n- `drugs` (list[object]) — Exactly 3 GLP-1 RAs.\n- `drugs[].generic_name` (string) — semaglutide / liraglutide / tirzepatide.\n- `drugs[].brand_names` (list[string]) — \n- `drugs[].indications` (list[object]) — Per-indication approval.\n- `drugs[].indications[].name` (string) — \n- `drugs[].indications[].approval_date` (string) — ISO YYYY-MM-DD.\n- `drugs[].manufacturer` (string) — \n- `drugs[].boxed_warning` (string) — Verbatim or paraphrased; empty string if none.\n- `drugs[].adverse_effects_top3` (list[string]) — \n- `drugs[].key_drug_interactions` (list[string]) — \n- `drugs[].faers_report_count` (integer) — \n- `drugs[].sources_cited` (list[string]) — URLs you read for this row.\n\n**Optional but graded if present:**\n\n- `drugs[].fda_application_number` (string) — \n- `drugs[].drugscom_url` (string) — \n- `drugs[].faers_dashboard_url` (string) — \n- `methodology_notes` (string) — \n\n**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.\n"


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
