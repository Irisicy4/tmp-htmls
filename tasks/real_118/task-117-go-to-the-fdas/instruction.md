Go to the FDA's drug database Drugs@FDA (https://www.accessdata.fda.gov/scripts/cder/daf/) and look up the approval history, approved indications, and prescribing information for three GLP-1 receptor agonists: semaglutide (Ozempic/Wegovy), liraglutide (Victoza/Saxenda), and tirzepatide (Mounjaro/Zepbound). For each drug, record: brand names, approval dates for each indication, manufacturer, and the boxed warning (if any) from the prescribing information. Then go to Drugs.com (https://www.drugs.com/) and look up the most common adverse effects and significant drug interactions for each. Finally, check the FDA MedWatch Adverse Event Reporting System (FAERS) public dashboard (https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/faers-public-dashboard) for the total number of adverse event reports for each drug. Compile a clinical risk summary table with columns: Drug, Brand Names, Indication, Approval Date, Boxed Warning, Top 3 Adverse Effects, Key Drug Interactions, FAERS Report Count.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `drugs` (list[object]) — Exactly 3 GLP-1 RAs.
- `drugs[].generic_name` (string) — semaglutide / liraglutide / tirzepatide.
- `drugs[].brand_names` (list[string]) — 
- `drugs[].indications` (list[object]) — Per-indication approval.
- `drugs[].indications[].name` (string) — 
- `drugs[].indications[].approval_date` (string) — ISO YYYY-MM-DD.
- `drugs[].manufacturer` (string) — 
- `drugs[].boxed_warning` (string) — Verbatim or paraphrased; empty string if none.
- `drugs[].adverse_effects_top3` (list[string]) — 
- `drugs[].key_drug_interactions` (list[string]) — 
- `drugs[].faers_report_count` (integer) — 
- `drugs[].sources_cited` (list[string]) — URLs you read for this row.

**Optional but graded if present:**

- `drugs[].fda_application_number` (string) — 
- `drugs[].drugscom_url` (string) — 
- `drugs[].faers_dashboard_url` (string) — 
- `methodology_notes` (string) — 

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.
