import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to the FDA's Drugs@FDA database and look up the approval history, approved indications, "
    "and prescribing information for three GLP-1 receptor agonists: semaglutide (Ozempic/Wegovy), "
    "liraglutide (Victoza/Saxenda), and tirzepatide (Mounjaro/Zepbound). For each drug, record: "
    "brand names, approval dates for each indication, manufacturer, and the boxed warning (if any) "
    "from the prescribing information. Then go to Drugs.com and look up the most common adverse "
    "effects and significant drug interactions for each. Finally, check the FDA MedWatch FAERS "
    "public dashboard for the total number of adverse event reports for each drug. Compile a "
    "clinical risk summary table with all eight specified columns.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: FDA Drugs@FDA (accessdata.fda.gov/scripts/cder/daf/), Drugs.com, AND FDA FAERS dashboard — all three required
- Drugs: semaglutide, liraglutide, AND tirzepatide — all three
- FDA fields: brand names, approval date(s) per indication, manufacturer, boxed warning text/description
- Drugs.com fields: top 3 adverse effects, significant drug interactions
- FAERS: total adverse event report count per drug
- Output: Eight-column table for all three drugs

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Drugs@FDA and look up approval data for all three drugs?
- Are brand names, approval dates, and boxed warnings retrieved for each drug?
- Did the agent access Drugs.com for adverse effects and interactions?
- Did the agent check the FAERS dashboard for adverse event report counts?
- Is there a complete eight-column table?

### Step 2: Dimension Scoring

#### A. Drugs@FDA Navigation & Approval Data
Did the agent navigate Drugs@FDA and retrieve specific approval and labeling data?

5 — Navigated accessdata.fda.gov/scripts/cder/daf/ for all three drugs; retrieved brand names, approval dates (YYYY-MM-DD), manufacturer name, and boxed warning description for each.
4 — Navigated Drugs@FDA for two of three drugs with full data; third drug's data is from a secondary source or has a missing field.
3 — Navigated Drugs@FDA and retrieved approval dates and brand names for all three drugs; boxed warning present for at least two drugs.
2 — Drugs@FDA mentioned; data retrieved for one drug only or data appears to be from prior knowledge.
1 — Drugs@FDA not navigated; all drug approval data is from prior knowledge.

#### B. FDA Data Accuracy
Are brand names, approval dates, manufacturers, and boxed warnings correct for all three drugs?

5 — All three drugs have correct brand names (semaglutide→Ozempic/Wegovy, liraglutide→Victoza/Saxenda, tirzepatide→Mounjaro/Zepbound), plausible approval dates (YYYY format), correct manufacturers (Novo Nordisk, Eli Lilly), and boxed warning descriptions.
4 — Two of three drugs have fully accurate data; one has a minor error (e.g., missing one brand name).
3 — Brand names and manufacturers correct for most drugs; approval dates are year-level only; boxed warning vague for one drug.
2 — Some correct brand names but manufacturer or approval dates are wrong or missing for multiple drugs.
1 — Data is generic or mostly fabricated; no specific approval dates or boxed warning text.

#### C. Drugs.com & FAERS Data
Did the agent retrieve adverse effects and interactions from Drugs.com AND FAERS report counts?

5 — Drugs.com visited; top 3 adverse effects and significant drug interactions retrieved for all three drugs. FAERS dashboard checked; specific adverse event report counts found for all three.
4 — Drugs.com data found for all three drugs; FAERS count found for two of three drugs.
3 — Drugs.com data for at least two drugs; FAERS referenced but counts approximate or from one drug only.
2 — Adverse effects cited for all three drugs from Drugs.com or similar; FAERS not checked or no specific counts.
1 — Neither Drugs.com nor FAERS navigated; all adverse effect and report count data is from prior knowledge.

#### D. Output Table Completeness
Is the clinical risk summary table complete with all eight columns for all three drugs?

5 — Complete eight-column table (Drug, Brand Names, Indication, Approval Date, Boxed Warning, Top 3 Adverse Effects, Key Drug Interactions, FAERS Report Count) for all three drugs.
4 — Table with seven of eight columns for all three drugs.
3 — Table with six or fewer columns or one drug missing from the table.
2 — Partial table with fewer than five columns.
1 — No structured table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "fda_navigation": <1-5>,
  "fda_data_accuracy": <1-5>,
  "drugscom_faers_data": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "fda_navigation": "<one sentence citing specific evidence>",
    "fda_data_accuracy": "<one sentence citing specific evidence>",
    "drugscom_faers_data": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "fda_navigation":     0.25,
    "fda_data_accuracy":  0.30,
    "drugscom_faers_data": 0.25,
    "output_table":       0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
