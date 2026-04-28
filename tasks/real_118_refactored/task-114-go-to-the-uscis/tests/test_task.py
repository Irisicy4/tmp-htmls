import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to the USCIS H-1B Employer Data Hub and look up the top five technology companies by "
    "H-1B petition approvals in fiscal year 2023. Record each company's name, total approvals, "
    "total denials, and denial rate. Then visit the careers page and Glassdoor profile for each "
    "of those five companies and note: number of open software engineering roles (as of today), "
    "average Glassdoor salary for Software Engineer, and whether the company explicitly mentions "
    "visa sponsorship in their job postings. Produce a visa-sponsorship employer comparison table "
    "with all specified columns including an 'H-1B friendliness score' (1-5).")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: USCIS H-1B Employer Data Hub, company careers pages, AND Glassdoor — all three required
- USCIS: Top 5 tech companies by H-1B approvals in FY2023; must use the actual data hub
- Per company: approvals count, denials count, denial rate (calculated as denials/petitions)
- Careers page: number of open SWE roles (current count)
- Glassdoor: average SWE salary + visa sponsorship mentioned in job postings (yes/no)
- Output: Seven-column table with H-1B friendliness score (1-5) based on approval volume and denial rate

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate the USCIS H-1B Employer Data Hub and retrieve FY2023 data?
- Are the top 5 tech companies plausible H-1B sponsors (e.g., Amazon, Infosys, TCS, Cognizant, Wipro, Google, Microsoft)?
- Did the agent visit company careers pages or Glassdoor for SWE data?
- Is there a complete table with all seven columns and a friendliness score?

### Step 2: Dimension Scoring

#### A. USCIS Data Hub Navigation
Did the agent navigate the USCIS H-1B Employer Data Hub and retrieve FY2023 petition data?

5 — Navigated USCIS Employer Data Hub, found top 5 tech companies by FY2023 approvals with specific approval and denial counts.
4 — Navigated USCIS Data Hub but retrieved data for four of five companies, or used FY2022 data instead of FY2023.
3 — Referenced USCIS data with plausible company names but counts appear approximated or from secondary sources.
2 — Company names are plausible H-1B sponsors but denial rates and approval counts appear fabricated or estimated.
1 — No USCIS navigation; companies and data are entirely from prior knowledge.

#### B. H-1B Data Accuracy
Are approval counts, denial counts, and denial rates specific and internally consistent?

5 — All five companies have specific approval counts, denial counts, and a calculated denial rate (denials/total petitions) that is internally consistent.
4 — Four of five companies have complete and consistent data; one is missing a denial count.
3 — All five companies named with approximate approval counts; denial rates estimated rather than calculated.
2 — Only two or three companies have specific counts; others are generic.
1 — No specific petition data; all figures are generic or fabricated.

#### C. Careers & Glassdoor Research
Did the agent visit careers pages and Glassdoor for current SWE data?

5 — Careers pages and/or Glassdoor visited for all five companies; open SWE role count, average SWE salary, and visa sponsorship mention (yes/no) reported for each.
4 — Careers/Glassdoor data found for four of five companies; one is missing one field.
3 — Data found for at least three companies; open SWE role count or visa sponsorship commonly missing.
2 — Glassdoor salaries cited for two or fewer companies; no careers page data.
1 — No careers page or Glassdoor navigation; all salary and sponsorship data is fabricated.

#### D. Output Table & Friendliness Score
Is the comparison table complete and are H-1B friendliness scores logical?

5 — Complete seven-column table for all five companies with H-1B friendliness scores (1-5) that are logically derived from approval volume and denial rate (higher approvals + lower denial rate = higher score).
4 — Table with six columns and friendliness scores present; scoring rationale implied but not stated.
3 — Table with five columns; friendliness scores present but not tied to specific USCIS metrics.
2 — Partial table with fewer than five columns; no friendliness score.
1 — No table; narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "uscis_navigation": "<one sentence citing specific evidence>",
    "h1b_data_accuracy": "<one sentence citing specific evidence>",
    "careers_glassdoor_research": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "uscis_navigation": <1-5>,
  "h1b_data_accuracy": <1-5>,
  "careers_glassdoor_research": <1-5>,
  "output_table": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "uscis_navigation": 0.35,
    "h1b_data_accuracy": 0.23,
    "careers_glassdoor_research": 0.23,
    "output_table": 0.19,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
