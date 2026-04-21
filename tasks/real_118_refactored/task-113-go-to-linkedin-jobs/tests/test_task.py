import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to LinkedIn Jobs and search for 'Head of Product' roles at Series B or later startups "
    "in San Francisco, CA posted in the last 30 days. Identify five distinct job postings and for "
    "each record: company name, job title, location (remote/hybrid/on-site), and any listed salary "
    "range. Then go to Glassdoor and look up the company rating, culture & values score, and CEO "
    "approval rating for each company. Finally, visit Crunchbase and find each company's funding "
    "stage, total funding raised, and most recent funding round date. Compile a Head of Product "
    "opportunity report table with all specified columns plus a 'priority tier' (High/Medium/Low) "
    "based on growth signals.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: LinkedIn Jobs, Glassdoor, AND Crunchbase — all three required
- Jobs: Five distinct "Head of Product" postings, SF area, Series B+, last 30 days
- LinkedIn fields: company name, job title, work mode (remote/hybrid/on-site), salary range (if listed)
- Glassdoor fields: overall company rating (out of 5), culture & values score, CEO approval %
- Crunchbase fields: funding stage, total funding raised, most recent round date
- Output: Table with all ten columns plus priority tier (High/Medium/Low)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate LinkedIn Jobs and find SF Head of Product postings from startups?
- Are five distinct company names identified with job details?
- Did the agent look up Glassdoor data for the companies?
- Did the agent look up Crunchbase funding data for the companies?
- Is there a complete table with priority tier assignments?

### Step 2: Dimension Scoring

#### A. LinkedIn Jobs Search
Did the agent navigate LinkedIn Jobs and find real SF startup Head of Product postings?

5 — Navigated LinkedIn Jobs, applied correct filters (title, location, date), and found five distinct postings from named companies with company names, work mode, and salary (or noted "not listed").
4 — Found four or five postings with most fields; one company is missing work mode or another field.
3 — Found at least three postings with company names; two postings are incomplete or filters partially applied.
2 — Found fewer than three specific postings; companies are generic or not clearly from recent LinkedIn search.
1 — No LinkedIn navigation; companies and job postings are fabricated.

#### B. Glassdoor Data Retrieval
Did the agent look up Glassdoor ratings for all five companies?

5 — Glassdoor overall rating, culture & values score, and CEO approval % retrieved for all five companies.
4 — Glassdoor data found for four of five companies; one is missing one score.
3 — Glassdoor data (at least overall rating) found for three or four companies.
2 — Glassdoor referenced for fewer than three companies; most data is estimated.
1 — Glassdoor not used; ratings absent or fabricated.

#### C. Crunchbase Funding Data
Did the agent retrieve funding data from Crunchbase for all five companies?

5 — Crunchbase consulted; funding stage, total funding, and most recent round date retrieved for all five companies.
4 — Crunchbase data found for four of five companies; one is missing the round date or total funding.
3 — Crunchbase data found for at least three companies; common gaps are total funding or round dates.
2 — Crunchbase referenced for fewer than three companies; funding data mostly estimated.
1 — Crunchbase not used; all funding data is from prior knowledge or fabricated.

#### D. Output Table & Priority Tier
Is the final report table complete with all columns and meaningful priority tier assignments?

5 — Complete ten-column table with priority tier (High/Medium/Low) for all five companies; tiers are logically assigned based on stated criteria (funding stage, Glassdoor score, growth signals).
4 — Table with nine of ten columns; priority tiers present for all companies but one tier is not justified.
3 — Table with eight or fewer columns; priority tiers present but criteria not explained.
2 — Table present with fewer than seven columns; no priority tier column.
1 — No table; narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "linkedin_search": <1-5>,
  "glassdoor_data": <1-5>,
  "crunchbase_data": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "linkedin_search": "<one sentence citing specific evidence>",
    "glassdoor_data": "<one sentence citing specific evidence>",
    "crunchbase_data": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "linkedin_search": 0.30,
    "glassdoor_data":  0.25,
    "crunchbase_data": 0.25,
    "output_table":    0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
