import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Search for and organize jobs that women aged 50 and above can do.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target demographic: women aged 50 or older
- Must search the web for relevant job options (not rely solely on internal knowledge)
- Should cover diverse job categories (full-time, part-time, self-employed, remote, etc.)
- Jobs should be realistically accessible to women in this age group
- Output should be organized/structured, not a flat unformatted list

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search the web for jobs suited to women over 50? What sources or keywords were used?
- How many distinct job types or career categories were identified?
- Are the jobs appropriate for women in this age group (e.g. leverages experience, manageable physical demands, accessible entry)?
- Is the output organized into categories or structured in a useful way?
- Are practical details (qualifications, pay range, how to get started) included?

### Step 2: Dimension Scoring

#### A. Job Coverage (0.30)
How many distinct job types or career categories were identified?

5 — 8 or more distinct job titles or categories with specific examples (e.g. tutoring, bookkeeping, healthcare aide, real estate agent, consultant, social worker, etc.).
4 — 5-7 distinct jobs or categories identified.
3 — 3-4 job types mentioned.
2 — Only 1-2 job types.
1 — No jobs identified or response is not about jobs.

#### B. Age-Group Relevance (0.30)
Are the jobs specifically appropriate and accessible for women aged 50 and older?

5 — Jobs explicitly vetted for the 50+ demographic: leverages decades of experience, manageable physical requirements, flexible or part-time options highlighted, re-entry or upskilling paths noted.
4 — Most jobs are suitable for 50+ women but some may have high physical demands or tech barriers without accommodation notes.
3 — General job list not specifically filtered or tailored for women over 50.
2 — Several jobs are poorly suited for this age group (heavy physical labor, highly competitive entry-level roles, etc.).
1 — Jobs are not relevant to women over 50.

#### C. Organization Quality (0.25)
Is the information well-organized and easy to navigate?

5 — Clear categories or groupings (e.g. "Healthcare", "Education", "Remote Work") with job titles and brief descriptions for each.
4 — Organized list with most categories or descriptions present.
3 — Simple list with job names but minimal structure or description.
2 — Unorganized or incomplete output.
1 — No structure; jobs mentioned in passing without a list or categories.

#### D. Practical Information (0.15)
Are actionable details provided to help women pursue these jobs?

5 — Qualification requirements, estimated pay range or demand outlook, and getting-started tips included for most jobs.
4 — At least two of the three types of practical info (qualifications, pay, getting started) for most jobs.
3 — Some practical info included for a few jobs only.
2 — Very limited practical info (e.g. only job titles with no supporting detail).
1 — No practical information provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "job_coverage": <1-5>,
  "age_group_relevance": <1-5>,
  "organization_quality": <1-5>,
  "practical_information": <1-5>,
  "dimension_reasoning": {{
    "job_coverage": "<one sentence citing specific evidence>",
    "age_group_relevance": "<one sentence citing specific evidence>",
    "organization_quality": "<one sentence citing specific evidence>",
    "practical_information": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "job_coverage": 0.30,
    "age_group_relevance": 0.30,
    "organization_quality": 0.25,
    "practical_information": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
