import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to https://mightydelivery.meetmighty.com/admin and log in with email: "
    "admin@mightydelivery.com and password: 12345678. Navigate to the deliveries "
    "or orders section and extract only the items scheduled for today. Present them "
    "in a structured table with order ID, recipient, address, and scheduled delivery time.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: mightydelivery.meetmighty.com/admin — must log in with provided credentials
- Filter: today's deliveries only — not all orders, not past/future orders
- Output format: structured table with order ID, recipient, address, and delivery time
- If no deliveries are scheduled for today, the agent should explicitly state this

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the admin panel and log in? Cite evidence.
- Did the agent locate the deliveries/orders section?
- Was a date filter applied to show only today's items?
- What data was returned? Is it clearly for today's date?
- Is the output structured as a table with the required fields?

### Step 2: Dimension Scoring

#### A. Login & Navigation
Did the agent successfully log in and reach the correct section?

5 — Agent logged in with provided credentials and navigated to deliveries/orders section.
4 — Agent logged in but had difficulty finding the correct section.
3 — Agent reached the site but could not log in or reached the wrong section.
2 — Agent attempted to navigate but encountered login or access errors.
1 — No navigation attempted.

#### B. Date Filtering
Did the agent correctly filter for today's deliveries only?

5 — Agent explicitly filtered for today's date and confirmed the results are for today.
4 — Agent filtered but the date boundary is approximate or not explicitly confirmed.
3 — Agent retrieved some orders but did not clearly filter for today only.
2 — Agent retrieved all orders without any date filtering.
1 — No filtering attempted.

#### C. Data Completeness
Did the agent extract the required fields for each delivery?

5 — All four fields present: order ID, recipient, address, scheduled delivery time.
4 — Three of four fields present.
3 — Two of four fields present.
2 — Only one field present or data is very incomplete.
1 — No structured data extracted.

#### D. Output Clarity
Is the output presented clearly as a structured table?

5 — Clear table format with headers and one row per delivery item.
4 — Table-like format but minor formatting issues.
3 — Data present but in paragraph or list form rather than a table.
2 — Data present but very hard to parse.
1 — No structured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "login_navigation": <1-5>,
  "date_filtering": <1-5>,
  "data_completeness": <1-5>,
  "output_clarity": <1-5>,
  "dimension_reasoning": {{
    "login_navigation": "<one sentence citing specific evidence>",
    "date_filtering": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "output_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "login_navigation":  0.25,
    "date_filtering":    0.30,
    "data_completeness": 0.25,
    "output_clarity":    0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
