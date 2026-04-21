import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Go to https://www.certosa.com/Services/FastTrac/Index.aspx, search for ID 17161, and give me a summary of the results.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must navigate to the exact URL provided
- Search: ID 17161 specifically
- Output: summary of whatever results the search returns
- If the site is unavailable or the ID returns no results, this should be noted

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the certosa.com FastTrac URL?
- Was the search form found and ID 17161 entered?
- What results were returned?
- Is the summary accurate to the returned data?

### Step 2: Dimension Scoring

#### A. Site Navigation (0.25)
Did the agent navigate to the correct URL?

5 — Agent navigated to certosa.com/Services/FastTrac/Index.aspx successfully.
4 — Agent reached certosa.com but had difficulty finding FastTrac.
3 — Agent found a related certosa.com page but not FastTrac.
2 — Agent described the URL without navigating.
1 — No navigation.

#### B. Search Execution (0.3)
Was the search for ID 17161 executed?

5 — ID 17161 entered in the search form and search submitted.
4 — Search attempted but form interaction had issues.
3 — Agent found the form but did not submit the search.
2 — Agent described the search without executing it.
1 — No search.

#### C. Result Accuracy (0.35)
Are the reported results accurate?

5 — Results accurately summarized from the actual page response.
4 — Results mostly accurate with minor gaps.
3 — Partial results reported.
2 — Results described without accuracy confidence.
1 — No results reported or site unavailable without noting it.

#### D. Summary Quality (0.1)
Is the summary well-written?

5 — Clear, concise summary covering key result fields.
4 — Good summary with minor gaps.
3 — Adequate but verbose.
2 — Incomplete or confusing.
1 — No summary.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "site_navigation": <1-5>,
  "search_execution": <1-5>,
  "result_accuracy": <1-5>,
  "summary_quality": <1-5>,
  "dimension_reasoning": {{
    "site_navigation": "<one sentence citing specific evidence>",
    "search_execution": "<one sentence citing specific evidence>",
    "result_accuracy": "<one sentence citing specific evidence>",
    "summary_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "site_navigation": 0.25,
    "search_execution": 0.3,
    "result_accuracy": 0.35,
    "summary_quality": 0.1,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
