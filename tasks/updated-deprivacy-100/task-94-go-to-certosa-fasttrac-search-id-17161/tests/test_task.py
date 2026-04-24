"""LLM-as-judge evaluator for task-94-go-to-certosa-fasttrac-search-id-17161.

Category: Data & ML Engineering
Task: Go to https://www.certosa.com/Services/FastTrac/Index.aspx, search for ID 17161, and give me a summary of the results.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Go to https://www.certosa.com/Services/FastTrac/Index.aspx, search for ID 17161, and give me a summary of the results.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves navigating a specific web form, entering a search ID, and summarizing the results returned.'

CONSTRAINTS = """- URL: must navigate to the exact URL provided
- Search: ID 17161 specifically
- Output: summary of whatever results the search returns
- If the site is unavailable or the ID returns no results, this should be noted"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to the certosa.com FastTrac URL?
- Was the search form found and ID 17161 entered?
- What results were returned?
- Is the summary accurate to the returned data?"""

DIMENSION_RUBRICS = """#### A. Site Navigation (0.25)
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
1 — No summary."""

DIMENSION_WEIGHTS = {
    'site_navigation': 0.25,
    'search_execution': 0.3,
    'result_accuracy': 0.35,
    'summary_quality': 0.1,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
