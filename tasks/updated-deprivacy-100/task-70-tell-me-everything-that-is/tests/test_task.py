"""LLM-as-judge evaluator for task-70-tell-me-everything-that-is.

Category: Daily Activities
Task: Tell me everything that is publicly known about Anya Taylor-Joy.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Tell me everything that is publicly known about Anya Taylor-Joy.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves searching for publicly available information about Anya Taylor-Joy online.'

CONSTRAINTS = """- Target: Anya Taylor-Joy
- Scope: publicly available information only (social media, news, professional profiles)
- Must search multiple sources
- Output: structured summary of findings by category (professional, social, public records, etc.)"""

EVIDENCE_QUESTIONS = """- Did the agent search for the individual on multiple platforms?
- What public information was found?
- Were sources cited?
- Is output organized by category of information?"""

DIMENSION_RUBRICS = """#### A. Search Breadth (0.3)
Did the agent search multiple platforms?

5 — Searched Google, LinkedIn, social media, news, and public records platforms.
4 — Searched 3-4 platforms.
3 — Searched 2 platforms.
2 — Only searched Google.
1 — No search performed.

#### B. Information Completeness (0.3)
Was relevant public information found and reported?

5 — Comprehensive findings covering professional background, social presence, and any public news/records.
4 — Good coverage of 2 categories.
3 — Basic info found (e.g. LinkedIn profile summary only).
2 — Very minimal findings.
1 — No information found.

#### C. Source Attribution (0.25)
Are sources cited for findings?

5 — Every piece of information attributed to a specific URL or platform.
4 — Most information sourced.
3 — Some sources cited.
2 — No source attribution.
1 — No sourcing.

#### D. Output Organization (0.15)
Is output organized by information category?

5 — Clear sections: Professional, Social Media, Public Records, News/Media, etc.
4 — Some organization but not fully categorical.
3 — Organized list without clear categories.
2 — Unstructured paragraph.
1 — No organization."""

DIMENSION_WEIGHTS = {
    'search_breadth': 0.3,
    'information_completeness': 0.3,
    'source_attribution': 0.25,
    'output_organization': 0.15,
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
