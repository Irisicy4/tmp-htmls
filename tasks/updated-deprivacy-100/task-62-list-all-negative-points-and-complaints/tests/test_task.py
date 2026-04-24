"""LLM-as-judge evaluator for task-62-list-all-negative-points-and-complaints.

Category: Marketing & Analytics
Task: List all negative points and complaints about Claro Brasil, CNPJ 40432544000147.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'List all negative points and complaints about Claro Brasil, CNPJ 40432544000147.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves researching a specific Brazilian company online to find negative reviews, complaints, and criticism from customers.'

CONSTRAINTS = """- Target: Claro Brasil, CNPJ 40432544000147
- Scope: all publicly available negative points — Google reviews, Reclame Aqui, social media, news
- Must cover multiple sources, not just one platform
- Output: structured list of complaints by category or source"""

EVIDENCE_QUESTIONS = """- Did the agent search for the company by name and CNPJ?
- What platforms were searched (Google, Reclame Aqui, Procon, social media)?
- What specific complaints were found?
- Are complaints organized by category or source?"""

DIMENSION_RUBRICS = """#### A. Search Breadth (0.3)
Did the agent search multiple platforms?

5 — Searched 4+ sources: Google reviews, Reclame Aqui, Procon, social media, forums.
4 — Searched 2-3 sources.
3 — Searched only one source (e.g. Google reviews only).
2 — Searched web generally without targeting complaint platforms.
1 — No research performed.

#### B. Complaint Specificity (0.35)
Are specific complaints reported with detail?

5 — Specific complaints with examples: dates, complaint text snippets, recurring themes.
4 — Specific complaints listed but without much detail.
3 — Categories of complaints listed (e.g. 'delivery issues') without specifics.
2 — Vague characterization ('bad reviews exist').
1 — No complaints found or reported.

#### C. Source Attribution (0.2)
Are sources cited for each complaint category?

5 — Each complaint or theme attributed to a specific source with context.
4 — Most complaints attributed to sources.
3 — Some attribution but inconsistent.
2 — No source attribution.
1 — No sourcing.

#### D. Output Organization (0.15)
Is the output well-organized and easy to use?

5 — Complaints organized by category or source with clear headings.
4 — Organized but one level of structure.
3 — List format without categorization.
2 — Wall of text.
1 — No organization."""

DIMENSION_WEIGHTS = {
    'search_breadth': 0.3,
    'complaint_specificity': 0.35,
    'source_attribution': 0.2,
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
