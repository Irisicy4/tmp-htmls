"""LLM-as-judge evaluator for task-54-search-for-the-douban-rating.

Category: Daily Activities
Task: Search for the Douban rating of the book 《智人之上》.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Search for the Douban rating of the book 《智人之上》.'

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent successfully retrieved the Douban rating for the book 《智人之上》(Homo Deus / Sapiens sequel by Harari).'

CONSTRAINTS = """- Platform: Douban (douban.com) specifically
- Book: 《智人之上》— must find the correct book entry
- Data: must retrieve the actual rating score and number of ratings
- Accuracy: rating must be from Douban, not Goodreads or other platforms"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to Douban?
- Did the agent find the correct book entry for 《智人之上》?
- What rating was reported (score out of 10, number of raters)?
- Is this clearly from Douban?"""

DIMENSION_RUBRICS = """#### A. Platform Navigation (0.25)
Did the agent navigate to Douban and find the book?

5 — Agent navigated to douban.com, searched for 《智人之上》, and reached the correct book page.
4 — Agent reached Douban but had difficulty finding the exact entry.
3 — Agent found rating data from Douban but via indirect method (e.g. Google snippet).
2 — Agent found rating from a non-Douban source.
1 — No navigation to Douban.

#### B. Rating Accuracy (0.4)
Is the reported rating accurate and from Douban?

5 — Rating score (e.g. 8.x/10) and number of raters clearly reported from Douban.
4 — Rating score reported but rater count missing.
3 — Rating approximately correct but imprecise.
2 — Rating reported but from wrong source or clearly incorrect.
1 — No rating provided.

#### C. Book Identification (0.25)
Did the agent identify the correct book?

5 — Confirmed correct book with author (Yuval Noah Harari) and edition details.
4 — Correct book identified with author but no edition detail.
3 — Book identified by title only without confirming author.
2 — Wrong edition or potential confusion with similar title.
1 — Wrong book identified.

#### D. Response Clarity (0.1)
Is the response clear and concise?

5 — Rating presented clearly with book title, score, and rater count.
4 — Clear but missing one detail.
3 — Information present but buried.
2 — Confusing or verbose.
1 — No useful output."""

DIMENSION_WEIGHTS = {
    'platform_navigation': 0.25,
    'rating_accuracy': 0.4,
    'book_identification': 0.25,
    'response_clarity': 0.1,
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
