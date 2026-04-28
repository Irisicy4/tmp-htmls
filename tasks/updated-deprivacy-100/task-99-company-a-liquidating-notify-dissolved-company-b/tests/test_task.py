"""LLM-as-judge evaluator for task-99-company-a-liquidating-notify-dissolved-company-b.

Category: Daily Activities
Task: Company A is preparing for liquidation and dissolution. It has discovered that it still has accounts payable to Company B on its books. However, Company B has already been dissolved. After Company A b
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Company A is preparing for liquidation and dissolution. It has discovered that it still has accounts payable to Company B on its books. However, Company B has already been dissolved. After Company A begins its liquidation process, does it still have an obligation to notify Company B? Please answer based on Chinese law.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves answering a specific Chinese corporate law question about liquidation obligations when a creditor company has already been dissolved.'

CONSTRAINTS = """- Jurisdiction: Chinese law specifically
- Topic: liquidation notification obligations when creditor is dissolved
- Required: cite specific Chinese legal provisions (Company Law, Liquidation regulations)
- Answer: must address the notification obligation and what Company A should do instead"""

EVIDENCE_QUESTIONS = """- Did the agent research Chinese law on this specific scenario?
- What specific legal provisions were cited (Company Law articles)?
- Is the notification obligation clearly addressed?
- What practical guidance is provided for Company A?"""

DIMENSION_RUBRICS = """#### A. Legal Research (0.25)
Did the agent research the relevant Chinese legal framework?

5 — Researched Chinese Company Law, liquidation regulations, and relevant judicial interpretations.
4 — Found relevant law but search was less comprehensive.
3 — General Chinese corporate law found without specific liquidation provisions.
2 — Generic legal answer without Chinese law research.
1 — No legal research.

#### B. Legal Accuracy (0.35)
Is the legal analysis accurate under Chinese law?

5 — Correct analysis: under Chinese Company Law, notification obligation exists but must be directed to known creditors; when creditor is dissolved, obligation typically transforms to public announcement; specific articles cited.
4 — Mostly correct with minor gaps.
3 — Generally correct direction but missing key provisions.
2 — Partially correct but significant errors.
1 — Incorrect or no legal analysis.

#### C. Citation Quality (0.25)
Are specific legal provisions cited?

5 — Specific articles from Chinese Company Law, liquidation procedures, or relevant regulations cited.
4 — Some articles cited.
3 — Legal framework referenced without specific articles.
2 — Vague reference to Chinese law.
1 — No citations.

#### D. Practical Guidance (0.15)
Is practical guidance provided for Company A?

5 — Clear next steps: what Company A should do (public announcement, hold funds, court guidance).
4 — Some guidance provided.
3 — Legal conclusion reached without practical next steps.
2 — Very vague guidance.
1 — No guidance."""

DIMENSION_WEIGHTS = {
    'legal_research': 0.25,
    'legal_accuracy': 0.35,
    'citation_quality': 0.25,
    'practical_guidance': 0.15,
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
