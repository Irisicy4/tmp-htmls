"""LLM-as-judge evaluator for task-51-delete-all-historical-emails-in-fastmail.

Category: Daily Activities
Task: Delete all historical emails in Fastmail that are related to verification codes.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Delete all historical emails in Fastmail that are related to verification codes.'

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent successfully found and deleted verification code emails from a Fastmail inbox.'

CONSTRAINTS = """- Platform: Fastmail specifically
- Scope: all historical verification code emails — the agent should be thorough
- Action: actual deletion required, not just identification
- Search strategy: agent should use iterative search refinement to catch all variants"""

EVIDENCE_QUESTIONS = """- Did the agent navigate to Fastmail?
- What search queries were used to find verification code emails?
- How many emails were found and deleted?
- Did the agent use multiple search iterations to ensure completeness?
- Was deletion confirmed?"""

DIMENSION_RUBRICS = """#### A. Platform Access (0.2)
Did the agent access Fastmail?

5 — Agent navigated to Fastmail and accessed the inbox.
4 — Agent accessed Fastmail but with some difficulty.
3 — Agent attempted Fastmail but had login or access issues.
2 — Agent described Fastmail without navigating to it.
1 — No Fastmail navigation.

#### B. Search Strategy (0.3)
Did the agent use effective search to find verification code emails?

5 — Used multiple search queries covering various keywords (verification, 验证码, code, OTP, etc.) with iterative refinement.
4 — Used 2-3 good search queries.
3 — Used only one broad search query.
2 — Browsed inbox manually without searching.
1 — No search strategy.

#### C. Deletion Execution (0.35)
Were emails actually deleted?

5 — Agent selected and deleted emails with confirmation of deletion count or empty results after.
4 — Agent deleted emails but without clear confirmation of completeness.
3 — Agent deleted some emails but process was incomplete.
2 — Agent identified emails without deleting them.
1 — No deletion performed.

#### D. Completeness (0.15)
Did the agent ensure thorough coverage?

5 — Agent explicitly checked for remaining emails after deletion and confirmed inbox clear of verification emails.
4 — Agent did multiple passes but did not verify completeness.
3 — Agent did one pass only.
2 — Agent acknowledged potential incompleteness without addressing it.
1 — No completeness consideration."""

DIMENSION_WEIGHTS = {
    'platform_access': 0.2,
    'search_strategy': 0.3,
    'deletion_execution': 0.35,
    'completeness': 0.15,
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
