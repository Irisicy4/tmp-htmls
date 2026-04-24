"""LLM-as-judge evaluator for task-23-please-design-comprehensive-learning-pla.

Category: Educational Planning
Task: Please design a comprehensive learning plan for Web 3.0. Do not take any action yet—first list the steps you will use to help me create the plan.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please design a comprehensive learning plan for Web 3.0. Do not take any action yet—first list the steps you will use to help me create the plan.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully outlined a clear and logical sequence of steps to create a comprehensive learning plan for Web 3.0.'

CONSTRAINTS = """- The steps must be logically ordered and relevant to Web 3.0 topics.
- The steps must include a mix of foundational knowledge, practical skills, and advanced concepts.
- The steps must be actionable and specific enough to guide the creation of the learning plan."""

EVIDENCE_QUESTIONS = """- Are the steps logically sequenced to build knowledge progressively?
- Do the steps address key aspects of Web 3.0, such as blockchain, decentralized applications, and semantic web?
- Are the steps actionable and specific enough to guide the creation of a learning plan?
- Do the steps include a balance of theoretical and practical learning components?"""

DIMENSION_RUBRICS = """#### A. Logical Sequence
Assesses whether the steps are presented in a logical and progressive order.

5 — The steps are perfectly ordered, with each step building clearly on the previous one.
4 — The steps are mostly well-ordered, with minor gaps or redundancies.
3 — The steps are somewhat ordered but include noticeable gaps or redundancies.
2 — The steps are poorly ordered, making it difficult to follow a logical progression.
1 — The steps are entirely disorganized and lack any logical sequence.

#### B. Topic Coverage
Evaluates whether the steps cover key topics relevant to Web 3.0 comprehensively.

5 — All major topics of Web 3.0 are covered comprehensively and appropriately.
4 — Most major topics are covered, with minor omissions.
3 — Some major topics are covered, but there are significant gaps.
2 — Few major topics are covered, with many critical omissions.
1 — The steps fail to address key topics of Web 3.0.

#### C. Actionability
Measures whether the steps are actionable and specific enough to guide the creation of a learning plan.

5 — All steps are highly actionable and specific, providing clear guidance.
4 — Most steps are actionable and specific, with minor ambiguities.
3 — Some steps are actionable, but others are vague or unclear.
2 — Few steps are actionable, with significant vagueness or lack of clarity.
1 — The steps are entirely vague and lack actionable guidance.

#### D. Balance Of Theory And Practice
Assesses whether the steps include a balance of theoretical and practical learning components.

5 — The steps achieve an excellent balance of theory and practice.
4 — The steps mostly balance theory and practice, with minor imbalances.
3 — The steps include some balance, but one aspect is noticeably underrepresented.
2 — The steps are heavily skewed toward either theory or practice.
1 — The steps lack any balance between theory and practice."""

DIMENSION_WEIGHTS = {
    'logical_sequence': 0.3,
    'topic_coverage': 0.3,
    'actionability': 0.2,
    'balance_of_theory_and_practice': 0.2,
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
