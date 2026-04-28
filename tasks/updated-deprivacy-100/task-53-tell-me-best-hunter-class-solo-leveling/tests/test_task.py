"""LLM-as-judge evaluator for task-53-tell-me-best-hunter-class-solo-leveling.

Category: (Self) Media
Task: Tell me the best hunter class to develop in the Netmarble game Solo Leveling: ARISE and explain why you chose it.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Tell me the best hunter class to develop in the Netmarble game Solo Leveling: ARISE and explain why you chose it.'

SYSTEM_PROMPT_EXTRA = 'Assess whether an AI agent provided an accurate, well-reasoned recommendation for the best hunter class in Solo Leveling: ARISE.'

CONSTRAINTS = """- Game: Solo Leveling: ARISE by Netmarble specifically
- Must recommend a specific hunter class (not a vague 'it depends')
- Recommendation must be based on current meta (tier lists, community consensus)
- Reasoning must address growth potential, versatility, and/or end-game viability"""

EVIDENCE_QUESTIONS = """- Did the agent search for current meta information on Solo Leveling ARISE?
- What specific hunter class was recommended?
- What reasoning was provided (tier ranking, versatility, growth potential)?
- Is the recommendation consistent with current community consensus?"""

DIMENSION_RUBRICS = """#### A. Research Quality (0.25)
Did the agent search for current meta data?

5 — Agent searched for current tier lists, guides, or community posts about Solo Leveling ARISE hunter classes.
4 — Agent found some meta info but sources are limited.
3 — Agent relied on general knowledge without specific search.
2 — Agent described classes generally without meta research.
1 — No research performed.

#### B. Recommendation Specificity (0.25)
Did the agent recommend a specific class with clear reasoning?

5 — Specific class recommended with clear reasons: tier ranking, versatility, growth rate, end-game value.
4 — Specific class recommended but reasoning covers only 1-2 factors.
3 — Class recommended but reasoning is vague or generic.
2 — Multiple classes mentioned without a clear winner.
1 — No specific recommendation.

#### C. Meta Accuracy (0.35)
Is the recommendation consistent with current game meta?

5 — Recommended class is S-tier or widely considered top-tier in current meta with supporting evidence.
4 — Recommended class is strong but not universally top-tier; reasoning compensates.
3 — Recommended class is reasonable but not optimal.
2 — Recommended class is sub-optimal or outdated.
1 — Recommendation is clearly wrong or the class doesn't exist.

#### D. Explanation Depth (0.15)
Is the explanation detailed and useful?

5 — Covers tier ranking, growth mechanics, skill synergies, and end-game role.
4 — Covers 2-3 of the above factors.
3 — Basic explanation with only 1 factor.
2 — Very thin explanation.
1 — No explanation."""

DIMENSION_WEIGHTS = {
    'research_quality': 0.25,
    'recommendation_specificity': 0.25,
    'meta_accuracy': 0.35,
    'explanation_depth': 0.15,
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
