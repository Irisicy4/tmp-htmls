"""LLM-as-judge evaluator for task-19-i-am-now-in-central.

Category: Web Research
Task: I am now in central London. Please tell me 10 places with a good view of Big Ben, their addresses, and for each place provide directions from the city center, nearby attractions, and nearby restaurant
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'I am now in central London. Please tell me 10 places with a good view of Big Ben, their addresses, and for each place provide directions from the city center, nearby attractions, and nearby restaurants.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully identified 10 places with a good view of Big Ben, provided their addresses, and included directions, nearby attractions, and nearby restaurants for each place.'

CONSTRAINTS = """- The list must include exactly 10 distinct places.
- Each place must have a clear view of Big Ben.
- Each entry must include an address, directions from central London, nearby attractions, and nearby restaurants.
- Information must be accurate and relevant to central London."""

EVIDENCE_QUESTIONS = """- Does each place on the list have a clear and unobstructed view of Big Ben?
- Are the addresses provided accurate and specific enough to locate the places?
- Are the directions from central London clear and feasible for a visitor to follow?
- Are the nearby attractions and restaurants relevant and within a reasonable distance of each place?"""

DIMENSION_RUBRICS = """#### A. Completeness
Measures whether all required information (view, address, directions, attractions, restaurants) is provided for each place.

5 — All 10 places include complete and accurate information for all required elements.
4 — 9 places include complete and accurate information; minor omissions for 1 place.
3 — 7-8 places include complete information; significant omissions for 2-3 places.
2 — 4-6 places include complete information; major omissions for the rest.
1 — Fewer than 4 places include complete information; most entries are incomplete.

#### B. Accuracy
Assesses whether the provided information (addresses, directions, attractions, restaurants) is factually correct.

5 — All information is accurate and verifiable for all 10 places.
4 — Minor inaccuracies in 1-2 places, but the majority is correct.
3 — Noticeable inaccuracies in 3-4 places, but some information is still correct.
2 — Significant inaccuracies in 5-6 places, making the information unreliable.
1 — Most or all information is inaccurate or unverifiable.

#### C. Relevance
Evaluates whether the listed places, attractions, and restaurants are relevant to the task and context.

5 — All places, attractions, and restaurants are highly relevant to the task and central London context.
4 — Most places, attractions, and restaurants are relevant; minor deviations in 1-2 entries.
3 — Several places, attractions, or restaurants (3-4) are not relevant to the task.
2 — Many places, attractions, or restaurants (5-6) are irrelevant or poorly chosen.
1 — Most or all places, attractions, or restaurants are irrelevant to the task.

#### D. Clarity
Assesses whether the information is presented in a clear and understandable manner.

5 — All information is clearly presented, easy to understand, and well-organized.
4 — Most information is clear, with minor issues in 1-2 entries.
3 — Some information is unclear or poorly organized in 3-4 entries.
2 — Significant clarity issues in 5-6 entries, making them hard to understand.
1 — Most or all information is unclear or disorganized.

#### E. Effort
Measures the apparent effort and thoroughness in researching and compiling the information.

5 — The response demonstrates exceptional effort with detailed and thoughtful research for all 10 places.
4 — The response shows good effort, with minor gaps in thoroughness for 1-2 places.
3 — The response shows moderate effort, with noticeable gaps in 3-4 places.
2 — The response shows minimal effort, with significant gaps in 5-6 places.
1 — The response shows little to no effort, with most entries lacking thoroughness."""

DIMENSION_WEIGHTS = {
    'completeness': 0.3,
    'accuracy': 0.25,
    'relevance': 0.2,
    'clarity': 0.15,
    'effort': 0.1,
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
