"""LLM-as-judge evaluator for task-13-need-to-purchase-esim-for.

Category: Shopping
Task: I need to purchase ESIM for my vacation to Osaka, Japan next week. I will be there from December 22nd 2025, until January 12th, 2026. I need to find an ESIM that has a lot of data (preferably 5G, but 
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "I need to purchase ESIM for my vacation to Osaka, Japan next week. I will be there from December 22nd 2025, until January 12th, 2026. I need to find an ESIM that has a lot of data (preferably 5G, but LTE will do), and is cheap. Please find the best plans for me. I need at least 50GB over the duration of my stay, but 75GB would be better. If there's a big price difference, feel free to choose LTE. If not, go 5G. I would prefer 5G over LTE though"

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully found appropriate eSIM plans matching a traveller's specific requirements."

CONSTRAINTS = """- Destination: Japan (Osaka) — eSIM must work in Japan
- Duration: December 22, 2025 – January 12, 2026 (21 days) — plan must cover full duration
- Data minimum: 50GB (hard minimum); 75GB preferred
- Network: 5G preferred; LTE acceptable if significantly cheaper
- Price decision rule: if 5G and LTE plans are similarly priced, choose 5G; if LTE is significantly cheaper, LTE is fine
- Agent must search for actual current plans, not rely on prior knowledge"""

EVIDENCE_QUESTIONS = """- Did the agent search for eSIM plans? What platform or site did it use?
- What specific plans were found? List them with data allowance, network type, price, and duration.
- Does each plan meet the 50GB minimum? Does it cover the full 21-day period?
- Did the agent reason about the 5G vs LTE tradeoff as instructed?
- Did the agent make a clear recommendation with justification?"""

DIMENSION_RUBRICS = """#### A. Search Execution
Did the agent actively search for current eSIM plans?

5 — Agent searched a real eSIM platform or comparison site (e.g. Airalo, Holafly, Nomad, Esimdb) and retrieved current listings.
4 — Agent searched but retrieved limited results or from only one provider without comparison.
3 — Agent retrieved eSIM information but from a general web search rather than a dedicated eSIM platform.
2 — Agent described what to look for without actually searching.
1 — No search performed; response is from prior knowledge or fabricated.

#### B. Constraint Satisfaction
Do the recommended plans meet all hard requirements?

5 — Recommended plan(s): (a) work in Japan, (b) cover full 21-day period, (c) provide ≥50GB data.
4 — All constraints met but one is weakly supported (e.g. plan says "30 days" but exact start date not verified).
3 — Two of three constraints clearly met; one is ambiguous or missing (e.g. data amount not specified).
2 — Only one constraint clearly met; plan may not cover the full duration or data is below 50GB.
1 — Constraints ignored; recommended plans clearly don't meet the requirements.

#### C. 5G/LTE Decision Quality
Did the agent reason correctly about the 5G vs LTE tradeoff?

5 — Agent explicitly compared 5G and LTE plan prices, applied the user's stated rule (choose 5G unless significant price difference), and justified the recommendation.
4 — Agent recommended 5G or LTE with brief price comparison; reasoning is present but not fully explicit.
3 — Agent mentioned both options but did not compare prices or apply the decision rule.
2 — Agent defaulted to one option without considering the other.
1 — 5G/LTE preference completely ignored.

#### D. Recommendation Quality
Is the final recommendation specific, actionable, and well-justified?

5 — Clear winner recommended with: plan name/provider, price, data, network type, duration, and why it's the best choice given the constraints.
4 — Good recommendation with most details; one element missing (e.g. no price or no comparison to alternatives).
3 — A plan is identified but the recommendation is thin (name only, or no justification).
2 — Multiple options listed without a clear recommendation.
1 — No recommendation made; agent only described the search process."""

DIMENSION_WEIGHTS = {
    'search_execution': 0.2,
    'constraint_satisfaction': 0.35,
    'network_decision_quality': 0.2,
    'recommendation_quality': 0.25,
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
