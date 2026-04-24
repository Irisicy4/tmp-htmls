"""LLM-as-judge evaluator for task-41-analyze-the-pros-and-cons.

Category: Marketing & Analytics
Task: Analyze the pros and cons of AI agent services such as Genspark AI, Kimi, Flowith, Manus AI, etc. Measure and list a score for each service.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Analyze the pros and cons of AI agent services such as Genspark AI, Kimi, Flowith, Manus AI, etc. Measure and list a score for each service.'

SYSTEM_PROMPT_EXTRA = 'This is a comparative analysis task covering AI agent platforms. The agent must both analyse and score each service — not just list features.'

CONSTRAINTS = """- Services mentioned: Genspark AI, Kimi, Flowith, Manus AI — all 4 must be covered; "etc." suggests additional services are welcome
- Required: pros AND cons for each (not just pros)
- Required: a numeric score for each service with methodology explained
- Research: agent should search for current capabilities/reviews rather than relying solely on prior knowledge"""

EVIDENCE_QUESTIONS = """- Were all 4 named services covered (Genspark, Kimi, Flowith, Manus)?
- Are pros AND cons provided for each?
- Is a score given for each service? What methodology?
- Did the agent research current information or rely on prior knowledge?"""

DIMENSION_RUBRICS = """#### A. Service Coverage
Were all required services covered?

5 — All 4 named services covered; 1+ additional services included as bonus.
4 — All 4 named services covered; no additional services.
3 — 3 of 4 services covered; one missing or only briefly mentioned.
2 — 2 of 4 services covered with depth.
1 — Fewer than 2 services covered.

#### B. Analysis Depth
Is the pros/cons analysis substantive and specific?

5 — 2+ specific pros and 2+ specific cons per service with concrete examples (e.g. "Kimi's long context window handles 128k tokens" rather than "good at long documents").
4 — 1–2 pros and 1–2 cons per service; mostly specific but some vague.
3 — Pros and cons present but generic (e.g. "easy to use", "limited features") without specifics.
2 — Only pros or only cons for most services.
1 — No meaningful analysis; just feature lists or descriptions.

#### C. Scoring Methodology
Is the scoring transparent, consistent, and justified?

5 — Clear scoring methodology (e.g. weighted criteria: capability, ease of use, pricing, reliability); scores explained per service; consistent scale used.
4 — Scores provided with brief justification; methodology partially explained.
3 — Scores given without methodology; scores appear arbitrary.
2 — Relative ranking only (1st, 2nd, 3rd) without numeric scores.
1 — No scores provided.

#### D. Output Structure
Is the analysis well-organised and easy to compare across services?

5 — Structured format: per-service sections with pros/cons + score; summary comparison table or final ranking.
4 — Per-service sections present; no summary comparison table.
3 — Services discussed in narrative form; hard to compare directly.
2 — Mixed narrative with service names scattered throughout.
1 — Unstructured response."""

DIMENSION_WEIGHTS = {
    'service_coverage': 0.2,
    'analysis_depth': 0.3,
    'scoring_methodology': 0.3,
    'output_structure': 0.2,
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
