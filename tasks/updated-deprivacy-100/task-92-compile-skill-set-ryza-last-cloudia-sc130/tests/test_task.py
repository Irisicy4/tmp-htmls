"""LLM-as-judge evaluator for task-92-compile-skill-set-ryza-last-cloudia-sc130.

Category: Daily Activities
Task: For the Last Cloudia x Atelier Ryza collaboration, please compile the recommended skill set for Ryza with an SC limit of 130, focusing on break specialization.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'For the Last Cloudia x Atelier Ryza collaboration, please compile the recommended skill set for Ryza with an SC limit of 130, focusing on break specialization.'

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves researching the mobile game Last Cloudia's collaboration with Atelier Ryza to find optimal skill builds for the Ryza character under specific constraints (SC 130, break spec)."

CONSTRAINTS = """- Game: Last Cloudia (ラストクラウディア) collaboration with Atelier Ryza
- Character: Ryza
- Constraint: SC (Skill Cost) limit of 130
- Specialization: break (ブレイク) focus
- Source: community guides, wikis, or YouTube — not just general advice"""

EVIDENCE_QUESTIONS = """- Did the agent search for Last Cloudia Ryza build guides?
- What specific skills were recommended?
- Do the recommended skills fit within SC 130?
- Is break specialization the focus?
- Were community sources (wiki, Reddit, YouTube) used?"""

DIMENSION_RUBRICS = """#### A. Source Research (0.25)
Did the agent find relevant community sources?

5 — Found wiki, forum, or YouTube guide specifically for Last Cloudia Ryza collab build.
4 — Found partial sources.
3 — Found general Last Cloudia build guides without Ryza-specific focus.
2 — Only general game info without build data.
1 — No research.

#### B. Skill Specificity (0.35)
Are specific skills listed?

5 — Named skills with SC costs listed; total SC within 130 confirmed.
4 — Skills named but SC costs not totaled.
3 — Skill categories suggested without specific skill names.
2 — Very vague 'use attack skills' without specifics.
1 — No skill information.

#### C. Sc Compliance (0.25)
Do the recommended skills fit within SC 130?

5 — Total SC calculated and confirmed ≤130.
4 — Skills appropriate for SC 130 but total not explicitly confirmed.
3 — SC constraint acknowledged but compliance unclear.
2 — SC constraint ignored.
1 — Skills clearly exceed SC 130.

#### D. Break Focus (0.15)
Is break specialization the focus?

5 — Break-focused skills prioritized with explanation of why they synergize with Ryza.
4 — Break skills included but focus not clearly stated.
3 — Mix of break and general skills without prioritization.
2 — Generic build not focused on break.
1 — No break focus."""

DIMENSION_WEIGHTS = {
    'source_research': 0.25,
    'skill_specificity': 0.35,
    'sc_compliance': 0.25,
    'break_focus': 0.15,
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
