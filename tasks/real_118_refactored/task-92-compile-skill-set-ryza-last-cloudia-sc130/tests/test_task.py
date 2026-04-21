import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""For the Last Cloudia x Atelier Ryza collaboration, please compile the recommended skill set for Ryza with an SC limit of 130, focusing on break specialization.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Game: Last Cloudia (ラストクラウディア) collaboration with Atelier Ryza
- Character: Ryza
- Constraint: SC (Skill Cost) limit of 130
- Specialization: break (ブレイク) focus
- Source: community guides, wikis, or YouTube — not just general advice

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Last Cloudia Ryza build guides?
- What specific skills were recommended?
- Do the recommended skills fit within SC 130?
- Is break specialization the focus?
- Were community sources (wiki, Reddit, YouTube) used?

### Step 2: Dimension Scoring

#### A. Source Research (0.25)
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
1 — No break focus.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "source_research": <1-5>,
  "skill_specificity": <1-5>,
  "sc_compliance": <1-5>,
  "break_focus": <1-5>,
  "dimension_reasoning": {{
    "source_research": "<one sentence citing specific evidence>",
    "skill_specificity": "<one sentence citing specific evidence>",
    "sc_compliance": "<one sentence citing specific evidence>",
    "break_focus": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "source_research": 0.25,
    "skill_specificity": 0.35,
    "sc_compliance": 0.25,
    "break_focus": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
