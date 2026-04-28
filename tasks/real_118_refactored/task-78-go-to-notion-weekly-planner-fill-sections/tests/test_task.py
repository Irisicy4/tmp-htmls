import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please fill in all the missing data on this page to make it complete. Go to https://www.notion.so/templates/weekly-planner and fill in all empty sections of the weekly planner template.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Notion weekly planner template at the provided URL
- Action: fill ALL empty sections — not just some
- Content: reasonable, realistic sample data (tasks, priorities, goals, notes)
- Completion: agent must confirm all sections are filled

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Notion weekly planner?
- What sections were identified as empty?
- Were all sections filled with content?
- Is the filled content reasonable (realistic tasks/goals)?
- Did the agent confirm completion?

### Step 2: Dimension Scoring

#### A. Platform Access (0.35)
Did the agent access the Notion weekly planner?

5 — Agent navigated to the Notion template URL and loaded the weekly planner.
4 — Agent reached Notion but had difficulty with the template.
3 — Agent described the Notion template without navigating to it.
2 — Agent attempted Notion but could not load the template.
1 — No Notion access.

#### B. Section Coverage (0.30)
Were all empty sections identified and filled?

5 — All empty sections identified and filled: days of week, priorities, goals, notes, etc.
4 — Most sections filled; 1-2 missed.
3 — About half the sections filled.
2 — Only a few sections filled.
1 — No sections filled.

#### C. Content Quality (0.25)
Is the filled content realistic and useful?

5 — Realistic tasks, goals, and notes that a real person might have; varied and context-appropriate.
4 — Good content but slightly generic.
3 — Content is plausible but very generic (e.g. 'Task 1', 'Goal 1').
2 — Placeholder-style content.
1 — No meaningful content.

#### D. Completion Confirmation (0.10)
Did the agent confirm all sections are complete?

5 — Agent explicitly confirmed all sections filled and reviewed the result.
4 — Completion implied but not explicitly confirmed.
3 — Agent stated completion without verification.
2 — Agent stopped mid-way without confirmation.
1 — No confirmation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "platform_access": "<one sentence citing specific evidence>",
    "section_coverage": "<one sentence citing specific evidence>",
    "content_quality": "<one sentence citing specific evidence>",
    "completion_confirmation": "<one sentence citing specific evidence>",
  }},
  "platform_access": <1-5>,
  "section_coverage": <1-5>,
  "content_quality": <1-5>,
  "completion_confirmation": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_access": 0.35,
    "section_coverage": 0.30,
    "content_quality": 0.25,
    "completion_confirmation": 0.10,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
