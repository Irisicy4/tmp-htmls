import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Go to https://pocketbase.io/_/ and navigate through all available settings menus. Reset all settings to their default values wherever the option exists.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: pocketbase.io demo admin panel
- Action: reset/restore defaults wherever available
- Coverage: agent should explore multiple settings sections, not just one
- Confirmation: agent should confirm what was reset

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the pocketbase demo admin panel?
- Did the agent find settings menus?
- Were reset/default options found and executed?
- Which sections were covered?
- Did the agent confirm completion?

### Step 2: Dimension Scoring

#### A. Navigation Execution (0.35)
Did the agent navigate to pocketbase.io admin and access settings?

5 — Agent navigated to pocketbase.io/_/ and accessed settings successfully.
4 — Agent reached the site but had difficulty with admin access.
3 — Agent reached the site but navigated to wrong section.
2 — Agent attempted navigation but failed to access admin.
1 — No navigation attempted.

#### B. Settings Discovery (0.26)
Did the agent find and explore multiple settings sections?

5 — Agent systematically explored multiple settings tabs/sections.
4 — Agent found settings but only explored one or two sections.
3 — Agent found the settings area but did not explore thoroughly.
2 — Agent described where settings might be without finding them.
1 — No settings found.

#### C. Reset Execution (0.3)
Did the agent successfully reset settings to defaults?

5 — Agent executed reset/restore defaults in at least one section with confirmation.
4 — Agent found reset options but execution was partial or unclear.
3 — Agent attempted reset but encountered errors.
2 — Agent identified reset options without executing them.
1 — No reset attempted.

#### D. Completion Report (0.09)
Did the agent clearly report what was done?

5 — Clear summary of which sections were reset and what changed.
4 — Summary present but incomplete.
3 — Vague confirmation of completion.
2 — No clear report of actions taken.
1 — No report.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "navigation_execution": "<one sentence citing specific evidence>",
    "settings_discovery": "<one sentence citing specific evidence>",
    "reset_execution": "<one sentence citing specific evidence>",
    "completion_report": "<one sentence citing specific evidence>",
  }},
  "navigation_execution": <1-5>,
  "settings_discovery": <1-5>,
  "reset_execution": <1-5>,
  "completion_report": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "navigation_execution": 0.35,
    "settings_discovery": 0.26,
    "reset_execution": 0.3,
    "completion_report": 0.09,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
