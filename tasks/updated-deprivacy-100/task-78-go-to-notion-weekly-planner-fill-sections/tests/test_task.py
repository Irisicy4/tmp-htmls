"""
LLM-as-judge evaluator for EvolveBench task-78-go-to-notion-weekly-planner-fill-sections.

Category: Content Creation
Task: Go to https://www.notion.so/templates/weekly-planner and open the Weekly Planner template. Fill in all the empty sections of the planner with appropriate placeholder data — including goals, daily task
"""

import os, json, re

TASK_INSTRUCTION = """Go to https://www.notion.so/templates/weekly-planner and open the Weekly Planner template. Fill in all the empty sections of the planner with appropriate placeholder data — including goals, daily tasks, notes, and any other incomplete fields — to make the page fully complete."""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to accurately and comprehensively fill out the Weekly Planner template on Notion with appropriate placeholder data. The evaluation focuses on whether the agent completed all sections of the planner, used relevant and realistic placeholder data, and maintained coherence and organization throughout the template.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Access the Weekly Planner template at https://www.notion.so/templates/weekly-planner.
- Fill in all empty sections, including goals, daily tasks, notes, and any other incomplete fields.
- Ensure placeholder data is realistic, relevant, and coherent.
- Maintain proper formatting and organization within the template.
- Do not leave any sections incomplete or partially filled.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access the correct Weekly Planner template from the provided URL?
- Were all sections of the planner filled with appropriate placeholder data?
- Is the placeholder data realistic and relevant to the context of a weekly planner?
- Did the agent maintain proper formatting and organization throughout the template?
- Are there any sections left incomplete or partially filled?

### Step 2: Dimension Scoring

#### A. Template Completion
Measures whether all sections of the Weekly Planner template were fully completed.

5 — All sections of the planner are fully completed with no omissions.
4 — Most sections are completed, with only minor omissions in less critical areas.
3 — Several sections are incomplete or partially filled.
2 — Many sections are left incomplete, with placeholder data missing in critical areas.
1 — The majority of the planner is incomplete or untouched.

#### B. Placeholder Data Quality
Evaluates the realism and relevance of the placeholder data used.

5 — All placeholder data is realistic, relevant, and contextually appropriate for a weekly planner.
4 — Most placeholder data is realistic and relevant, with minor inconsistencies.
3 — Placeholder data is somewhat realistic but includes noticeable inconsistencies or irrelevance.
2 — Placeholder data is largely unrealistic or irrelevant to the context of a weekly planner.
1 — Placeholder data is entirely inappropriate or missing.

#### C. Organization And Formatting
Assesses the coherence, organization, and formatting of the completed planner.

5 — The planner is well-organized and formatted, with clear and coherent structure throughout.
4 — The planner is mostly organized and formatted, with minor issues in structure or clarity.
3 — The planner has noticeable issues in organization or formatting that affect readability.
2 — The planner is poorly organized and formatted, making it difficult to follow.
1 — The planner lacks any coherent organization or formatting.

#### D. Attention To Detail
Measures the agent's thoroughness in addressing all aspects of the task.

5 — Every detail of the task is addressed meticulously, with no errors or oversights.
4 — Most details are addressed thoroughly, with only minor oversights.
3 — Several details are overlooked or handled inadequately.
2 — Many critical details are ignored or handled poorly.
1 — The task lacks attention to detail, with numerous errors and omissions.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_completion": <1-5>,
  "placeholder_data_quality": <1-5>,
  "organization_and_formatting": <1-5>,
  "attention_to_detail": <1-5>,
  "dimension_reasoning": {{
    "template_completion": "<one sentence citing specific evidence>",
    "placeholder_data_quality": "<one sentence citing specific evidence>",
    "organization_and_formatting": "<one sentence citing specific evidence>",
    "attention_to_detail": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_completion": 0.35,
    "placeholder_data_quality": 0.3,
    "organization_and_formatting": 0.2,
    "attention_to_detail": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

