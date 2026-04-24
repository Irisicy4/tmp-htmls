"""LLM-as-judge evaluator for task-49-go-to-demo-pocketbase-io-reset.

Category: System Configuration
Task: Go to https://demo.pocketbase.io and reset all settings to their default values. Navigate through all available settings menus and restore defaults wherever the option exists.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Go to https://demo.pocketbase.io and reset all settings to their default values. Navigate through all available settings menus and restore defaults wherever the option exists.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully navigated the PocketBase demo interface and reset all settings to their default values. This involves exploring all available settings menus, identifying default restoration options, and ensuring every setting is correctly restored. The evaluation emphasizes thoroughness, precision, and compliance with the task instructions.'

CONSTRAINTS = """- Access the PocketBase demo interface at https://demo.pocketbase.io.
- Navigate through all available settings menus systematically.
- Restore settings to their default values wherever the option exists.
- Ensure no settings are skipped or left unchanged.
- Avoid making any changes beyond restoring defaults."""

EVIDENCE_QUESTIONS = """- Did the agent access the PocketBase demo interface successfully?
- Did the agent navigate through all available settings menus?
- Did the agent restore settings to their default values wherever the option existed?
- Were all settings menus addressed without omissions?
- Did the agent avoid making changes beyond restoring defaults?"""

DIMENSION_RUBRICS = """#### A. Navigation Completeness
Measures whether the agent navigated through all settings menus.

5 — The agent navigated through all available settings menus without missing any.
4 — The agent navigated through most settings menus, missing only one or two.
3 — The agent navigated through some settings menus but missed several.
2 — The agent navigated through very few settings menus.
1 — The agent failed to navigate through any settings menus.

#### B. Default Restoration Accuracy
Measures the accuracy of restoring settings to their default values.

5 — The agent restored all settings to their default values accurately.
4 — The agent restored most settings to their default values, with minor inaccuracies.
3 — The agent restored some settings to their default values but made notable errors.
2 — The agent restored very few settings to their default values and made significant errors.
1 — The agent failed to restore any settings to their default values.

#### C. Task Adherence
Measures whether the agent avoided making changes beyond restoring defaults.

5 — The agent strictly adhered to the task and made no changes beyond restoring defaults.
4 — The agent mostly adhered to the task but made one or two minor changes beyond restoring defaults.
3 — The agent somewhat adhered to the task but made several changes beyond restoring defaults.
2 — The agent poorly adhered to the task and made many changes beyond restoring defaults.
1 — The agent completely disregarded the task and made numerous unrelated changes.

#### D. Efficiency
Measures the speed and organization of the agent's actions.

5 — The agent completed the task quickly and systematically without unnecessary delays.
4 — The agent completed the task efficiently but with minor delays or disorganization.
3 — The agent completed the task with moderate delays or disorganization.
2 — The agent completed the task with significant delays or disorganization.
1 — The agent failed to complete the task or was extremely slow and disorganized."""

DIMENSION_WEIGHTS = {
    'navigation_completeness': 0.3,
    'default_restoration_accuracy': 0.3,
    'task_adherence': 0.2,
    'efficiency': 0.2,
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
