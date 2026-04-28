"""LLM-as-judge evaluator for task-12-please-compose-an-email-to.

Category: Professional Communication
Task: Please compose an email to David Sullivan (dgs@bu.edu). Tell him that the one-day basic Java class he requested last time is available on either November 29th or November 30th. Since the weather has g
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Please compose an email to David Sullivan (dgs@bu.edu). Tell him that the one-day basic Java class he requested last time is available on either November 29th or November 30th. Since the weather has gotten colder, write politely and include a reminder to be careful not to catch a cold. Stop just before pressing the send button — do not actually send it.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully composed a polite and professional email to David Sullivan, providing the requested information about the Java class and including a seasonal reminder about health.'

CONSTRAINTS = """- The email must be addressed to David Sullivan at dgs@bu.edu.
- It must mention the availability of the one-day basic Java class on November 29th or November 30th.
- The tone should be polite and professional.
- Include a seasonal reminder about being careful not to catch a cold.
- The email must stop before pressing the send button."""

EVIDENCE_QUESTIONS = """- Does the email correctly address David Sullivan and use the provided email address?
- Does the email clearly state the availability of the Java class on November 29th or November 30th?
- Is the tone of the email polite and professional throughout?
- Does the email include a seasonal reminder about health in a polite manner?
- Did the AI agent stop before pressing the send button?"""

DIMENSION_RUBRICS = """#### A. Accuracy Of Information
Evaluates whether the email contains the correct details about the Java class availability and recipient information.

5 — All details are accurate, including recipient email and class dates.
4 — Minor inaccuracies in details but the core information is correct.
3 — Some inaccuracies in key details but the intent is clear.
2 — Significant inaccuracies that could confuse the recipient.
1 — The information is entirely incorrect or missing.

#### B. Tone And Politeness
Assesses whether the email maintains a polite and professional tone.

5 — The tone is consistently polite and professional throughout.
4 — The tone is mostly polite and professional with minor lapses.
3 — The tone is neutral but lacks warmth or professionalism.
2 — The tone is somewhat impolite or unprofessional.
1 — The tone is rude or inappropriate.

#### C. Seasonal Reminder Inclusion
Checks if the email includes a polite seasonal reminder about health.

5 — The reminder is included and phrased politely and appropriately.
4 — The reminder is included but phrasing is slightly awkward.
3 — The reminder is included but lacks politeness or appropriateness.
2 — The reminder is missing or poorly executed.
1 — The reminder is entirely absent.

#### D. Completion Without Sending
Evaluates whether the AI agent stopped before pressing the send button.

5 — The email was completed and the agent clearly stopped before sending.
4 — The email was completed but evidence of stopping is slightly unclear.
3 — The email was mostly completed but stopping before sending is ambiguous.
2 — The email was completed but the agent pressed the send button.
1 — The email was incomplete or sent prematurely."""

DIMENSION_WEIGHTS = {
    'accuracy_of_information': 0.3,
    'tone_and_politeness': 0.25,
    'seasonal_reminder_inclusion': 0.2,
    'completion_without_sending': 0.25,
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
