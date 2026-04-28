"""LLM-as-judge evaluator for task-22-send-message-to-my-manager.

Category: Professional Communication
Task: Compose an email to my manager (manager@example.com) telling them I will be over an hour late due to traffic conditions. Write it politely. Stop just before pressing send — do not actually send it.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Compose an email to my manager (manager@example.com) telling them I will be over an hour late due to traffic conditions. Write it politely. Stop just before pressing send — do not actually send it.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully composed a polite and professional email to a manager, explaining their delay of over an hour due to traffic conditions. The evaluation emphasizes clarity, tone, grammatical accuracy, and adherence to the instruction to stop before sending the email.'

CONSTRAINTS = """- The email must be addressed to manager@example.com.
- The reason for being late must explicitly reference traffic conditions.
- The tone must be polite and professional.
- The agent must stop before pressing send."""

EVIDENCE_QUESTIONS = """- Does the email include the correct recipient address (manager@example.com)?
- Does the email clearly explain the reason for being late as traffic conditions?
- Is the tone of the email polite and professional?
- Did the agent stop before pressing send?"""

DIMENSION_RUBRICS = """#### A. Clarity Of Reason
How clearly the email explains the reason for being late.

5 — The email explicitly states traffic conditions as the reason for being late in a clear and concise manner.
4 — The email states traffic conditions as the reason for being late, but with minor ambiguity or unnecessary details.
3 — The email mentions traffic conditions but lacks clarity or includes irrelevant information.
2 — The email vaguely references being late without clearly attributing it to traffic conditions.
1 — The email does not mention traffic conditions as the reason for being late.

#### B. Tone And Politeness
How polite and professional the tone of the email is.

5 — The email uses consistently polite and professional language throughout.
4 — The email is mostly polite and professional, with minor lapses in tone.
3 — The email is somewhat polite but includes noticeable lapses in professionalism.
2 — The email is minimally polite and lacks professionalism in tone.
1 — The email is rude or unprofessional in tone.

#### C. Adherence To Instructions
How well the agent follows the task instructions.

5 — The agent fully adheres to all instructions, including stopping before pressing send.
4 — The agent adheres to most instructions but with minor deviations.
3 — The agent partially adheres to instructions, missing key details or steps.
2 — The agent minimally adheres to instructions, ignoring significant aspects.
1 — The agent does not adhere to the instructions at all.

#### D. Grammar And Spelling
The grammatical correctness and spelling accuracy of the email.

5 — The email is free of grammatical errors and spelling mistakes.
4 — The email has one or two minor grammatical errors or spelling mistakes.
3 — The email contains several noticeable grammatical errors or spelling mistakes.
2 — The email has frequent grammatical errors or spelling mistakes that affect readability.
1 — The email is riddled with grammatical errors and spelling mistakes, making it difficult to understand."""

DIMENSION_WEIGHTS = {
    'clarity_of_reason': 0.3,
    'tone_and_politeness': 0.3,
    'adherence_to_instructions': 0.2,
    'grammar_and_spelling': 0.2,
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
