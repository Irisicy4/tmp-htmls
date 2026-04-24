"""LLM-as-judge evaluator for task-98-send-the-following-text-message.

Category: Daily Activities
Task: Send the following text message: 'Thank you, man' to the contact Nexora Systems on WhatsApp.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "Send the following text message: 'Thank you, man' to the contact Nexora Systems on WhatsApp."

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves opening WhatsApp Web or the WhatsApp desktop app, finding a specific contact, and sending a predefined text message.'

CONSTRAINTS = """- Platform: WhatsApp specifically
- Contact: Nexora Systems
- Message: exactly 'Thank you, man'
- Must confirm message was sent"""

EVIDENCE_QUESTIONS = """- Did the agent open WhatsApp (web or desktop)?
- Was the target contact found?
- Was the message typed correctly?
- Was the message sent and confirmed?"""

DIMENSION_RUBRICS = """#### A. Platform Access (0.25)
Did the agent access WhatsApp?

5 — Agent opened WhatsApp Web or desktop app successfully.
4 — WhatsApp accessed but with difficulty.
3 — Agent described WhatsApp steps without accessing it.
2 — Agent suggested using WhatsApp without navigating.
1 — No WhatsApp access.

#### B. Contact Location (0.25)
Was the correct contact found?

5 — Target contact found by name search in WhatsApp.
4 — Contact found but required scrolling without search.
3 — Contact searched but not found; agent attempted alternative.
2 — Contact not found.
1 — No contact search.

#### C. Message Accuracy (0.3)
Was the exact message typed and sent?

5 — Exact message 'Thank you, man' typed and send button clicked.
4 — Message sent but with minor variation.
3 — Message typed but not confirmed sent.
2 — Message entered incorrectly.
1 — No message sent.

#### D. Send Confirmation (0.2)
Was delivery confirmed?

5 — Sent confirmation visible (double tick, 'Sent' status, or timestamp).
4 — Agent confirmed send without visible status evidence.
3 — Send assumed but not confirmed.
2 — Send failed.
1 — No confirmation."""

DIMENSION_WEIGHTS = {
    'platform_access': 0.25,
    'contact_location': 0.25,
    'message_accuracy': 0.3,
    'send_confirmation': 0.2,
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
