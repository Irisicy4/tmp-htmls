import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Send the following text message: 'Thank you, man' to the contact Nexora Systems on WhatsApp.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: WhatsApp specifically
- Contact: Nexora Systems
- Message: exactly 'Thank you, man'
- Must confirm message was sent

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent open WhatsApp (web or desktop)?
- Was the target contact found?
- Was the message typed correctly?
- Was the message sent and confirmed?

### Step 2: Dimension Scoring

#### A. Platform Access (0.35)
Did the agent access WhatsApp?

5 — Agent opened WhatsApp Web or desktop app successfully.
4 — WhatsApp accessed but with difficulty.
3 — Agent described WhatsApp steps without accessing it.
2 — Agent suggested using WhatsApp without navigating.
1 — No WhatsApp access.

#### B. Contact Location (0.22)
Was the correct contact found?

5 — Target contact found by name search in WhatsApp.
4 — Contact found but required scrolling without search.
3 — Contact searched but not found; agent attempted alternative.
2 — Contact not found.
1 — No contact search.

#### C. Message Accuracy (0.26)
Was the exact message typed and sent?

5 — Exact message 'Thank you, man' typed and send button clicked.
4 — Message sent but with minor variation.
3 — Message typed but not confirmed sent.
2 — Message entered incorrectly.
1 — No message sent.

#### D. Send Confirmation (0.17)
Was delivery confirmed?

5 — Sent confirmation visible (double tick, 'Sent' status, or timestamp).
4 — Agent confirmed send without visible status evidence.
3 — Send assumed but not confirmed.
2 — Send failed.
1 — No confirmation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "platform_access": "<one sentence citing specific evidence>",
    "contact_location": "<one sentence citing specific evidence>",
    "message_accuracy": "<one sentence citing specific evidence>",
    "send_confirmation": "<one sentence citing specific evidence>",
  }},
  "platform_access": <1-5>,
  "contact_location": <1-5>,
  "message_accuracy": <1-5>,
  "send_confirmation": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_access": 0.35,
    "contact_location": 0.22,
    "message_accuracy": 0.26,
    "send_confirmation": 0.17,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
