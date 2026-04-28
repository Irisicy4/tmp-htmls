import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Compose an email to my manager (manager@example.com) telling them I will be "
    "over an hour late due to traffic conditions. Write it politely. Stop just before "
    "pressing send — do not actually send it.")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Recipient: manager@example.com (exact address required)
- Required content: (a) will be over an hour late, (b) reason is traffic conditions
- Tone: polite and professional
- Critical stopping rule: agent must stop BEFORE pressing send — the email must NOT be sent

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question:

- Was an email composed (in a mail client, as a draft, or as written text)?
- Was the recipient set to manager@example.com?
- Did the email mention being over an hour late?
- Did the email cite traffic conditions as the reason?
- Was the tone polite and professional?
- Did the agent stop before sending — is there any "send" action in the trace?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. Draft Compliance (0.40)
Did the agent compose the email and stop before sending?

5 — Email composed and agent explicitly stopped before send; trace confirms no send action.
4 — Email composed and not sent, but stopping point is slightly ambiguous in the trace.
3 — Email partially composed, or whether send was clicked is unclear.
2 — Email was sent despite the instruction to stop, OR no email was composed.
1 — Agent ignored the stopping constraint entirely or abandoned the task.

#### B. Message Accuracy (0.30)
Does the email contain the required information?

5 — Correct recipient (manager@example.com), explicitly states over an hour late, cites traffic conditions.
4 — All elements present but one is vague (e.g. "late" without specifying over an hour, or traffic only implied).
3 — Two of three required elements present; one missing.
2 — Only one required element present.
1 — Email content is entirely wrong or no email was composed.

#### C. Tone & Politeness (0.20)
Is the email written politely and professionally?

5 — Polite salutation and closing; professional tone throughout; appropriately apologetic without being excessive.
4 — Polite overall with minor tone issues.
3 — Neutral/flat tone; functional but not genuinely polite.
2 — Tone is inappropriate (too casual, terse, or unprofessional) for a manager communication.
1 — No email text produced or tone is completely wrong.

#### D. Execution Completeness (0.10)
Did the agent fully compose the email (recipient, subject, body all filled)?

5 — All fields complete: recipient (manager@example.com), subject line, full email body.
4 — Recipient and body complete; subject line missing or generic.
3 — Body drafted but recipient or subject not filled in the email client.
2 — Agent only wrote email text in its response without opening an email client.
1 — No email drafted; agent only described what it would write.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "dimension_reasoning": {{
    "draft_compliance": "<one sentence citing specific evidence>",
    "message_accuracy": "<one sentence citing specific evidence>",
    "tone_politeness": "<one sentence citing specific evidence>",
    "execution_completeness": "<one sentence citing specific evidence>"
  }},
  "draft_compliance": <1-5>,
  "message_accuracy": <1-5>,
  "tone_politeness": <1-5>,
  "execution_completeness": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "draft_compliance":      0.40,
    "message_accuracy":      0.30,
    "tone_politeness":       0.20,
    "execution_completeness": 0.10,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
