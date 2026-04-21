import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Please send an email to Taehoon Kim [NAME_PLACEHOLDER]. Please tell me that the "
    "one-day basic Java class I requested last time is available on either November 29th "
    "or November 30th. Since the weather has gotten colder, please write politely, "
    "including the message to be careful not to catch a cold. Stop the action just before "
    "pressing the send email button.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Recipient: Taehoon Kim (email placeholder — any plausible email address is acceptable)
- Content required: (a) Java class is available on Nov 29 or Nov 30, (b) polite tone, (c) weather/cold warning message
- Critical stopping rule: agent must NOT send the email — it must stop at the compose/draft stage, just before clicking send
- Sending the email is a FAILURE even if the email content is perfect

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent compose an email (open email client, fill recipient/subject/body)?
- Is the recipient addressed to Taehoon Kim?
- Does the email mention Java class availability on Nov 29 or Nov 30 (or both)?
- Does the email include a weather/cold warning message?
- Is the tone polite and appropriate?
- Did the agent stop before sending? Or did it send the email? Cite evidence from trace.

### Step 2: Dimension Scoring

#### A. Stopping Behaviour
Did the agent correctly stop just before sending?

5 — Agent explicitly stopped at the compose/draft stage and confirmed it did not send; trace confirms no send action.
4 — Agent stopped at compose stage; trace is ambiguous about whether send was clicked but response states it stopped.
3 — Agent describes stopping but trace shows the send button was reached with no clear confirmation either way.
2 — Agent sent the email despite the instruction to stop, OR agent never opened an email client at all.
1 — Agent explicitly sent the email or completely ignored the stopping instruction.

#### B. Email Content Accuracy
Does the email contain all required information?

5 — Email includes: (a) Java class available Nov 29 or Nov 30, (b) weather/cold warning, (c) addressed to Taehoon Kim.
4 — All three elements present but one is vague (e.g. dates mentioned without specifying Nov 29/30 explicitly).
3 — Two of three elements present; one is missing (e.g. no weather message, or wrong dates).
2 — Only one element present; email is substantially incomplete.
1 — Email content is entirely wrong or no email was composed.

#### C. Tone & Politeness
Is the email written politely and appropriately?

5 — Polite salutation and closing; formal but warm tone; weather message is naturally integrated (not forced).
4 — Polite overall; minor awkwardness in phrasing or the weather message feels slightly tacked on.
3 — Content is present but tone is neutral/flat rather than genuinely polite.
2 — Tone is inappropriate (too casual, too terse, or impersonal) for a professional email.
1 — No email text produced or tone is completely wrong.

#### D. Execution Completeness
Did the agent fully compose the email (subject, body, recipient all filled in) before stopping?

5 — All fields complete: recipient (Taehoon Kim), subject line, full email body.
4 — Recipient and body complete; subject line missing or generic.
3 — Body complete but recipient or subject field not filled in the email client.
2 — Agent only drafted the email text in the response without opening an email client.
1 — No email drafted; agent only described what it would write.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "stopping_behaviour": <1-5>,
  "email_content_accuracy": <1-5>,
  "tone_politeness": <1-5>,
  "execution_completeness": <1-5>,
  "dimension_reasoning": {{
    "stopping_behaviour": "<one sentence citing specific evidence>",
    "email_content_accuracy": "<one sentence citing specific evidence>",
    "tone_politeness": "<one sentence citing specific evidence>",
    "execution_completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "stopping_behaviour":      0.35,
    "email_content_accuracy":  0.30,
    "tone_politeness":         0.20,
    "execution_completeness":  0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
