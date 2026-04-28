import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Please compose an email to David Sullivan (dgs@bu.edu). Tell him that the one-day basic Java class he requested last time is available on either November 29th or November 30th. Since the weather has gotten colder, write politely and include a reminder to be careful not to catch a cold. Stop just before pressing the send button — do not actually send it.""")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Recipient: David Sullivan at dgs@bu.edu (exact address required)
- Dates: both November 29th and November 30th must appear as options
- Tone: polite and warm
- Required content: cold weather health reminder ("be careful not to catch a cold" or equivalent)
- Critical constraint: agent must stop BEFORE pressing send — the email must NOT actually be sent

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question:

- Was an email composed (in a mail client, as a draft, or as written text)?
- Was the recipient David Sullivan / dgs@bu.edu?
- Were both November 29th and November 30th mentioned as options?
- Was a polite and warm tone used throughout?
- Was the cold weather health reminder included?
- Did the agent stop before sending — is there any "send" action in the trace?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. Draft Compliance (0.50)
Did the agent compose the email and stop before sending?

5 — Email fully composed with correct recipient; agent explicitly stopped before pressing send (confirmed by trace or stated in response).
4 — Email composed and not sent, but one minor element missing (e.g. one date option omitted).
3 — Email partially composed, or agent's stopping point before send is unclear.
2 — Email composed but sent (violates the hard constraint), or major content missing.
1 — No email composed, or task abandoned.

#### B. Email Quality (0.25)
Was the email correctly addressed, appropriately toned, and fully contentful?

5 — Correct recipient (dgs@bu.edu), both dates mentioned, polite/warm tone throughout, cold weather reminder included.
4 — Mostly correct; one minor content element missing or tone slightly off.
3 — Core content present (dates, Java class) but tone issues or health reminder absent.
2 — Significant content errors (wrong recipient, wrong dates, missing health reminder).
1 — Email content bears no relation to the task.

#### C. Content Requirements (0.25)
Did the agent address every required element of the task?

5 — All four elements present: correct recipient, both date options, polite tone, cold weather reminder.
4 — Three of four elements present.
3 — Two of four elements present.
2 — Only one element addressed.
1 — None of the required elements addressed.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "dimension_reasoning": {{
    "draft_compliance": "<one sentence citing specific evidence>",
    "email_quality": "<one sentence citing specific evidence>",
    "content_requirements": "<one sentence citing specific evidence>"
  }},
  "draft_compliance": <1-5>,
  "email_quality": <1-5>,
  "content_requirements": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "draft_compliance":    0.50,
    "email_quality":       0.25,
    "content_requirements": 0.25,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
