import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Search on Naver for 3 famous hair salons in Gangnam, Seoul, and book the one that offers the best value. I want to get a men's perm.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Naver (Korean search engine — naver.com)
- Location: Gangnam, Seoul (강남, 서울)
- Required count: 3 salons found
- Service: men's perm specifically
- Action: must attempt to book the best-value salon (not just list them)
- Value assessment: agent must compare options (price/reputation) and justify its choice

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent navigate Naver? Cite evidence.
- How did the agent search for salons in Gangnam, Seoul on Naver?
- Were 3 salon options identified (from any source)?
- Did the agent compare options and select best value?
- Was a booking attempted? What method (phone, website, app)?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent attempt to use Naver as instructed?

5 — Agent navigated Naver, searched for Gangnam, Seoul hair salons, and found results or clearly described the process.
4 — Agent used Naver but found limited results; supplemented with another platform.
3 — Agent attempted Naver but immediately switched to a different platform without explanation.
2 — Agent used a completely different platform (not Naver) without acknowledgement.
1 — No search performed.

#### B. Salon Discovery
Did the agent identify 3 named hair salons relevant to the task?

5 — 3 specific, named salons in Gangnam, Seoul found with location and men's perm pricing or service info.
4 — 3 salons found but one lacks specific location or pricing detail.
3 — 2 salons found with detail; third is vague or unverified.
2 — Only 1 salon found, or salons found are generic (not named establishments).
1 — No specific salons identified.

#### C. Value Assessment & Selection
Did the agent compare salons and select the best value with justification?

5 — Agent compared 2+ salons on price and/or reputation, selected one, and justified the choice (e.g. "salon X offers men's perm at ¥X with good reviews").
4 — Agent selected a salon with brief justification; comparison is implicit.
3 — Agent selected a salon without comparing alternatives or without justification.
2 — Agent listed salons but did not make a selection.
1 — No selection made.

#### D. Booking Attempt
Did the agent attempt to book the selected salon?

5 — Booking attempted via appropriate channel (website, phone, reservation app) with confirmation or next steps described.
4 — Booking attempted but outcome unclear or channel is indirect (e.g. found contact info but did not call/submit).
3 — Agent identified booking method but did not execute it.
2 — Agent noted that booking is required but took no action.
1 — No booking attempt of any kind.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "platform_execution": <1-5>,
  "salon_discovery": <1-5>,
  "value_assessment": <1-5>,
  "booking_attempt": <1-5>,
  "dimension_reasoning": {{"platform_execution": "<one sentence>", "salon_discovery": "<one sentence>", "value_assessment": "<one sentence>", "booking_attempt": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {"platform_execution": 0.20, "salon_discovery": 0.25, "value_assessment": 0.25, "booking_attempt": 0.30}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
