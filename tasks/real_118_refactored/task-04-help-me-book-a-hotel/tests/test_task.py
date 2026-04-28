import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Help me book a hotel near Abisko, Sweden. "
    "The budget is 800-1000. Requirements: close to the lake, within walking "
    "distance, and with high hygiene standards.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Budget: 800–1000 (local currency) (hard constraint — any hotel clearly outside this range is a failure)
- Location: must be near Lake Torneträsk / Abisko area, Sweden — walking distance to the lake is explicitly required
- Hygiene: hotel must have evidence of high hygiene standards (rating, review mentions, or cleanliness score)
- Action required: the task says "book" — the agent should attempt an actual booking or at minimum identify a specific bookable option and initiate/complete the booking process

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for hotels or navigate a booking platform? Cite evidence.
- What hotel(s) did the agent identify? List names and prices if mentioned.
- Is the price within 800–1000 (local currency)? Is walking distance to Lake Torneträsk / Abisko area stated?
- Is there any evidence of hygiene quality (cleanliness rating, review mention, platform hygiene score)?
- Did the agent attempt or complete a booking, or only recommend options?
- Did the agent stop short and ask for clarification instead of completing the task?

### Step 2: Dimension Scoring

#### A. Constraint Satisfaction
Did the agent respect budget, location, and hygiene constraints?

5 — Selected hotel is priced 800–1000 (local currency), explicitly within walking distance of Lake Torneträsk / Abisko area, and has evidence of high hygiene.
4 — All three constraints addressed but one is weakly supported (e.g. hygiene inferred from star rating alone).
3 — Two of three constraints clearly met; one is missing or ambiguous.
2 — Only one constraint met; location or budget is clearly violated.
1 — Constraints ignored; no evidence hotel meets any requirement.

#### B. Platform & Search Execution
Did the agent actively search a booking platform to find options?

5 — Agent searched a real booking platform (e.g. Booking.com, Hotels.com, Expedia, etc.) and retrieved live listings.
4 — Agent used a platform but only accessed 1 listing or search results without drilling into details.
3 — Agent performed a web search but not on a dedicated booking platform; results may be outdated.
2 — Agent described what to search for without actually searching.
1 — No search performed; response is from prior knowledge or hallucinated.

#### C. Booking Action
Did the agent attempt or complete the booking as requested?

5 — Agent completed or formally initiated a booking (confirmation page, booking reference, or booking form filled).
4 — Agent reached the booking page for a specific hotel and stopped just before confirming (e.g. awaiting user payment details).
3 — Agent identified a specific bookable option with a direct link or clear next step for the user.
2 — Agent listed options but made no attempt to proceed with booking.
1 — Agent only gave general advice or refused to attempt booking.

#### D. Response Usefulness
Is the output actionable and well-organised for someone who wants to book a hotel?

5 — Clear recommendation with: hotel name, price, location relative to Lake Torneträsk / Abisko area, hygiene evidence, and booking link or next step.
4 — Most details present; one piece of information missing (e.g. no direct booking link).
3 — Hotel identified but details are thin; user would need to do significant follow-up research.
2 — Output is vague or lists many options without a clear recommendation.
1 — Output is a refusal, error, or contains no actionable hotel information.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "constraint_satisfaction": "<one sentence citing specific evidence>",
    "platform_execution": "<one sentence citing specific evidence>",
    "booking_action": "<one sentence citing specific evidence>",
    "response_usefulness": "<one sentence citing specific evidence>"
  }},
  "constraint_satisfaction": <1-5>,
  "platform_execution": <1-5>,
  "booking_action": <1-5>,
  "response_usefulness": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_execution": 0.35,
    "constraint_satisfaction": 0.28,
    "booking_action": 0.24,
    "response_usefulness": 0.13,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
