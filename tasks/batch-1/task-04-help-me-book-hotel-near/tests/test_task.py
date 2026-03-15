"""
LLM-as-judge evaluator for EvolveBench task-04.

Category: Travel & Planning
Task: "Help me book a hotel near the West Lake in Hangzhou near the city.
       The budget is 800-1000. Requirements: close to the lake, within walking
       distance, and with high hygiene standards."
"""

import os, json, re

TASK_INSTRUCTION = (
    "Help me book a hotel near the West Lake in Hangzhou near the city. "
    "The budget is 800-1000. Requirements: close to the lake, within walking "
    "distance, and with high hygiene standards."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based hotel booking or hotel search task.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Budget: 800–1000 CNY per night (hard constraint — any hotel clearly outside this range is a failure)
- Location: must be near West Lake (西湖), Hangzhou — walking distance to the lake is explicitly required
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
- Is the price within 800–1000 CNY? Is walking distance to West Lake stated?
- Is there any evidence of hygiene quality (cleanliness rating, review mention, platform hygiene score)?
- Did the agent attempt or complete a booking, or only recommend options?
- Did the agent stop short and ask for clarification instead of completing the task?

### Step 2: Dimension Scoring

#### A. Constraint Satisfaction
Did the agent respect budget, location, and hygiene constraints?

5 — Selected hotel is priced 800–1000 CNY, explicitly within walking distance of West Lake, and has evidence of high hygiene.
4 — All three constraints addressed but one is weakly supported (e.g. hygiene inferred from star rating alone).
3 — Two of three constraints clearly met; one is missing or ambiguous.
2 — Only one constraint met; location or budget is clearly violated.
1 — Constraints ignored; no evidence hotel meets any requirement.

#### B. Platform & Search Execution
Did the agent actively search a booking platform to find options?

5 — Agent searched a real booking platform (e.g. Ctrip, Booking.com, Meituan) and retrieved live listings.
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

5 — Clear recommendation with: hotel name, price, location relative to West Lake, hygiene evidence, and booking link or next step.
4 — Most details present; one piece of information missing (e.g. no direct booking link).
3 — Hotel identified but details are thin; user would need to do significant follow-up research.
2 — Output is vague or lists many options without a clear recommendation.
1 — Output is a refusal, error, or contains no actionable hotel information.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "constraint_satisfaction": <1-5>,
  "platform_execution": <1-5>,
  "booking_action": <1-5>,
  "response_usefulness": <1-5>,
  "dimension_reasoning": {{
    "constraint_satisfaction": "<one sentence citing specific evidence>",
    "platform_execution": "<one sentence citing specific evidence>",
    "booking_action": "<one sentence citing specific evidence>",
    "response_usefulness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "constraint_satisfaction": 0.35,
    "platform_execution":      0.20,
    "booking_action":          0.30,
    "response_usefulness":     0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
    return ""

def _parse_answer_tag(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }