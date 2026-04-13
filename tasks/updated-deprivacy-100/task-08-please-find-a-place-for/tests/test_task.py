"""
LLM-as-judge evaluator for EvolveBench task-08.

Category: Travel & Planning
Task: "Please find a place for a date in Sydney, Australia on December 25, 2025.
       I wish there was a place with easy parking."
"""

import os, json, re

TASK_INSTRUCTION = (
    "Please find a place for a date in Sydney, Australia on December 25, 2025. "
    "I wish there was a place with easy parking."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully found suitable date spots in Sydney, Australia for Christmas Day (December 25, 2025).

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Location: Sydney, Australia — recommendations must be in Sydney or Greater Sydney area
- Date: December 25, 2025 (Christmas Day) — venues must be open or accessible on a public holiday
- Parking: easy parking is explicitly requested — recommendations should include parking availability or nearby parking options
- Output: the agent should present specific venue recommendations, not just generic advice

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for date spots or navigate a relevant platform? Cite evidence.
- What specific venues did the agent recommend? List names and locations if mentioned.
- Are the venues in Sydney or Greater Sydney area?
- Did the agent address Christmas Day availability (December 25 is a public holiday in Australia)?
- Did the agent address parking availability at or near the recommended venues?

### Step 2: Dimension Scoring

#### A. Location Relevance
Are the recommended venues actually in Sydney or Greater Sydney area?

5 — All recommendations are clearly in Sydney or Greater Sydney, with specific suburb or address details.
4 — Recommendations are Sydney-based but location details are vague (e.g. no suburb or address).
3 — Most recommendations are Sydney-based but one or more are unclear or outside the area.
2 — Recommendations are for Australia generally without Sydney specificity.
1 — Recommendations are not in Sydney or are entirely generic with no location detail.

#### B. Christmas Day Availability
Did the agent address whether venues are open or suitable on December 25?

5 — Agent explicitly verified or noted Christmas Day operating hours or suitability for each venue.
4 — Agent acknowledged Christmas Day context and noted most venues should be accessible, with some specifics.
3 — Agent mentioned it is Christmas Day but did not verify individual venue availability.
2 — Date noted without any holiday context or availability check.
1 — Christmas Day context ignored entirely.

#### C. Parking Coverage
Did the agent address the easy parking requirement?

5 — Parking availability or nearby parking options explicitly noted for recommended venues.
4 — Parking mentioned for most venues; one or two lack parking detail.
3 — Parking mentioned in general terms but not tied to specific venues.
2 — Parking briefly mentioned without actionable detail.
1 — Parking requirement not addressed.

#### D. Recommendation Quality
Are the recommendations specific, useful, and suitable for a date?

5 — 3+ specific venue recommendations with name, location, why it suits a date, and parking info.
4 — 3+ venues named with most key details; one element missing per venue.
3 — 2–3 venues mentioned with partial details; user would need to do follow-up research.
2 — 1–2 venues mentioned or recommendations are vague.
1 — No specific venues; only generic advice or refusal.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "location_relevance": <1-5>,
  "christmas_availability": <1-5>,
  "parking_coverage": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{
    "location_relevance": "<one sentence citing specific evidence>",
    "christmas_availability": "<one sentence citing specific evidence>",
    "parking_coverage": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "location_relevance":      0.30,
    "christmas_availability":  0.25,
    "parking_coverage":        0.25,
    "recommendation_quality":  0.20,
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
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

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