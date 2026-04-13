"""
LLM-as-judge evaluator for EvolveBench task-65.

Category: Travel & Planning
Task: Find recommended dinner restaurants in Tuscany, Italy that are open on December 26th and 27th.
"""

import os, json, re
PASS_THRESHOLD = 3.0

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse(text):
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

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_INSTRUCTION,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
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
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)])
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }


TASK_INSTRUCTION = """Find recommended dinner restaurants in Tuscany, Italy that are open on December 26th and 27th."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves finding dinner restaurant recommendations in Tuscany, Italy, with confirmed availability on specific dates (December 26-27).

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Location: Tuscany, Italy
- Meal type: dinner specifically
- Dates: both December 26th AND 27th — restaurants must be confirmed open on both days
- Source: must use a credible restaurant discovery platform (TripAdvisor, Google Maps, TheFork, etc.)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for restaurants in Tuscany, Italy?
- Were dinner-specific results returned?
- Were December 26 and 27 opening hours verified?
- How many restaurants were recommended?
- What platform was used?

### Step 2: Dimension Scoring

#### A. Search Execution (0.2)
Did the agent search for Tuscany restaurants on a credible platform?

5 — Used TripAdvisor, Google Maps, TheFork, or equivalent to search Tuscany, Italy dinner restaurants.
4 — Used a credible platform but search was less targeted.
3 — Found results via general web search without a restaurant platform.
2 — Used an unsuitable platform.
1 — No restaurant search performed.

#### B. Date Verification (0.35)
Were December 26 and 27 opening hours verified?

5 — Both dates explicitly verified as open for each recommended restaurant.
4 — Opening hours checked but one date not explicitly confirmed.
3 — Regular weekly hours checked without confirming those specific dates.
2 — Restaurants recommended without any date verification.
1 — No date verification.

#### C. Recommendation Quality (0.3)
Are the restaurant recommendations high-quality?

5 — 3+ restaurants with name, cuisine type, address, hours, and why recommended.
4 — 2-3 restaurants with most details.
3 — Restaurants named but details thin.
2 — Only one restaurant or very generic recommendations.
1 — No specific recommendations.

#### D. Dinner Focus (0.15)
Are recommendations specifically for dinner?

5 — All restaurants confirmed open for dinner with dinner service hours.
4 — Dinner service implied but not explicitly stated.
3 — Restaurants are general (lunch+dinner) without dinner-specific confirmation.
2 — Mix of dinner and non-dinner options.
1 — Not dinner-focused.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "search_execution": <1-5>,
  "date_verification": <1-5>,
  "recommendation_quality": <1-5>,
  "dinner_focus": <1-5>,
  "dimension_reasoning": {{
    "search_execution": "<one sentence citing specific evidence>",
    "date_verification": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
    "dinner_focus": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "search_execution": 0.2,
    "date_verification": 0.35,
    "recommendation_quality": 0.3,
    "dinner_focus": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())