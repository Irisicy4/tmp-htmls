"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Create a 5-day itinerary for a trip to Kyoto, Japan including attractions, admission fees, and travel costs.
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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

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


TASK_INSTRUCTION = """Create a 5-day itinerary for a trip to Kyoto, Japan using Google Sheets. Include day-by-day plans for attractions such as Kinkaku-ji, Fushimi Inari Taisha, and Nishiki Market. Research the approximate admission fees and travel costs for each attraction on the Japan Guide website and Kyoto city's tourism board."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a detailed 5-day itinerary for a trip to Kyoto, Japan. The itinerary must include plans for visiting specific attractions, approximate admission fees, and travel costs. The agent must use Google Sheets to organize the itinerary and gather information from Japan Guide and Kyoto city's tourism board.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a 5-day itinerary for a trip to Kyoto, Japan using Google Sheets. Include day-by-day plans for attractions such as Kinkaku-ji, Fushimi Inari Taisha, and Nishiki Market. Research the approximate admission fees and travel costs for each attraction on the Japan Guide website and Kyoto city's tourism board.

## Task-Specific Constraints
- Must use Google Sheets to organize the itinerary.
- Must research admission fees and travel costs for each attraction using Japan Guide and Kyoto city's tourism board.
- Must include plans for visiting Kinkaku-ji, Fushimi Inari Taisha, and Nishiki Market.
- Must provide a structured, day-by-day itinerary.
- Must include approximate costs for admission and travel for each day.
- Must visit at least 3 of the specified platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets, Japan Guide, and Kyoto city's tourism board?
- Does the response include a structured, day-by-day itinerary?
- Are admission fees and travel costs included for each attraction?
- Are the specified attractions (Kinkaku-ji, Fushimi Inari Taisha, Nishiki Market) present in the itinerary?
- Is the output organized in Google Sheets or a similar tabular format?

### Step 2: Dimension Scoring

#### A. Itinerary Completeness (0.35)
Measures whether the 5-day itinerary includes all required attractions and plans.

5 — Includes all specified attractions and plans for all 5 days.
4 — Includes most attractions and plans for at least 4 days.
3 — Includes some attractions and plans for at least 3 days.
2 — Includes few attractions and plans for less than 3 days.
1 — No meaningful itinerary provided.

#### B. Platform Usage Accuracy (0.30)
Measures whether the agent used the required platforms (Google Sheets, Japan Guide, Kyoto tourism board).

5 — Successfully used all 3 platforms and sourced data correctly.
4 — Used at least 2 platforms and sourced most data correctly.
3 — Used at least 1 platform and sourced some data correctly.
2 — Used platforms incorrectly or sourced minimal data.
1 — Did not use any required platforms.

#### C. Cost and Detail Specificity (0.25)
Measures whether admission fees and travel costs are included and detailed.

5 — Includes accurate costs for all attractions and travel.
4 — Includes costs for most attractions and travel.
3 — Includes costs for some attractions and travel.
2 — Includes minimal or inaccurate cost details.
1 — No cost details provided.

#### D. Output Organization (0.10)
Measures whether the response is structured and organized in Google Sheets or similar format.

5 — Output is fully organized in Google Sheets or equivalent tabular format.
4 — Output is mostly organized but lacks minor details.
3 — Output is partially organized but usable.
2 — Output is poorly organized and hard to use.
1 — Output is unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "itinerary_completeness": <1-5>,
  "platform_usage_accuracy": <1-5>,
  "cost_and_detail_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "itinerary_completeness": "<one sentence citing specific evidence>",
    "platform_usage_accuracy": "<one sentence citing specific evidence>",
    "cost_and_detail_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "itinerary_completeness": 0.35,
    "platform_usage_accuracy": 0.30,
    "cost_and_detail_specificity": 0.25,
    "output_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())