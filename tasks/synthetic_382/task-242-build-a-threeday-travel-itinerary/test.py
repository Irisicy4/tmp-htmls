"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Build a three-day travel itinerary for Paris in December using a Google Sheets template.
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


TASK_INSTRUCTION = """Build a three-day travel itinerary for Paris in December using a Google Sheets template. Include day-by-day activities such as visiting landmarks, dining recommendations, and estimated costs for each activity."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a three-day travel itinerary for Paris in December. The itinerary must include day-by-day activities such as visiting landmarks, dining recommendations, and estimated costs for each activity. The agent must use a Google Sheets template and gather information from relevant platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Build a three-day travel itinerary for Paris in December using a Google Sheets template. Include day-by-day activities such as visiting landmarks, dining recommendations, and estimated costs for each activity.

## Task-Specific Constraints
- Must use Google Sheets to structure the itinerary.
- Must gather information from tripadvisor.com and parisinfo.com.
- Must include at least three landmarks in the itinerary.
- Must provide estimated costs for all activities and dining recommendations.
- Must organize the output as a clear table in the Google Sheets template.
- Must ensure the itinerary is feasible for December weather and conditions.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tripadvisor.com and parisinfo.com to gather information?
- Did the agent use Google Sheets to structure the itinerary?
- Are at least three landmarks included in the itinerary?
- Are estimated costs provided for all activities and dining recommendations?
- Is the output organized as a clear table in the Google Sheets template?

### Step 2: Dimension Scoring

#### A. Itinerary Completeness (0.35)
Measures whether the itinerary includes all required elements: landmarks, dining recommendations, and estimated costs.

5 — Includes at least three landmarks, dining recommendations, and costs for all activities.
4 — Includes most required elements but misses minor details.
3 — Includes some required elements but lacks significant details.
2 — Includes few required elements and is largely incomplete.
1 — Does not include any required elements.

#### B. Platform Usage Accuracy (0.30)
Measures whether the agent correctly used the required platforms to gather information.

5 — Gathers information from both tripadvisor.com and parisinfo.com.
4 — Gathers information from one platform fully and partially from the other.
3 — Gathers information from only one platform.
2 — Attempts to gather information but fails to use the platforms effectively.
1 — Does not use any required platforms.

#### C. Detail Specificity (0.20)
Measures the depth and specificity of the itinerary, including accurate cost estimates and feasibility for December.

5 — Provides detailed cost estimates and ensures feasibility for December weather.
4 — Provides good cost estimates but lacks minor feasibility details.
3 — Provides some cost estimates but lacks depth or feasibility considerations.
2 — Provides vague or incomplete details.
1 — Provides no specific details.

#### D. Output Organization (0.15)
Measures the clarity and structure of the itinerary in the Google Sheets template.

5 — Output is organized as a clear, well-formatted table in the template.
4 — Output is mostly organized but has minor formatting issues.
3 — Output is partially organized but lacks clarity.
2 — Output is poorly organized and difficult to follow.
1 — Output is not organized in the required format.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "The agent used Google Sheets and visited tripadvisor.com and parisinfo.com. The itinerary includes three landmarks, dining recommendations, and estimated costs, organized as a table.",
  "itinerary_completeness": 5,
  "platform_usage_accuracy": 5,
  "detail_specificity": 4,
  "output_organization": 5,
  "dimension_reasoning": {{
    "itinerary_completeness": "Includes all required elements: landmarks, dining recommendations, and costs.",
    "platform_usage_accuracy": "Successfully gathered information from both required platforms.",
    "detail_specificity": "Provides detailed cost estimates but minor feasibility details are missing.",
    "output_organization": "Output is organized as a clear, well-formatted table in the template."
  }},
  "overall_score": 4.75,
  "passed": true
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "itinerary_completeness": 0.35,
    "platform_usage_accuracy": 0.30,
    "detail_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())