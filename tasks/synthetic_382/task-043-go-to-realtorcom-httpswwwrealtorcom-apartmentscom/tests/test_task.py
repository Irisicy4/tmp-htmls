"""
LLM-as-judge evaluator for EvolveBench task.

Category: Real Estate
Task: Search for rental apartments in Denver, CO, meeting specific criteria, and extract key details from the most recent listings.
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


TASK_INSTRUCTION = """Go to Realtor.com (https://www.realtor.com/), Apartments.com (https://www.apartments.com/), and Zumper (https://www.zumper.com/) to search for rental apartments in Denver, CO, with 2+ bedrooms, 2+ bathrooms, and monthly rent under $3,000. Extract the five most recently listed properties and record: address, rent price, number of bedrooms and bathrooms, date listed, and amenities (e.g., parking, pool, pet-friendly)."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to search for rental apartments in Denver, CO, on three specific platforms (Realtor.com, Apartments.com, and Zumper). The agent must extract details about the five most recently listed properties that meet specific criteria (2+ bedrooms, 2+ bathrooms, rent under $3,000). The output must include address, rent price, number of bedrooms and bathrooms, date listed, and amenities.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Realtor.com (https://www.realtor.com/), Apartments.com (https://www.apartments.com/), and Zumper (https://www.zumper.com/) to search for rental apartments in Denver, CO, with 2+ bedrooms, 2+ bathrooms, and monthly rent under $3,000. Extract the five most recently listed properties and record: address, rent price, number of bedrooms and bathrooms, date listed, and amenities (e.g., parking, pool, pet-friendly).

## Task-Specific Constraints
- Must visit all three specified platforms: Realtor.com, Apartments.com, and Zumper.
- Must extract details for exactly five properties that meet the specified criteria.
- Each property must include: address, rent price, number of bedrooms, number of bathrooms, date listed, and amenities.
- The output must be organized in a structured format, such as a table or JSON.
- The extracted data must match the task criteria (e.g., rent under $3,000, 2+ bedrooms, 2+ bathrooms).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (Realtor.com, Apartments.com, Zumper)?
- Did the agent extract data for exactly five properties that meet the specified criteria?
- Does the response include all required details (address, rent price, number of bedrooms, number of bathrooms, date listed, and amenities) for each property?
- Is the output organized in a structured format (e.g., table or JSON)?
- Are the extracted details accurate and consistent with the task criteria?

### Step 2: Dimension Scoring

#### A. Data Completeness (0.35)
Measures whether the agent provided all required details for each property.

5 — All five properties include all required details (address, rent price, bedrooms, bathrooms, date listed, amenities).
4 — Four properties include all required details; minor omissions in the fifth.
3 — Three properties include most required details; others have significant omissions.
2 — Only one or two properties include required details; most are incomplete.
1 — No properties include the required details.

#### B. Platform Coverage (0.30)
Measures whether the agent used all three specified platforms.

5 — The agent successfully used all three platforms and extracted data from each.
4 — The agent used two platforms and extracted data from both.
3 — The agent used one platform and extracted data from it.
2 — The agent attempted to use the platforms but failed to extract data.
1 — The agent did not attempt to use any of the specified platforms.

#### C. Criteria Adherence (0.20)
Measures whether the extracted properties meet the specified criteria (2+ bedrooms, 2+ bathrooms, rent under $3,000).

5 — All five properties meet the criteria.
4 — Four properties meet the criteria; one minor error.
3 — Three properties meet the criteria; others have significant errors.
2 — One or two properties meet the criteria; others do not.
1 — None of the properties meet the criteria.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Output is fully structured (e.g., table or JSON) and easy to read.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured but difficult to interpret.
2 — Output is unstructured and difficult to follow.
1 — Output is completely unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "criteria_adherence": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "data_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "criteria_adherence": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_completeness": 0.35,
    "platform_coverage": 0.30,
    "criteria_adherence": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())