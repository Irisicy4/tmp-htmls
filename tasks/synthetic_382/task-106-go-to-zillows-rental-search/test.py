"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Extract detailed information about San Francisco apartments for rent with specific filters from Zillow.
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


TASK_INSTRUCTION = """Go to Zillow's rental search tool and extract detailed information about apartments available for rent in San Francisco, CA, with 2 bedrooms, under $3,000/month. Apply filters for pet-friendly properties, and collect data on the first 5 listings, including price, address, and key amenities."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Zillow's rental search tool to find apartments in San Francisco, CA, with 2 bedrooms, under $3,000/month, and pet-friendly. The agent must extract data for the first 5 listings, including price, address, and key amenities. A successful completion involves accurate data extraction and proper filtering.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Zillow's rental search tool and extract detailed information about apartments available for rent in San Francisco, CA, with 2 bedrooms, under $3,000/month. Apply filters for pet-friendly properties, and collect data on the first 5 listings, including price, address, and key amenities.

## Task-Specific Constraints
- Must use Zillow's rental search tool.
- Must apply filters for 2 bedrooms, under $3,000/month, and pet-friendly properties.
- Must extract data for the first 5 listings only.
- Must include price, address, and key amenities for each listing.
- Output must be structured as a table or a structured list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Zillow's rental search tool and apply the required filters?
- Did the agent extract data for exactly 5 listings?
- Are price, address, and key amenities included for each listing?
- Is the output structured as a table or a structured list?
- Are there any major inaccuracies or missing data in the extracted information?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the extracted data (price, address, and key amenities) is correct and matches the task requirements.

5 — All extracted data is accurate and matches the task requirements.
4 — Minor inaccuracies in 1-2 listings but overall correct.
3 — Noticeable inaccuracies in 3 listings or incomplete data.
2 — Significant inaccuracies in most listings.
1 — Data is completely inaccurate or missing.

#### B. Coverage of Listings (0.30)
Measures whether the agent extracted data for exactly 5 listings and applied the required filters.

5 — Data for exactly 5 listings with all filters applied.
4 — Data for 4-5 listings with minor filter issues.
3 — Data for 3 listings or partial filter application.
2 — Data for 1-2 listings or major filter issues.
1 — No data extracted or filters not applied.

#### C. Detail and Specificity (0.20)
Measures whether the extracted data includes sufficient detail, such as key amenities and structured formatting.

5 — Includes all required details (price, address, key amenities) in a structured format.
4 — Includes most details but minor omissions or formatting issues.
3 — Includes some details but lacks structure or key elements.
2 — Minimal details included or poorly formatted.
1 — No details or structure present.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Output is clear, well-organized, and easy to interpret.
4 — Output is mostly clear but with minor formatting issues.
3 — Output is somewhat clear but disorganized.
2 — Output is poorly organized and hard to interpret.
1 — Output is completely unclear or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "coverage_of_listings": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_listings": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "coverage_of_listings": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())