"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Evaluate whether the agent successfully searched for wireless noise-canceling headphones on three platforms, applied price filters, and extracted top 5 options including name, price, and seller rating.
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


TASK_INSTRUCTION = """Visit Target.com, Amazon.com, and eBay.com, and search for wireless noise-canceling headphones. Apply filters to only show options priced between $150–$200, and extract the top 5 options from each site including name, price, and seller rating."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves searching for wireless noise-canceling headphones on three e-commerce platforms (Target, Amazon, eBay), applying price filters between $150–$200, and extracting the top 5 options from each site with name, price, and seller rating. A successful completion requires accurate data extraction and adherence to the task constraints.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit Target.com, Amazon.com, and eBay.com, and search for wireless noise-canceling headphones. Apply filters to only show options priced between $150–$200, and extract the top 5 options from each site including name, price, and seller rating.

## Task-Specific Constraints
- Must visit all three specified platforms: Target.com, Amazon.com, and eBay.com.
- Must apply price filters between $150–$200 on each platform.
- Must extract exactly 5 options from each platform.
- Must include name, price, and seller rating for each extracted option.
- Output must be organized as a structured table or list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms? Which ones were actually visited?
- Did the agent apply the correct price filters on each platform?
- Are the top 5 options extracted from each platform, and do they include name, price, and seller rating?
- Is the output organized as a structured table or list?
- Are the extracted details accurate and consistent with the task requirements?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted data is correct, complete, and matches the task requirements.

5 — Extracts correct name, price, and seller rating for 5 options from each platform.
4 — Extracts correct data for 4–5 options from each platform, with minor omissions.
3 — Extracts correct data for at least 3 options per platform, but incomplete or partially incorrect.
2 — Extracts data for fewer than 3 options per platform, with significant errors.
1 — Fails to extract meaningful data or includes major inaccuracies.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and applied the correct filters.

5 — Visits all three platforms and applies price filters correctly on each.
4 — Visits all three platforms but applies filters incorrectly on one.
3 — Visits at least two platforms and applies filters correctly on one.
2 — Visits only one platform or applies filters incorrectly on multiple platforms.
1 — Fails to visit any required platform or apply filters.

#### C. Data Specificity (0.20)
Measures the depth and specificity of the extracted data.

5 — Includes detailed and accurate seller ratings, prices, and names for all extracted options.
4 — Includes detailed data for most options, with minor omissions.
3 — Includes basic data for most options, but lacks depth or accuracy.
2 — Includes incomplete or vague data for most options.
1 — Fails to provide meaningful data.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Organizes data as a clear, structured table or list with all required fields.
4 — Organizes data mostly well, with minor formatting issues.
3 — Provides data in a readable format but lacks structure or clarity.
2 — Provides data in a disorganized or confusing format.
1 — Fails to provide organized output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "data_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "data_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())