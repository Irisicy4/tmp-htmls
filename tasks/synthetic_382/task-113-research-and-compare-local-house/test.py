"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Research and compare local house cleaning services in Boston, MA on price, reviews, and weekend availability.
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


TASK_INSTRUCTION = """Research and compare local house cleaning services in Boston, MA. Use websites like Yelp, Angi, and HomeAdvisor to find at least three services. Compare them on price, customer reviews, and availability for weekend appointments."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare local house cleaning services in Boston, MA using Yelp, Angi, and HomeAdvisor. A successful completion requires identifying at least three services and comparing them based on price, customer reviews, and availability for weekend appointments. The output must be structured and include specific details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare local house cleaning services in Boston, MA. Use websites like Yelp, Angi, and HomeAdvisor to find at least three services. Compare them on price, customer reviews, and availability for weekend appointments.

## Task-Specific Constraints
- Must visit at least three of the specified platforms (Yelp, Angi, HomeAdvisor).
- Must identify at least three house cleaning services.
- Must include price data for all services compared.
- Must include customer reviews for all services compared.
- Must specify availability for weekend appointments for all services.
- Output must be organized as a table or structured list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify at least three house cleaning services?
- Are price data, customer reviews, and weekend availability included for all services?
- Is the output organized as a table or structured list?
- Are the claims made in the response accurate and sourced?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and compared three services based on price, reviews, and weekend availability.

5 — Identifies and compares at least three services with complete data for price, reviews, and weekend availability.
4 — Identifies and compares three services but one data type (e.g., price or reviews) is incomplete.
3 — Identifies and compares three services but two data types are incomplete.
2 — Identifies fewer than three services or most data is missing.
1 — No valid services or data identified.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms (Yelp, Angi, HomeAdvisor).

5 — Uses all three platforms and extracts data from each.
4 — Uses two platforms and extracts data from both.
3 — Uses one platform and extracts data.
2 — Navigates platforms but fails to extract meaningful data.
1 — Does not navigate any platform.

#### C. Depth of Comparison (0.25)
Measures the specificity and detail of the comparisons provided.

5 — Provides detailed comparisons with specific numbers, quotes, or examples for all services.
4 — Provides comparisons with some specific details but lacks depth in one area.
3 — Provides general comparisons but lacks specific details in multiple areas.
2 — Provides vague or incomplete comparisons.
1 — No meaningful comparisons provided.

#### D. Output Organization (0.10)
Measures whether the output is structured and easy to understand.

5 — Output is organized as a clear table or structured list with labeled columns/sections.
4 — Output is mostly organized but lacks some clarity or labels.
3 — Output is minimally organized but usable.
2 — Output is disorganized and difficult to follow.
1 — Output is completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_comparison": 0.25,
    "output_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())