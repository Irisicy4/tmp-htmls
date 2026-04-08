"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Calculate the total cost of purchasing a gaming laptop from Newegg, including shipping fees, optional 2-year warranty, and sales tax for ZIP code 90001. Compare three laptops and recommend the best option based on features and total price.
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


TASK_INSTRUCTION = """Calculate the total cost of purchasing a gaming laptop from Newegg, including shipping fees, optional 2-year warranty, and sales tax for ZIP code 90001. Compare three laptops and recommend the best option based on features and total price."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the total cost of purchasing a gaming laptop from Newegg, including shipping fees, optional 2-year warranty, and sales tax for ZIP code 90001. The agent must compare three laptops and recommend the best option based on features and total price.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of purchasing a gaming laptop from Newegg, including shipping fees, optional 2-year warranty, and sales tax for ZIP code 90001. Compare three laptops and recommend the best option based on features and total price.

## Task-Specific Constraints
- Must visit Newegg.com and sales-tax.com to gather relevant data.
- Must include price data for all three laptops compared.
- Must calculate and include shipping fees, warranty costs, and sales tax for ZIP code 90001.
- Output must be organized as a structured table or list.
- Recommendation must clearly justify the best option based on features and total price.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Newegg.com and sales-tax.com? Which platforms were actually visited?
- Are price data, shipping fees, warranty costs, and sales tax calculations present for all three laptops?
- Is the output organized as a structured table or list?
- Does the recommendation clearly justify the best option based on features and total price?
- Are all numerical calculations accurate and sourced?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's main output (total cost calculations and recommendation) is correct and complete.

5 — All calculations (price, shipping, warranty, tax) are accurate and complete for all three laptops; recommendation is justified and correct.
4 — Minor errors in calculations or recommendation justification, but overall usable.
3 — Partial calculations present; recommendation lacks justification or is unclear.
2 — Major errors in calculations; recommendation is incorrect or missing.
1 — No calculations or recommendation provided.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and gathered necessary data.

5 — Successfully visited Newegg.com and sales-tax.com; gathered all required data.
4 — Visited both platforms but missed minor data points.
3 — Visited only one platform or gathered incomplete data.
2 — Attempted platform visits but failed to gather usable data.
1 — Did not visit required platforms.

#### C. Depth of Comparison (0.20)
Measures whether the agent provided detailed comparisons of the laptops based on features and total price.

5 — Detailed comparison of all three laptops, including features and total price breakdowns.
4 — Comparison is present but lacks minor details or clarity.
3 — Comparison is partially complete; lacks depth or specificity.
2 — Minimal comparison; missing key details.
1 — No comparison provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list; sources are credible and cited.
4 — Output is mostly well-organized; minor structural issues or unclear sourcing.
3 — Output is partially organized; lacks clarity or credible sourcing.
2 — Output is disorganized or sources are questionable.
1 — Output is unstructured or sources are absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_comparison": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())