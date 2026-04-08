"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Research and recommend the most cost-effective cordless drill under $150, including warranty, shipping, and accessory costs.
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


TASK_INSTRUCTION = """Research on Amazon.com, Lowes.com, and HomeDepot.com to find cordless drills priced under $150. Include warranty cost, shipping fees, and accessory costs (such as extra batteries). Calculate the total cost of ownership for the best option and recommend the most cost-effective drill."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research cordless drills priced under $150 across Amazon.com, Lowes.com, and HomeDepot.com. The agent must calculate the total cost of ownership, including warranty, shipping, and accessory costs, and recommend the most cost-effective drill. This is a shopping task requiring detailed comparison and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research on Amazon.com, Lowes.com, and HomeDepot.com to find cordless drills priced under $150. Include warranty cost, shipping fees, and accessory costs (such as extra batteries). Calculate the total cost of ownership for the best option and recommend the most cost-effective drill.

## Task-Specific Constraints
- Must visit Amazon.com, Lowes.com, and HomeDepot.com.
- Must include price data for all compared drills.
- Must calculate and include warranty, shipping, and accessory costs.
- Output must be organized as a table or structured list.
- Must recommend the most cost-effective drill based on total cost of ownership.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon.com, Lowes.com, and HomeDepot.com? Which platforms were actually visited?
- Are price, warranty, shipping, and accessory costs included for all compared drills?
- Is the output organized as a table or structured list?
- Is the recommendation for the most cost-effective drill based on accurate calculations?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the most cost-effective drill based on total cost of ownership.

5 — Correctly identifies the best drill with accurate calculations for price, warranty, shipping, and accessories.
4 — Identifies a good drill but with minor calculation errors or omissions.
3 — Identifies a drill but with significant calculation errors or missing elements.
2 — Incorrect recommendation or incomplete calculations.
1 — No recommendation or completely incorrect calculations.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included relevant data.

5 — Visits Amazon.com, Lowes.com, and HomeDepot.com, and includes data from all.
4 — Visits at least two platforms and includes data from them.
3 — Visits one platform and includes partial data.
2 — Visits one platform but includes little or no data.
1 — Does not visit any required platforms.

#### C. Detail Specificity (0.20)
Measures the inclusion of detailed costs (price, warranty, shipping, accessories) for all compared drills.

5 — Includes all required costs for all compared drills.
4 — Includes most required costs with minor omissions.
3 — Includes some required costs but with significant omissions.
2 — Includes minimal cost details.
1 — Includes no cost details.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with all required elements.
4 — Output is mostly structured but with minor formatting issues.
3 — Output is partially structured but lacks clarity.
2 — Output is poorly structured and hard to interpret.
1 — Output is unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
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
    "detail_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())