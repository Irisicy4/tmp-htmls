"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Calculate the total cost of purchasing an Apple MacBook Air, including the base price from Apple, a carrying case from Amazon, and a Magic Mouse from Walmart, factoring in shipping costs and taxes for California.
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


TASK_INSTRUCTION = """Calculate the total cost of purchasing an Apple MacBook Air, including the base price from Apple, a carrying case from Amazon, and a Magic Mouse from Walmart. Include shipping costs and applicable taxes from each site, assuming a shipping address in California."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to calculate the total cost of purchasing an Apple MacBook Air, a carrying case, and a Magic Mouse from three specific platforms (Apple, Amazon, Walmart). The agent must include shipping costs and applicable taxes for California. A successful completion requires accurate price data, correct tax/shipping calculations, and a structured output summarizing the total cost.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the total cost of purchasing an Apple MacBook Air, including the base price from Apple, a carrying case from Amazon, and a Magic Mouse from Walmart. Include shipping costs and applicable taxes from each site, assuming a shipping address in California.

## Task-Specific Constraints
- Must visit apple.com, amazon.com, and walmart.com to retrieve price data.
- Must include shipping costs and applicable taxes for California in the calculations.
- Output must be organized as a structured table or list summarizing costs per item and the total.
- Must clearly identify the base price, tax, and shipping costs for each item.
- Must correctly calculate the final total cost.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to apple.com, amazon.com, and walmart.com? Which platforms were visited?
- Are the base prices, tax amounts, and shipping costs for all three items present in the response?
- Is the output organized as a structured table or list summarizing costs per item and the total?
- Are the tax and shipping calculations for California accurate and consistent with the provided evidence?
- Does the agent provide a clear and complete total cost calculation?

### Step 2: Dimension Scoring

#### A. Total Cost Accuracy (0.35)
Measures whether the agent correctly calculates the total cost, including base prices, taxes, and shipping.

5 — All calculations are correct and complete, with no errors.
4 — Minor errors in tax or shipping calculations, but total cost is mostly accurate.
3 — Partial calculations provided; some key components (e.g., tax or shipping) are missing.
2 — Significant errors in calculations; total cost is largely incorrect.
1 — No meaningful attempt at calculating the total cost.

#### B. Platform Coverage (0.30)
Measures whether the agent retrieves data from all required platforms (Apple, Amazon, Walmart).

5 — Data from all three platforms is retrieved and used correctly.
4 — Data from two platforms is retrieved; one is missing or incomplete.
3 — Data from one platform is retrieved; others are missing.
2 — No meaningful data retrieved from the required platforms.
1 — No attempt to retrieve data from any platform.

#### C. Detail and Specificity (0.20)
Measures whether the agent provides detailed breakdowns of costs (base price, tax, shipping) for each item.

5 — Detailed breakdowns for all items, including base price, tax, and shipping.
4 — Minor omissions or inaccuracies in breakdowns for one item.
3 — Partial breakdowns provided; some details missing for multiple items.
2 — Very limited detail; most breakdowns are missing or incorrect.
1 — No meaningful breakdowns provided.

#### D. Output Structure and Clarity (0.15)
Measures whether the agent organizes the output in a clear, structured format (e.g., table or list).

5 — Output is well-organized, easy to read, and fully structured.
4 — Output is mostly clear, with minor formatting issues.
3 — Output is partially organized but lacks clarity or structure.
2 — Output is poorly organized and difficult to interpret.
1 — No meaningful structure or clarity in the output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "total_cost_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "total_cost_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "total_cost_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())