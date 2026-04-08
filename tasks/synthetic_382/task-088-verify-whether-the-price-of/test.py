"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Verify whether the price of Apple's AirPods Pro is consistent across Amazon, Target, and Best Buy for the current date.
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


TASK_INSTRUCTION = """Verify whether the price of Apple's AirPods Pro is consistent across Amazon, Target, and Best Buy for the current date. Include details such as listed price, shipping fees, and availability. Flag any discrepancies between the platforms."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify price consistency for Apple's AirPods Pro across three major e-commerce platforms: Amazon, Target, and Best Buy. The agent must include details such as listed price, shipping fees, and availability, and flag any discrepancies between the platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the price of Apple's AirPods Pro is consistent across Amazon, Target, and Best Buy for the current date. Include details such as listed price, shipping fees, and availability. Flag any discrepancies between the platforms.

## Task-Specific Constraints
- Must visit Amazon, Target, and Best Buy platforms.
- Must include price data, shipping fees, and availability for AirPods Pro from each platform.
- Output must be organized as a structured table or list format.
- Must explicitly flag any discrepancies between platforms.
- Must include the current date in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Target, and Best Buy platforms? Which ones were actually visited?
- Does the response include price, shipping fees, and availability for AirPods Pro from all platforms?
- Is the output organized as a structured table or list format?
- Are discrepancies between platforms explicitly flagged?
- Is the current date included in the response?

### Step 2: Dimension Scoring

#### A. Price Consistency Analysis (0.35)
Measures whether the agent correctly identified and compared prices across platforms.

5 — Identifies prices for AirPods Pro from all three platforms and flags discrepancies accurately.
4 — Identifies prices for AirPods Pro from all three platforms but misses minor discrepancies.
3 — Identifies prices for AirPods Pro from at least two platforms but lacks accuracy or detail.
2 — Identifies prices for AirPods Pro from only one platform or contains major errors.
1 — Fails to identify prices for AirPods Pro from any platform.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included relevant data.

5 — Visits Amazon, Target, and Best Buy, and includes data from all three.
4 — Visits all three platforms but misses minor details from one.
3 — Visits at least two platforms and includes partial data.
2 — Visits only one platform or includes minimal data.
1 — Fails to visit any platform or includes no data.

#### C. Detail and Specificity (0.25)
Measures the depth of information provided, including shipping fees, availability, and structured output.

5 — Includes shipping fees, availability, and organizes output in a structured format.
4 — Includes most details but misses minor elements or lacks organization.
3 — Includes partial details but lacks depth or structure.
2 — Includes minimal details and lacks structure.
1 — Fails to include any relevant details.

#### D. Output Organization and Credibility (0.10)
Measures the credibility of sources and the organization of the output.

5 — Sources are credible, and output is well-organized and easy to interpret.
4 — Sources are credible, but output has minor organizational issues.
3 — Sources are partially credible, and output is somewhat disorganized.
2 — Sources are questionable, and output is poorly organized.
1 — Sources are not credible, and output is completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "price_consistency_analysis": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_organization_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "price_consistency_analysis": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_organization_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "price_consistency_analysis": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.25,
    "output_organization_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())