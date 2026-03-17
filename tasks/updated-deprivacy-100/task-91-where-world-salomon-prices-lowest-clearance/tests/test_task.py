"""
LLM-as-judge evaluator for EvolveBench task-91.

Category: Shopping
Task: Where in the world are Salomon prices the lowest and where is it most likely to find clearance stock?
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


TASK_INSTRUCTION = """Where in the world are Salomon prices the lowest and where is it most likely to find clearance stock?"""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching global pricing disparities for Salomon outdoor gear and identifying regions with the lowest prices and highest likelihood of clearance/sale stock.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Brand: Salomon specifically
- Coverage: global comparison — multiple countries/regions
- Clearance: must identify where clearance stock is most likely (seasonal, outlet stores, end-of-line)
- Output: ranked list of regions with price reasoning

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research Salomon prices across multiple countries?
- What specific regions/countries were identified as cheapest?
- What explains the price differences (currency, VAT, market positioning)?
- Where is clearance stock most likely to be found and why?

### Step 2: Dimension Scoring

#### A. Geographic Coverage (0.25)
Were multiple global regions compared?

5 — 5+ regions compared (e.g. Japan, US, EU, China, Outlet regions) with specific price data.
4 — 3-4 regions compared.
3 — 2 regions compared.
2 — Only one region analyzed.
1 — No geographic comparison.

#### B. Price Analysis (0.3)
Are price differences explained with evidence?

5 — Specific prices in local currency with reasoning (VAT, import duty, regional positioning, currency strength).
4 — Prices found with partial explanation.
3 — Regions identified as cheaper but without clear price data.
2 — Vague 'Japan is cheaper' without evidence.
1 — No price analysis.

#### C. Clearance Insights (0.3)
Were clearance opportunities identified?

5 — Specific clearance channels identified: outlet stores, end-of-season markets, specific retailers known for Salomon clearance.
4 — Good clearance insights but less specific.
3 — General 'check outlets' advice without specifics.
2 — Clearance mentioned without insight.
1 — No clearance analysis.

#### D. Actionability (0.15)
Is the output actionable for a buyer?

5 — Clear ranked list with specific stores, websites, or regions to target for best prices.
4 — Mostly actionable.
3 — Informative but requires further research to act.
2 — Too vague to act on.
1 — Not actionable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "geographic_coverage": <1-5>,
  "price_analysis": <1-5>,
  "clearance_insights": <1-5>,
  "actionability": <1-5>,
  "dimension_reasoning": {{
    "geographic_coverage": "<one sentence citing specific evidence>",
    "price_analysis": "<one sentence citing specific evidence>",
    "clearance_insights": "<one sentence citing specific evidence>",
    "actionability": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "geographic_coverage": 0.25,
    "price_analysis": 0.3,
    "clearance_insights": 0.3,
    "actionability": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())