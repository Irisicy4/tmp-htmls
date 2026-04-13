"""
LLM-as-judge evaluator for EvolveBench task-86.

Category: Finance & Economics
Task: Research and analyze the upside potential for TSMC and Samsung Electronics stock prices over the next 30 days based on publicly available analyst reports, recent earnings data, and news. Infer a peak price for each.
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
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

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


TASK_INSTRUCTION = """Research and analyze the upside potential for TSMC and Samsung Electronics stock prices over the next 30 days based on publicly available analyst reports, recent earnings data, and news. Infer a peak price for each."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching publicly available financial data — analyst reports, earnings releases, and news — for two global semiconductor stocks, TSMC and Samsung Electronics, to forecast their price upside over a 30-day forward-looking window.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Stocks: TSMC and Samsung Electronics — both required
- Horizon: over the next 30 days (forward-looking)
- Analysis: must be based on publicly available analyst reports, earnings data, and news
- Output: peak price inference with reasoning for each stock

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research publicly available materials (analyst reports, earnings, news)?
- Were both stocks analyzed individually?
- What price targets or peak prices were inferred?
- Is the analysis grounded in fundamentals, technicals, or both?
- Is the 30-day forward-looking window addressed?

### Step 2: Dimension Scoring

#### A. Material Analysis (0.25)
Did the agent analyze publicly available financial materials?

5 — Agent researched and referenced specific data from analyst reports, earnings releases, or financial news for the global semiconductor market.
4 — Materials referenced but analysis is shallow.
3 — General financial analysis without clear reference to sources.
2 — Agent described the task without analyzing materials.
1 — No material analysis.

#### B. Stock Coverage (0.2)
Were both stocks analyzed?

5 — Both TSMC and Samsung Electronics analyzed individually with separate assessments.
4 — Both covered but one more superficially.
3 — Only one stock analyzed in depth.
2 — Both mentioned but neither analyzed.
1 — Only one stock or neither.

#### C. Price Inference (0.35)
Were peak price levels inferred with reasoning?

5 — Specific peak price range stated for each stock over the next 30 days, with technical (resistance levels, momentum) or fundamental (valuation, catalyst) rationale.
4 — Price range inferred with partial rationale.
3 — Upside direction identified without specific price target.
2 — Vague 'could go up' without quantification.
1 — No price inference.

#### D. Analytical Rigor (0.2)
Is the analysis methodologically sound?

5 — Clear methodology: technical analysis, fundamental valuation, catalyst identification, risk factors.
4 — Good methodology but one element weak.
3 — Basic analysis with limited rigor.
2 — Opinion-based without analytical framework.
1 — No analytical rigor.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "material_analysis": <1-5>,
  "stock_coverage": <1-5>,
  "price_inference": <1-5>,
  "analytical_rigor": <1-5>,
  "dimension_reasoning": {{
    "material_analysis": "<one sentence citing specific evidence>",
    "stock_coverage": "<one sentence citing specific evidence>",
    "price_inference": "<one sentence citing specific evidence>",
    "analytical_rigor": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "material_analysis": 0.25,
    "stock_coverage": 0.2,
    "price_inference": 0.35,
    "analytical_rigor": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())