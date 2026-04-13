"""
LLM-as-judge evaluator for EvolveBench task-81.

Category: Shopping
Task: Research the cheapest shops where you can buy the iPhone 15 Plus and iPhone 16 Plus under a 2-year return program.
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


TASK_INSTRUCTION = """Research the cheapest shops where you can buy the iPhone 15 Plus and iPhone 16 Plus under a 2-year return program."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves finding the cheapest prices for iPhone 15 Plus and iPhone 16 Plus in Japan under a 2-year return (残価設定型) program across carrier and retail options.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Products: iPhone 15 Plus AND iPhone 16 Plus — both required
- Program: 2-year return/残価設定型 program (不要になったら返却できるプログラム)
- Market: Japan (based on language context)
- Comparison: multiple shops/carriers compared for each device

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for both iPhone models under 2-year return programs?
- Which shops/carriers were checked (docomo, au, SoftBank, Apple, etc.)?
- What are the monthly or total prices for each model?
- Which shop was identified as cheapest for each model?

### Step 2: Dimension Scoring

#### A. Product Coverage (0.2)
Were both iPhone 15 Plus and 16 Plus covered?

5 — Both models researched with prices under 2-year return program.
4 — Both models covered but one less thoroughly.
3 — Only one model researched.
2 — Prices found but not specifically for 2-year return program.
1 — No product research.

#### B. Shop Comparison (0.3)
Were multiple shops compared?

5 — 4+ shops/carriers compared for each model (docomo, au, SoftBank, Apple, etc.).
4 — 3 shops compared.
3 — 2 shops compared.
2 — Only 1 shop.
1 — No comparison.

#### C. Price Accuracy (0.35)
Are prices accurate and program-specific?

5 — Monthly installment and total program cost clearly stated; 2-year return program confirmed at each carrier.
4 — Prices present but program type not always confirmed.
3 — Approximate prices without program-specific detail.
2 — General iPhone prices without return program context.
1 — No reliable prices.

#### D. Best Deal (0.15)
Was the cheapest option identified?

5 — Clear winner for each model with total cost breakdown.
4 — Best deal identified but without full cost breakdown.
3 — Best deal suggested without clear comparison basis.
2 — Prices listed without identifying best deal.
1 — No recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "product_coverage": <1-5>,
  "shop_comparison": <1-5>,
  "price_accuracy": <1-5>,
  "best_deal": <1-5>,
  "dimension_reasoning": {{
    "product_coverage": "<one sentence citing specific evidence>",
    "shop_comparison": "<one sentence citing specific evidence>",
    "price_accuracy": "<one sentence citing specific evidence>",
    "best_deal": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "product_coverage": 0.2,
    "shop_comparison": 0.3,
    "price_accuracy": 0.35,
    "best_deal": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())