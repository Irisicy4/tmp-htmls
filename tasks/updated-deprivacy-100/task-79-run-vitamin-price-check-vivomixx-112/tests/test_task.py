"""
LLM-as-judge evaluator for EvolveBench task-79.

Category: Shopping
Task: Run a vitamin price check on Vivomixx 112 food supplement across multiple vendor websites and compile a price comparison
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


TASK_INSTRUCTION = """Run a vitamin price check on Vivomixx 112 food supplement across multiple vendor websites and compile a price comparison report."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching prices for Vivomixx 112 (a specific probiotic supplement) across multiple online retailers and producing a structured comparison report.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Product: Vivomixx 112 capsules specifically (not other Vivomixx variants)
- Coverage: at least 4 vendor websites compared
- Report: structured with vendor name, price, shipping, and URL
- Best deal: must identify the lowest price

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Vivomixx 112 specifically?
- How many vendors were checked?
- What prices were found (with currency)?
- Was a comparison table or report produced?
- Was the best deal identified?

### Step 2: Dimension Scoring

#### A. Product Targeting (0.2)
Did the agent search for Vivomixx 112 specifically?

5 — Searched for Vivomixx 112 (capsule count) and found it on multiple sites.
4 — Found Vivomixx 112 on some sites; others show different variants.
3 — Found Vivomixx generally but not always the 112-count variant.
2 — Found similar products but not Vivomixx 112 specifically.
1 — Wrong product or no product found.

#### B. Vendor Coverage (0.25)
How many vendors were compared?

5 — 5 or more vendors with prices.
4 — 4 vendors.
3 — 3 vendors.
2 — 2 vendors.
1 — Only 1 vendor or none.

#### C. Price Accuracy (0.35)
Are prices accurate with clear sourcing?

5 — Prices from vendor pages with URLs, current date, correct product variant.
4 — Prices accurate but missing some URLs.
3 — Prices found but sourcing unclear.
2 — Prices estimated or from aggregators without direct verification.
1 — No reliable prices.

#### D. Report Structure (0.2)
Is the comparison report well-structured?

5 — Table format: vendor, price, shipping, total, URL; best deal highlighted.
4 — Structured list with most fields.
3 — Prices listed without full structure.
2 — Narrative without clear comparison structure.
1 — No report structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "product_targeting": <1-5>,
  "vendor_coverage": <1-5>,
  "price_accuracy": <1-5>,
  "report_structure": <1-5>,
  "dimension_reasoning": {{
    "product_targeting": "<one sentence citing specific evidence>",
    "vendor_coverage": "<one sentence citing specific evidence>",
    "price_accuracy": "<one sentence citing specific evidence>",
    "report_structure": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "product_targeting": 0.2,
    "vendor_coverage": 0.25,
    "price_accuracy": 0.35,
    "report_structure": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())