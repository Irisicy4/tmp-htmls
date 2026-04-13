"""
LLM-as-judge evaluator for EvolveBench task-26.

Category: Shopping
Task: Find large-size hip support and thigh compression sleeves on Amazon
      for a 280 lb, 50-year-old user.
"""

import os, json, re

TASK_INSTRUCTION = "Find me large-size hip support and thigh compression sleeves on Amazon. I weigh 280 lbs and I am 50 years old."
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks assessing an Amazon product search task with specific user constraints."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Amazon specifically
- Product type: hip support AND/OR thigh compression sleeves (both are acceptable; either alone is partial)
- Size constraint: must fit a 280 lb user — agent must filter for plus-size/bariatric/XXL options
- Age context: 50 years old — implies recovery/joint support context, not athletic performance
- Agent must search Amazon and retrieve actual current listings with prices/ratings

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent search Amazon? Cite evidence.
- Were products found for hip support AND thigh compression, or only one?
- Are the products suitable for a 280 lb user (plus-size, XXL, or explicitly rated for heavy users)?
- Are specific product names/ASINs/prices provided?
- Did the agent consider the user's age/context in recommendations?

### Step 2: Dimension Scoring

#### A. Amazon Search Execution
Did the agent actively search Amazon for current listings?

5 — Agent searched Amazon and retrieved current product listings with names, prices, and ratings.
4 — Agent searched Amazon but retrieved limited results or only product names without prices/ratings.
3 — Agent used general web search referencing Amazon rather than navigating Amazon directly.
2 — Agent described what to search for without performing the search.
1 — No search; response from prior knowledge or refused.

#### B. Size Appropriateness
Are the recommended products suitable for a 280 lb user?

5 — All products explicitly accommodate 280 lb users; size charts or weight limits mentioned; XXL/2XL/bariatric options specified.
4 — Products appear to be plus-size or large range; weight limit not explicitly verified but plausibly suitable.
3 — Agent mentioned "large size" without verifying 280 lb suitability; standard large may not fit.
2 — Products are standard sizes without considering the user's weight.
1 — Size constraint ignored entirely.

#### C. Product Coverage
Did the agent find both hip support AND thigh compression products?

5 — Both product types found with 2+ options each; clear distinction between hip support and thigh compression.
4 — Both types found but only 1 option for one type, or types are conflated.
3 — Only one product type found; the other is described but not searched.
2 — Only one product type found with no mention of the other.
1 — Neither product type found as a specific Amazon listing.

#### D. Recommendation Quality
Are the recommendations specific, actionable, and suitable for the user's context?

5 — Specific products with name, ASIN or link, price, rating, and brief explanation of why suitable for this user (280 lb, joint support context).
4 — Specific products with most details; user context (age/weight) briefly considered.
3 — Product names provided but lacking price/rating or user-context consideration.
2 — Very generic recommendations without specific product details.
1 — No actionable recommendations.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "amazon_search_execution": <1-5>,
  "size_appropriateness": <1-5>,
  "product_coverage": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{"amazon_search_execution": "<one sentence>", "size_appropriateness": "<one sentence>", "product_coverage": "<one sentence>", "recommendation_quality": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"amazon_search_execution": 0.25, "size_appropriateness": 0.35, "product_coverage": 0.20, "recommendation_quality": 0.20}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    t = result.get("task_result") or ""
    if isinstance(t, str) and t.strip(): return t
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and len(c) > 20: return c
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
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task_instruction=TASK_INSTRUCTION, agent_response=agent_response, execution_summary=execution_summary or "Not available.")}],
            max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid)//2] for d in DIMENSIONS}
    overall = sum(agg[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    agg["overall_score"] = round(overall, 2); agg["passed"] = overall >= PASS_THRESHOLD
    median = sorted(valid, key=lambda v: abs(v.get("overall_score",0)-overall))[0]
    agg["evidence_summary"] = median.get("evidence_summary",""); agg["dimension_reasoning"] = median.get("dimension_reasoning",{}); agg["_votes_used"] = len(valid)
    return agg

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.", "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)]) if abs(float(overall) - PASS_THRESHOLD) <= 0.5 else (first.__setitem__("_votes_used", 1) or first)
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0); passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"] + [f"  {d}: {scores[d]}/5" for d in DIMENSIONS if d in scores]
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    if scores.get("dimension_reasoning"):
        lines.append("\nDimension reasoning:")
        for d, r in scores["dimension_reasoning"].items(): lines.append(f"  {d}: {r}")
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline: {scores['_votes_used']} calls, majority vote)")
    return {"passed": bool(passed), "feedback": "\n".join(lines), "details": {"task_completed": result.get("status") == "success", "overall_score": overall, "dimension_scores": {d: scores.get(d) for d in DIMENSIONS}, "evidence_summary": scores.get("evidence_summary"), "dimension_reasoning": scores.get("dimension_reasoning"), "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)}}