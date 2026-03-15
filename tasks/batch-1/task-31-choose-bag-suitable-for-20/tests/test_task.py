"""
LLM-as-judge evaluator for EvolveBench task-31.

Category: Shopping
Amazon backpack search across 4 specific brands, under ¥15,000 for a 20yo.
"""

import os, json, re

TASK_INSTRUCTION = 'Choose a bag suitable for a 20-year-old. From these four brands (OUTDOOR, GREGORY, COLEMAN, MARIMEKKO) find backpacks under 15,000 yen. Search on Amazon, pay attention to price, and after searching report the product names.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks assessing a constrained Amazon product search task.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Platform: Amazon (amazon.co.jp implied given yen pricing)\n- Brands: must search all four — OUTDOOR, GREGORY, COLEMAN, MARIMEKKO\n- Price: under ¥15,000 hard limit\n- Output: product names (not just brand names); specific model names required\n- Age suitability: 20-year-old context (style/functionality appropriate for young adult)\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent search Amazon for backpacks from these 4 brands?\n- Were products found under ¥15,000 from each brand?\n- Are specific product names (model names) listed?\n- Did the agent apply the price filter correctly?\n\n### Step 2: Dimension Scoring\n\n#### AMAZON SEARCH (weight 0.25)\nDid the agent search Amazon and retrieve current listings?\n\n5 — Agent navigated Amazon, searched each of the 4 brands, and retrieved current product listings with prices.\n4 — Agent searched Amazon for most brands; one brand missing or prices not verified.\n3 — Agent used general web search referencing Amazon rather than navigating directly.\n2 — Agent described what to search without performing the search.\n1 — No search performed.\n\n#### BRAND COVERAGE (weight 0.25)\nWere backpacks found from all 4 brands?\n\n5 — At least one backpack under ¥15,000 found and named for each of the 4 brands.\n4 — 3 of 4 brands have qualifying products listed; one brand had no qualifying product (noted).\n3 — 2 of 4 brands covered with products.\n2 — Only 1 brand covered.\n1 — No brand-specific products found.\n\n#### PRICE COMPLIANCE (weight 0.3)\nAre all listed products genuinely under ¥15,000?\n\n5 — All products explicitly priced and confirmed under ¥15,000; prices cited from Amazon.\n4 — All products stated to be under ¥15,000 but prices not explicitly cited for all.\n3 — Some products priced; a few are above ¥15,000 or price unclear.\n2 — Price filter applied loosely; several products may exceed ¥15,000.\n1 — Price constraint ignored entirely.\n\n#### PRODUCT SPECIFICITY (weight 0.2)\nAre specific product model names reported (not just brand names)?\n\n5 — Full model names listed for each product (e.g. "Gregory Nano 16 Backpack").\n4 — Model names listed for most; 1–2 are brand+type only (e.g. "COLEMAN hiking bag").\n3 — Mix of model names and generic descriptions.\n2 — Only brand names listed without model names.\n1 — No product names at all.\n\n\n### Step 3: Output\nRespond ONLY with valid JSON inside <Answer></Answer> tags:\n\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "amazon_search": <1-5>,\n  "brand_coverage": <1-5>,\n  "price_compliance": <1-5>,\n  "product_specificity": <1-5>,\n  "dimension_reasoning": {"amazon_search": "<one sentence>", "brand_coverage": "<one sentence>", "price_compliance": "<one sentence>", "product_specificity": "<one sentence>"},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>\n'

DIMENSION_WEIGHTS = {'amazon_search': 0.25, 'brand_coverage': 0.25, 'price_compliance': 0.3, 'product_specificity': 0.2}
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
    import re
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except: return None

def _call(agent_response, execution_summary, system_prompt, user_prompt_template, task_instruction):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt_template.format(
                          task_instruction=task_instruction,
                          agent_response=agent_response,
                          execution_summary=execution_summary or "Not available.")}],
            max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes, dimensions, weights, pass_threshold):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dimensions)]
    if not valid: return votes[0] if votes else {"error": "All calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid)//2] for d in dimensions}
    overall = sum(agg[d] * weights[d] for d in dimensions)
    agg["overall_score"] = round(overall, 2); agg["passed"] = overall >= pass_threshold
    median = sorted(valid, key=lambda v: abs(v.get("overall_score",0)-overall))[0]
    agg["evidence_summary"] = median.get("evidence_summary","")
    agg["dimension_reasoning"] = median.get("dimension_reasoning",{})
    agg["_votes_used"] = len(valid)
    return agg

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first,
                _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION),
                _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)],
                DIMENSIONS, DIMENSION_WEIGHTS, PASS_THRESHOLD)
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    lines += [f"  {d}: {scores[d]}/5" for d in DIMENSIONS if d in scores]
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    if scores.get("dimension_reasoning"):
        lines.append("\nDimension reasoning:")
        for d, r in scores["dimension_reasoning"].items(): lines.append(f"  {d}: {r}")
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline: {scores['_votes_used']} calls, majority vote)")
    return {"passed": bool(passed), "feedback": "\n".join(lines),
            "details": {"task_completed": result.get("status") == "success",
                        "overall_score": overall,
                        "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                        "evidence_summary": scores.get("evidence_summary"),
                        "dimension_reasoning": scores.get("dimension_reasoning"),
                        "pass_threshold": PASS_THRESHOLD,
                        "votes_used": scores.get("_votes_used", 1)}}