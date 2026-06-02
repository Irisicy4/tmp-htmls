"""
LLM-as-judge evaluator for EvolveBench task-32.
Category: Shopping
"""
import os, json, re

TASK_INSTRUCTION = 'Help me choose the perfect birthday gift for my 28-year-old male friend. He likes gaming, computers and gadgets. The budget should be 10,000-15,000 yen, and it should be something you can buy on Amazon. Pick 3 different options and add them to your shopping cart.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks assessing an Amazon shopping task that requires cart addition, not just product search.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Platform: Amazon specifically\n- Budget: ¥10,000–¥15,000 per item (both bounds are hard constraints)\n- Interest profile: gaming, computers, gadgets\n- Quantity: exactly 3 different options\n- Action: add all 3 to shopping cart (not just list them)\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent search Amazon for gaming/tech gifts?\n- Are 3 distinct products found within ¥10k–¥15k?\n- Did the agent add items to cart? Evidence from trace (add-to-cart action)?\n- Do the products fit the gaming/computers/gadgets interest profile?\n\n### Step 2: Dimension Scoring\n\n#### A. Amazon Search Execution\nDid the agent search Amazon and retrieve current listings?\n\n5 — Agent navigated Amazon and retrieved specific products with current prices.\n4 — Agent searched Amazon; prices found but not all verified as current.\n3 — General web search referencing Amazon without direct navigation.\n2 — Products described without searching.\n1 — No search.\n\n#### B. Budget & Interest Fit\nAre the 3 products within budget and appropriate for the interest profile?\n\n5 — All 3 products within ¥10k–¥15k AND clearly related to gaming/computers/gadgets.\n4 — 2 of 3 within budget and on-interest; one is borderline (just over/under or tangentially related).\n3 — Products are gaming/tech themed but 1–2 are outside the budget range.\n2 — Products within budget but not relevant to stated interests.\n1 — Constraints ignored.\n\n#### C. Cart Addition\nDid the agent add all 3 items to the shopping cart?\n\n5 — All 3 items added to cart; trace confirms add-to-cart actions for each.\n4 — 2 items added to cart; one failed or was skipped.\n3 — Cart addition attempted for at least 1 item; outcome ambiguous.\n2 — Agent identified products but did not attempt to add to cart.\n1 — No cart action of any kind.\n\n#### D. Option Diversity\nAre the 3 options meaningfully different from each other?\n\n5 — 3 distinct product categories (e.g. headset, controller, keyboard — not 3 versions of the same item).\n4 — 2 distinct categories; 2 products are similar but from different brands.\n3 — Products are different items but all from the same sub-category.\n2 — Products are nearly identical (e.g. 3 similar controllers).\n1 — Only 1 option provided or all are the same product.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "amazon_search_execution": <1-5>,\n  "budget_interest_fit": <1-5>,\n  "cart_addition": <1-5>,\n  "option_diversity": <1-5>,\n  "dimension_reasoning": {{"amazon_search_execution": "<one sentence>", "budget_interest_fit": "<one sentence>", "cart_addition": "<one sentence>", "option_diversity": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'amazon_search_execution': 0.2, 'budget_interest_fit': 0.3, 'cart_addition': 0.35, 'option_diversity': 0.15}
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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except: return None

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