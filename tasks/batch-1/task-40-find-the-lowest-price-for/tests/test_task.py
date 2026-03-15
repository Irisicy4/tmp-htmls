"""
LLM-as-judge evaluator for EvolveBench task-40.
Category: Shopping
"""
import os, json, re

TASK_INSTRUCTION = 'Find the lowest price for Numi organic tea flower tea gift set across the web.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks assessing a cross-web lowest-price search task for a specific product.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Product: Numi Organic Tea Flowering Tea Gift Set specifically (not generic Numi tea)\n- Scope: across the web — must check multiple retailers (Amazon, Numi website, specialty tea retailers, etc.)\n- Goal: lowest price — must compare and identify the cheapest option\n- Shipping: ideally should consider total cost including shipping\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent search for the specific Numi flowering tea gift set?\n- How many retailers were checked?\n- What is the lowest price found? From which retailer?\n- Were prices verified from actual current listings?\n- Was shipping cost considered?\n\n### Step 2: Dimension Scoring\n\n#### A. Search Breadth\nDid the agent check multiple retailers?\n\n5 — 3+ retailers checked (e.g. Amazon, Numi website, iHerb, Walmart, specialty tea shops); prices compared.\n4 — 2 retailers checked with actual prices.\n3 — 1 retailer checked; others referenced but not verified.\n2 — General web search only; no specific retailer navigation.\n1 — No search; response from prior knowledge.\n\n#### B. Price Verification\nAre the prices current and from actual listings?\n\n5 — Prices cited with retailer source; prices are plausibly current (not clearly outdated); specific product variant identified.\n4 — Prices from actual listings; one may be slightly outdated or product variant not fully specified.\n3 — Prices given but source unclear or agent did not verify they are current.\n2 — Prices estimated or from prior knowledge without verification.\n1 — No prices; or fabricated prices.\n\n#### C. Lowest Price Identified\nDid the agent clearly identify the lowest price?\n\n5 — Lowest price explicitly stated with retailer name; comparison to other options shown.\n4 — Lowest price stated; comparison to alternatives not fully shown.\n3 — Multiple prices listed but lowest not explicitly called out.\n2 — Prices listed in no particular order without identifying lowest.\n1 — No price comparison; just one price from one source.\n\n#### D. Result Actionability\nCan the user immediately act on the recommendation?\n\n5 — Retailer name, price, and direct URL or product name for easy search; shipping cost noted.\n4 — Retailer and price provided; URL missing but product findable.\n3 — Retailer named but price or product specifics unclear.\n2 — Vague guidance ("check Amazon for best price") without specific finding.\n1 — No actionable information.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "search_breadth": <1-5>,\n  "price_verification": <1-5>,\n  "lowest_price_identified": <1-5>,\n  "result_actionability": <1-5>,\n  "dimension_reasoning": {{"search_breadth": "<one sentence>", "price_verification": "<one sentence>", "lowest_price_identified": "<one sentence>", "result_actionability": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'search_breadth': 0.25, 'price_verification': 0.35, 'lowest_price_identified': 0.25, 'result_actionability': 0.15}
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