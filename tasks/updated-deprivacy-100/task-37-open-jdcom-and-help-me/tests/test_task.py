"""
LLM-as-judge evaluator for EvolveBench task-37.
Category: Shopping
"""
import os, json, re

TASK_INSTRUCTION = 'Open JD.com and help me find the latest phones from Xiaomi, Huawei, vivo, and OPPO in the 3000–4000 yuan price range, and compare their performance.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks assessing a JD.com multi-brand phone search and comparison task.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Platform: JD.com (京东) specifically\n- Brands: Xiaomi, Huawei, vivo, OPPO — all four required\n- Price range: ¥3,000–¥4,000 (both bounds hard constraints)\n- Recency: "latest phones" — must be current/recent models, not discontinued\n- Comparison: performance comparison required (CPU, camera, battery, display)\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent navigate JD.com? Cite evidence.\n- Were phones from all 4 brands found within ¥3k–¥4k?\n- Are the models current/latest (not discontinued)?\n- Is a performance comparison provided?\n\n### Step 2: Dimension Scoring\n\n#### A. JD.com Execution\nDid the agent navigate JD.com as instructed?\n\n5 — Agent navigated JD.com, searched by brand within price range, retrieved current listings.\n4 — Agent accessed JD.com but search was partial (1–2 brands searched directly).\n3 — Agent referenced JD.com from general search without navigating the site.\n2 — Agent used a different platform (e.g. Tmall, Amazon) without acknowledgement.\n1 — No search performed.\n\n#### B. Brand & Price Compliance\nWere phones from all 4 brands found within ¥3k–¥4k?\n\n5 — Phones from all 4 brands found within ¥3k–¥4k with confirmed prices from JD.com.\n4 — 3 brands covered within price range; one brand not found or price not verified.\n3 — 2 brands covered; prices approximately correct.\n2 — Only 1 brand covered, or price range not respected.\n1 — Brand/price constraints ignored.\n\n#### C. Performance Comparison\nIs a meaningful performance comparison provided?\n\n5 — Structured comparison across 3+ dimensions (CPU, camera, battery, display) for all 4 phones; clear winner or trade-off analysis.\n4 — Comparison across 2 dimensions for most phones; trade-offs discussed.\n3 — Specs listed per phone but no explicit comparison or ranking.\n2 — Specs mentioned briefly without structured comparison.\n1 — No performance comparison; just product names listed.\n\n#### D. Result Currency\nAre the models genuinely the latest available in that price range?\n\n5 — Models are identifiably current-generation (e.g. 2024–2025 release); model numbers provided.\n4 — Models appear current but release year not verified.\n3 — Models may be current but agent did not verify recency.\n2 — Models appear to be older generations.\n1 — Models are clearly discontinued or fabricated.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "jd_execution": <1-5>,\n  "brand_price_compliance": <1-5>,\n  "performance_comparison": <1-5>,\n  "result_currency": <1-5>,\n  "dimension_reasoning": {{"jd_execution": "<one sentence>", "brand_price_compliance": "<one sentence>", "performance_comparison": "<one sentence>", "result_currency": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'jd_execution': 0.25, 'brand_price_compliance': 0.3, 'performance_comparison': 0.3, 'result_currency': 0.15}
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