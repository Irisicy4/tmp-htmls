"""
LLM-as-judge evaluator for EvolveBench task-36.
Category: Shopping
"""
import os, json, re

TASK_INSTRUCTION = 'I want to buy an M5 MacBook Pro with at least 1TB SSD and 16GB RAM. Research apple.com, Amazon, and Best Buy for the best prices and detailed specs. Then open Google Docs, create a new document and add the specs, pricing and other details to it.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = "You are an expert evaluator for AI agent benchmarks assessing a multi-site product research task followed by Google Docs document creation. Note: As of early 2026, the M5 MacBook Pro may not yet exist — the agent should handle this gracefully (search, note availability status, and use the closest available model if M5 isn't released)."

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Specs required: ≥1TB SSD AND ≥16GB RAM (both are hard minimums)\n- Sites: apple.com, Amazon, Best Buy — all three must be checked\n- Note: M5 MacBook Pro may not be released; agent should note availability and use M4 Pro/M4 Max if necessary\n- Output: Google Docs document with specs, pricing from all three sites, and comparison\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent visit all 3 sites (apple.com, Amazon, Best Buy)?\n- Were ≥1TB/≥16GB models found? Are specs verified?\n- Did the agent handle M5 availability (note if not released, use M4)?\n- Was a Google Docs document created with the findings?\n\n### Step 2: Dimension Scoring\n\n#### A. Multi-Site Research\nDid the agent research all three required sites?\n\n5 — All 3 sites (apple.com, Amazon, Best Buy) navigated; prices retrieved from each.\n4 — 2 of 3 sites researched; one was inaccessible or skipped.\n3 — 1 site researched directly; others referenced from prior knowledge or general search.\n2 — None of the 3 sites navigated; response from prior knowledge.\n1 — No research performed.\n\n#### B. Spec & Price Accuracy\nAre the specifications and prices accurate and correctly filtered?\n\n5 — ≥1TB/≥16GB models identified with correct specs; prices from all available sites; M5 availability correctly handled.\n4 — Specs correct but price from 1 site missing or M5 availability not explicitly addressed.\n3 — Models found but specs not verified to meet minimums; prices approximate.\n2 — Models listed without spec verification; prices not from the required sites.\n1 — Specs/prices fabricated or completely wrong.\n\n#### C. Google Docs Save\nWas a Google Docs document created with the research?\n\n5 — Google Docs document created; URL or trace confirmation; contains specs, pricing comparison, and summary.\n4 — Document creation attempted; trace confirms Google Docs API call; content present.\n3 — Agent compiled research in response but did not create Google Docs document.\n2 — Agent mentioned creating a doc without doing so.\n1 — No document created.\n\n#### D. Information Quality\nIs the overall information well-organised and useful for a purchase decision?\n\n5 — Price comparison table across sites; specs clearly listed; best value recommendation given.\n4 — Information present but comparison is narrative rather than structured table.\n3 — Information present but disorganised; hard to compare options at a glance.\n2 — Partial information; missing prices or specs from most sites.\n1 — No useful purchase information.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "multi_site_research": <1-5>,\n  "spec_price_accuracy": <1-5>,\n  "google_docs_save": <1-5>,\n  "information_quality": <1-5>,\n  "dimension_reasoning": {{"multi_site_research": "<one sentence>", "spec_price_accuracy": "<one sentence>", "google_docs_save": "<one sentence>", "information_quality": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'multi_site_research': 0.3, 'spec_price_accuracy': 0.25, 'google_docs_save': 0.3, 'information_quality': 0.15}
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