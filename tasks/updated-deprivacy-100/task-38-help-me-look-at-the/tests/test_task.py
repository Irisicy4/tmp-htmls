"""
LLM-as-judge evaluator for EvolveBench task-38.
Category: Shopping
"""
import os, json, re

TASK_INSTRUCTION = 'Help me look at the price and configuration of Mac Mini, Mac Studio, MacBook Pro and MacBook Air, and give a trade-off for choosing. I want to compare their price and performance and find the one with the best value.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks. This is an Apple product comparison and value assessment task covering 4 product lines.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Products: Mac Mini, Mac Studio, MacBook Pro, MacBook Air — all 4 required\n- Data needed: current prices AND key specs (chip, RAM, storage, GPU, display)\n- Analysis required: trade-off comparison AND best-value recommendation\n- Source: should reference apple.com or other authoritative pricing\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Were all 4 Apple products covered?\n- Are current prices and specs provided for each?\n- Is a trade-off analysis (not just specs listing) present?\n- Is a best-value recommendation made with justification?\n\n### Step 2: Dimension Scoring\n\n#### A. Product Coverage\nAre all 4 products covered with price and spec data?\n\n5 — All 4 products covered with current base price and key specs (chip, RAM, storage minimum).\n4 — 3 of 4 covered fully; one has partial info.\n3 — All 4 mentioned but 1–2 lack specific prices or specs.\n2 — Only 2–3 products covered with data.\n1 — Fewer than 2 products or no data.\n\n#### B. Spec & Price Accuracy\nIs the information accurate and current?\n\n5 — Prices and specs are consistent with current Apple lineup (M4/M4 Pro/M4 Max chips as of 2025–2026); no clearly wrong claims.\n4 — Mostly accurate; 1–2 minor errors or slightly outdated specs.\n3 — Generally correct direction; some specs are vague or approximate.\n2 — Significant inaccuracies (wrong chip generation, wrong price tier).\n1 — Fabricated or clearly wrong.\n\n#### C. Trade-off Analysis\nDoes the agent provide genuine trade-off analysis (not just specs listing)?\n\n5 — Explicit trade-offs discussed: portability vs performance, desktop vs laptop, value for money per use case (e.g. creative professionals vs students vs home users).\n4 — Trade-offs discussed for 2–3 dimension pairs; one use-case segment not addressed.\n3 — Implicit comparison (e.g. "Mac Studio is faster but costs more") without structured trade-off framework.\n2 — Products listed side by side without any trade-off commentary.\n1 — No trade-off analysis.\n\n#### D. Value Recommendation\nDoes the agent make a specific best-value recommendation with justification?\n\n5 — Clear recommendation with stated assumption (e.g. "for most users, Mac Mini M4 offers best value because..."); alternative for power users noted.\n4 — Recommendation made with brief justification; alternatives not discussed.\n3 — Agent suggests a product without clear justification.\n2 — Multiple options suggested without a clear recommendation.\n1 — No recommendation.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "product_coverage": <1-5>,\n  "spec_price_accuracy": <1-5>,\n  "tradeoff_analysis": <1-5>,\n  "value_recommendation": <1-5>,\n  "dimension_reasoning": {{"product_coverage": "<one sentence>", "spec_price_accuracy": "<one sentence>", "tradeoff_analysis": "<one sentence>", "value_recommendation": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'product_coverage': 0.2, 'spec_price_accuracy': 0.25, 'tradeoff_analysis': 0.35, 'value_recommendation': 0.2}
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