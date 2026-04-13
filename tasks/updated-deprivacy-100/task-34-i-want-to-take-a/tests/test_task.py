"""
LLM-as-judge evaluator for EvolveBench task-34.
Category: Travel & Planning
Task: "I want to take a train from Kanazawa to Hakone. Please help me plan
       how to buy tickets for travel on February 12, 2026."
"""
import os, json, re

TASK_INSTRUCTION = 'I want to take a train from Kanazawa to Hakone. Please help me plan how to buy tickets for travel on February 12, 2026.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks. This is a Japanese rail ticket planning task. The agent should identify the appropriate route from Kanazawa to Hakone, which typically involves the Shinkansen (bullet train) and possibly transfers. Relevant platforms include JR Tickets, Hyperdia, ekitan.com, and the JR Pass may be applicable.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Origin: Kanazawa, Ishikawa Prefecture\n- Destination: Hakone, Kanagawa Prefecture\n- Date: February 12, 2026\n- Platforms: JR Tickets, Hyperdia, ekitan.com are common Japanese rail booking and search platforms; JR Pass may be relevant\n- Agent should identify the route (likely via Shinkansen with transfers), check train options, and advise on ticket purchase process\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent identify the route from Kanazawa to Hakone (including any transfers)?\n- Did the agent search a Japanese rail platform (JR Tickets, Hyperdia, ekitan.com, etc.) for trains on Feb 12?\n- Were specific train options (train number, departure time, duration, price) provided?\n- Was the JR Pass or ticket purchase process discussed?\n- Were ticket purchase steps outlined?\n\n### Step 2: Dimension Scoring\n\n#### A. Route Clarity\nDid the agent correctly identify the full route (origin station → destination station, including transfers)?\n\n5 — Full route clearly identified (e.g. Kanazawa → Nagoya/Tokyo via Shinkansen → Hakone-Yumoto); transfer requirements and stations noted.\n4 — Route identified but transfer details incomplete or one leg vague.\n3 — General direction correct but route details vague.\n2 — Only one endpoint identified; agent asked for clarification without providing useful info.\n1 — Route not identified; response is generic travel advice.\n\n#### B. Ticket Platform Guidance\nDid the agent guide on how to buy tickets via Japanese rail platforms?\n\n5 — Agent navigated JR Tickets, Hyperdia, ekitan.com, or equivalent and provided specific steps or found actual available trains on Feb 12.\n4 — Agent described a Japanese rail booking platform process accurately without actually searching.\n3 — Agent mentioned a Japanese rail platform without detailed guidance.\n2 — Agent mentioned buying tickets without specifying a platform.\n1 — No platform guidance.\n\n#### C. Date Context Awareness\nDid the agent address the February 12, 2026 travel date appropriately?\n\n5 — Agent acknowledged the specific travel date, checked or advised on availability, and noted any relevant booking lead times.\n4 — Agent noted the date and provided general booking timing advice.\n3 — Agent referenced the date but did not address availability or booking implications.\n2 — Date noted without any contextual guidance.\n1 — Date context ignored.\n\n#### D. Actionability\nCan the user immediately act on the information provided?\n\n5 — Complete action plan: full route identified, platform specified, train options or search steps provided, ticket purchase guidance included.\n4 — Most elements present; one step (e.g. specific train options) missing.\n3 — Partial guidance; user would need to do significant additional research.\n2 — High-level overview only; not actionable.\n1 — No actionable information.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "route_clarity": <1-5>,\n  "ticket_platform_guidance": <1-5>,\n  "date_context_awareness": <1-5>,\n  "actionability": <1-5>,\n  "dimension_reasoning": {{"route_clarity": "<one sentence>", "ticket_platform_guidance": "<one sentence>", "date_context_awareness": "<one sentence>", "actionability": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'route_clarity': 0.25, 'ticket_platform_guidance': 0.3, 'date_context_awareness': 0.25, 'actionability': 0.2}
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