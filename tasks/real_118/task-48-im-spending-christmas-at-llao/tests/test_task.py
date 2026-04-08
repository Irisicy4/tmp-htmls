"""task-48: Llao Llao Hotel & Resort Argentine Patagonia Christmas itinerary"""
import os, json, re

TASK_INSTRUCTION = "I'm spending Christmas at Llao Llao Hotel & Resort in Argentine Patagonia. Please write an efficient itinerary for how to move around inside the resort and nearby area."
PASS_THRESHOLD = 3.0
SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent produced a high-quality, efficient itinerary for spending Christmas at Llao Llao Hotel & Resort in Argentine Patagonia."""
USER_PROMPT_TEMPLATE = """## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Venue: Llao Llao Hotel & Resort in Argentine Patagonia specifically — content must be resort-specific, not generic\n- Occasion: Christmas — seasonal events and activities should be included if available\n- Focus: efficiency of movement inside the resort and nearby area — routing and timing matter\n- Nearby area includes: Nahuel Huapi Lake boat trips, hiking trails (e.g. Cerro Lopez, Circuito Chico), and the town of Bariloche\n- Output: a structured itinerary with times, locations, and movement guidance\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n### Step 1: Evidence Analysis\n- Did the agent research Llao Llao Hotel & Resort in Argentine Patagonia specifically?\n- Does the itinerary include resort-specific locations (hotel facilities, grounds, nearby Patagonian attractions)?\n- Are Christmas-specific events or activities mentioned?\n- Is movement efficiency addressed (routing, travel times between venues)?\n- Is the itinerary structured with times?\n\n### Step 2: Dimension Scoring\n\n#### A. Resort Specificity (0.30)\n5 — Itinerary references specific Llao Llao Hotel & Resort facilities, grounds, and nearby Patagonian attractions (e.g. Nahuel Huapi Lake, hiking trails, Bariloche).\n4 — Mostly resort-specific but some generic content.\n3 — Mix of resort-specific and generic travel advice.\n2 — Mostly generic resort itinerary not tailored to Llao Llao.\n1 — No resort-specific content.\n\n#### B. Christmas Relevance (0.20)\n5 — Christmas events, decorations, or seasonal programming at Llao Llao or in the Patagonia/Bariloche area specifically mentioned.\n4 — Christmas theme addressed but events are generic rather than resort-specific.\n3 — Christmas mentioned but itinerary is not seasonally adapted.\n2 — Itinerary could apply to any time of year.\n1 — Christmas not addressed at all.\n\n#### C. Movement Efficiency (0.30)\n5 — Itinerary explicitly optimises routing (e.g. geographically grouped, minimises backtracking, accounts for travel times between resort and nearby areas such as the lake or Bariloche).\n4 — Good sequencing but efficiency rationale not explicitly stated.\n3 — Itinerary is sequential but not clearly optimised for movement.\n2 — Activities listed without regard to location or routing.\n1 — No movement guidance.\n\n#### D. Structure & Completeness (0.20)\n5 — Full day itinerary with times, locations, durations, and practical tips.\n4 — Times and locations present but missing some practical details.\n3 — Structured list but missing times or durations.\n2 — Paragraph description without clear schedule.\n1 — No structure.\n\n### Step 3: Output\nRespond ONLY with valid JSON inside <Answer></Answer> tags:\n\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "resort_specificity": <1-5>,\n  "christmas_relevance": <1-5>,\n  "movement_efficiency": <1-5>,\n  "structure_completeness": <1-5>,\n  "dimension_reasoning": {{\n    "resort_specificity": "<one sentence>",\n    "christmas_relevance": "<one sentence>",\n    "movement_efficiency": "<one sentence>",\n    "structure_completeness": "<one sentence>"\n  }},\n  "overall_score": <weighted average>,\n  "passed": <true or false>\n}}\n</Answer>\n"""
DIMENSION_WEIGHTS = {"resort_specificity": 0.30, "christmas_relevance": 0.20, "movement_efficiency": 0.30, "structure_completeness": 0.20}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task_instruction=TASK_INSTRUCTION, agent_response=agent_response, execution_summary=execution_summary or "Not available.")}], max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", ""); aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {}); aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result); execution_summary = result.get("execution_summary", "")
    if not agent_response.strip(): return {"passed": False, "feedback": "No response found from agent.", "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)]) if abs(float(overall) - PASS_THRESHOLD) <= 0.5 else (first.__setitem__("_votes_used", 1) or first)
    else: scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0); passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning: lines.append("\nDimension reasoning:"); [lines.append(f"  {dim}: {reason}") for dim, reason in reasoning.items()]
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {"passed": bool(passed), "feedback": "\n".join(lines), "details": {"task_completed": result.get("status") == "success", "overall_score": overall, "dimension_scores": {d: scores.get(d) for d in DIMENSIONS}, "evidence_summary": scores.get("evidence_summary"), "dimension_reasoning": scores.get("dimension_reasoning"), "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)}}