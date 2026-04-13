"""
LLM-as-judge evaluator for EvolveBench task-57.

Category: Travel & Planning
Task: Provide a detailed route guide for the Tongariro Alpine Crossing combining the
      Mangatepopo Valley approach and the Emerald Lakes descent route.
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

def _call(agent_response, execution_summary, system_prompt, user_prompt_template, task_instruction):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(
                    task_instruction=task_instruction,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes, dimensions, weights, pass_threshold):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dimensions)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in dimensions}
    overall = sum(aggregated[d] * weights[d] for d in dimensions)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= pass_threshold
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated


TASK_INSTRUCTION = """Please provide a detailed route guide for the Tongariro Alpine Crossing combining the Mangatepopo Valley approach and the Emerald Lakes descent route."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent produced a comprehensive, specific hiking route guide for the Tongariro Alpine Crossing in New Zealand, specifically covering the Mangatepopo Valley approach and the Emerald Lakes descent route."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Specific route: must cover the Mangatepopo Valley approach AND the Emerald Lakes descent
- Location: Tongariro National Park, New Zealand (Tongariro Alpine Crossing)
- Key waypoints: Mangatepopo Valley, Soda Springs, South Crater, Red Crater, Emerald Lakes, Blue Lake, Ketetahi area
- Guide must include: distance (~19.4 km), elevation gain (~1,600 m), duration (typically 7–9 hours), key waypoints, difficulty level
- Practical info: transportation (shuttle required — one-way track), best season (Oct–Apr), gear recommendations, volcanic hazard and weather change warnings
- Safety considerations: active volcanic area, NZ DOC alerts, sudden weather changes, appropriate footwear and layers

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research this specific crossing, including both the Mangatepopo Valley approach and Emerald Lakes descent?
- Are key waypoints (South Crater, Red Crater, Emerald Lakes) identified and described?
- What specific distances, durations, and elevation figures are mentioned?
- Is practical information (shuttle transport, season, gear, volcanic hazards) included?

### Step 2: Dimension Scoring

#### A. Route Specificity (0.35)
Does the guide cover the specific Tongariro Alpine Crossing route combination?

5 — Both Mangatepopo Valley approach and Emerald Lakes descent explicitly covered with specific waypoints (e.g. South Crater, Red Crater, Blue Lake) and route description.
4 — Both elements covered but one is less detailed.
3 — One element covered well; the other mentioned briefly.
2 — Generic Tongariro Alpine Crossing guide without specific mention of both named route sections.
1 — No route-specific content.

#### B. Technical Detail (0.3)
Does the guide include distance, elevation, duration, and difficulty?

5 — All four metrics present with specific numbers (e.g. ~19.4 km, ~1,600 m elevation gain, 7–9 hours, Grade: Demanding).
4 — Three of four metrics present.
3 — Two of four metrics present.
2 — Only one metric present.
1 — No technical details.

#### C. Waypoint Coverage (0.2)
Are key waypoints along the route described?

5 — 5 or more named waypoints described (e.g. Mangatepopo car park, Soda Springs, South Crater, Red Crater summit, Emerald Lakes, Blue Lake, Ketetahi Shelter).
4 — 3–4 named waypoints.
3 — 1–2 named waypoints.
2 — Waypoints mentioned but not named or described.
1 — No waypoints.

#### D. Practical Information (0.15)
Is practical travel information included?

5 — Transportation (shuttle logistics), best season, gear recommendations, and volcanic/weather safety warnings all addressed.
4 — Three of four practical elements addressed.
3 — Two practical elements.
2 — One practical element.
1 — No practical information.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "route_specificity": <1-5>,
  "technical_detail": <1-5>,
  "waypoint_coverage": <1-5>,
  "practical_information": <1-5>,
  "dimension_reasoning": {{
    "route_specificity": "<one sentence citing specific evidence>",
    "technical_detail": "<one sentence citing specific evidence>",
    "waypoint_coverage": "<one sentence citing specific evidence>",
    "practical_information": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "route_specificity": 0.35,
    "technical_detail": 0.3,
    "waypoint_coverage": 0.2,
    "practical_information": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

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