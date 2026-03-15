"""
LLM-as-judge evaluator for EvolveBench task-19.

Category: Travel & Planning
Task: List 10 places in central Paris with good Eiffel Tower views, with addresses,
      directions from city center, nearby attractions, and nearby restaurants for each.
"""

import os, json, re

TASK_INSTRUCTION = (
    "I am now in central Paris. Please tell me 10 places with a good view of the Eiffel Tower, "
    "their addresses, and for each place provide directions from the city center, nearby "
    "attractions, and nearby restaurants."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully compiled a comprehensive list of Eiffel Tower viewpoints with detailed per-location information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Quantity: exactly 10 places (not fewer)
- Per-place requirements: (1) address, (2) directions from city center, (3) nearby attractions, (4) nearby restaurants
- Viewpoint quality: places must actually have a good view of the Eiffel Tower (not just near it)
- Research: agent should search for current, accurate location information rather than relying solely on prior knowledge

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- How many viewpoint locations did the agent list? Name them briefly.
- For each location: does it have (a) address, (b) directions, (c) nearby attractions, (d) nearby restaurants?
- Are the locations genuinely good Eiffel Tower viewpoints (e.g. Trocadéro, Champ de Mars, Bir-Hakeim Bridge)?
- Did the agent search for this information or rely on prior knowledge? Cite evidence.
- Are any locations clearly wrong (not in Paris, no Eiffel Tower view)?

### Step 2: Dimension Scoring

#### A. Quantity & Viewpoint Quality
Did the agent list 10 places that genuinely offer good Eiffel Tower views?

5 — Exactly 10 locations listed; all are well-known or credibly positioned viewpoints with actual Eiffel Tower sightlines.
4 — 10 locations listed but 1–2 are questionable viewpoints (e.g. too far, obstructed view).
3 — 8–9 locations listed, or 10 listed but several are not genuine viewpoints.
2 — 5–7 locations listed, or majority are not good viewpoints.
1 — Fewer than 5 locations or locations are clearly wrong.

#### B. Information Completeness Per Location
Does each location have all four required components?

5 — All 4 components (address, directions, nearby attractions, nearby restaurants) present for 9–10 locations.
4 — All 4 components present for 7–8 locations; 2–3 have minor gaps.
3 — Most locations have 2–3 components; one component (e.g. directions or restaurants) consistently missing.
2 — Only 1–2 components present for most locations.
1 — Required components largely absent; mostly just names and addresses.

#### C. Information Specificity
Is the per-location information specific and actionable?

5 — Addresses include street names; directions reference specific Metro lines/stops or landmarks; restaurants/attractions are named specifically.
4 — Most information is specific; directions for 2–3 locations are vague (e.g. "take the Metro").
3 — Information is present but mostly generic (e.g. "nearby cafes", "local museums") without specific names.
2 — Information is very thin; mostly location names with minimal useful detail.
1 — No specific information; response is a generic Paris tourism overview.

#### D. Research Evidence
Did the agent actively research rather than rely entirely on prior knowledge?

5 — Clear evidence of web search or map navigation in trace; at least some locations verified with current information.
4 — Some search evidence; agent may have combined prior knowledge with light research.
3 — No clear search evidence but information appears accurate and is plausibly current.
2 — Response appears to be entirely from prior knowledge with no research attempted.
1 — Information is clearly outdated, inaccurate, or fabricated.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "quantity_viewpoint_quality": <1-5>,
  "information_completeness": <1-5>,
  "information_specificity": <1-5>,
  "research_evidence": <1-5>,
  "dimension_reasoning": {{
    "quantity_viewpoint_quality": "<one sentence citing specific evidence>",
    "information_completeness": "<one sentence citing specific evidence>",
    "information_specificity": "<one sentence citing specific evidence>",
    "research_evidence": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "quantity_viewpoint_quality": 0.25,
    "information_completeness":   0.35,
    "information_specificity":    0.25,
    "research_evidence":          0.15,
}
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

def _parse_answer_tag(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }