"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Set up a Notion travel planning board for a trip to Greece with sections for itinerary, expenses, and packing list, using data from Lonely Planet, Expedia, and Airbnb.
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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_INSTRUCTION,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
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
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)])
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


TASK_INSTRUCTION = """Set up a Notion travel planning board for a trip to Greece. Create sections for itinerary, expenses, and packing list. Fill the itinerary section with placeholder activities from Lonely Planet and the expenses section with budget estimates sourced from Expedia and Airbnb."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves setting up a Notion travel planning board for a trip to Greece. The board must include three sections: itinerary, expenses, and packing list. The itinerary section must contain placeholder activities sourced from Lonely Planet, while the expenses section must include budget estimates sourced from Expedia and Airbnb. A successful completion requires accurate data sourcing, proper organization, and adherence to the task constraints.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a Notion travel planning board for a trip to Greece. Create sections for itinerary, expenses, and packing list. Fill the itinerary section with placeholder activities from Lonely Planet and the expenses section with budget estimates sourced from Expedia and Airbnb.

## Task-Specific Constraints
- Must create a Notion board with three distinct sections: itinerary, expenses, and packing list.
- Must source placeholder activities for the itinerary section from Lonely Planet.
- Must source budget estimates for the expenses section from both Expedia and Airbnb.
- Must include at least three activities in the itinerary section.
- Must provide at least two budget estimates in the expenses section (one from Expedia and one from Airbnb).
- Output must be clearly organized and reflect the required structure.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Lonely Planet, Expedia, and Airbnb as required?
- Did the agent create a Notion board with three distinct sections: itinerary, expenses, and packing list?
- Are there at least three activities in the itinerary section, and are they sourced from Lonely Planet?
- Are there at least two budget estimates in the expenses section, one from Expedia and one from Airbnb?
- Is the output well-organized and clearly structured as a travel planning board?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the Notion board contains all required sections and content.

5 — All sections (itinerary, expenses, packing list) are present and complete with accurate data.
4 — All sections are present but one is incomplete or inaccurate.
3 — All sections are present but more than one is incomplete or inaccurate.
2 — One or more sections are missing or mostly incorrect.
1 — No valid sections are present.

#### B. Source Coverage (0.30)
Measures whether the agent used all required platforms (Lonely Planet, Expedia, Airbnb).

5 — All three platforms were used, and data is correctly sourced from each.
4 — All three platforms were used, but one has minor sourcing issues.
3 — At least two platforms were used, with partial or incomplete sourcing.
2 — Only one platform was used, or sourcing is mostly incorrect.
1 — No platforms were used, or all sourcing is incorrect.

#### C. Detail Specificity (0.20)
Measures the level of detail in the itinerary and expense sections.

5 — Itinerary includes at least three detailed activities, and expenses include at least two detailed budget estimates.
4 — Itinerary includes three activities, but one lacks detail, or expenses are missing one budget estimate.
3 — Itinerary includes fewer than three activities, or expenses include only one budget estimate.
2 — Itinerary and expenses are mostly incomplete or lack detail.
1 — No meaningful details are included.

#### D. Output Organization (0.15)
Measures whether the output is well-structured and easy to follow.

5 — Output is clearly organized with distinct sections and logical formatting.
4 — Output is organized but has minor formatting or clarity issues.
3 — Output is partially organized but lacks clarity or logical structure.
2 — Output is poorly organized and difficult to follow.
1 — Output is completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "source_coverage": 0.30,
    "detail_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())