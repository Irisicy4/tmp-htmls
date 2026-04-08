"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Create a one-page travel flyer for a weekend getaway to Amsterdam, Netherlands using Canva, including a 3-day itinerary, estimated costs, and top attractions sourced from specified platforms.
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


TASK_INSTRUCTION = """Using Canva, configure a one-page travel flyer promoting a weekend getaway to Amsterdam, Netherlands. Include a 3-day itinerary section, estimated costs, and top attractions. Select a free travel flyer template and complete the population with data sourced from GetYourGuide, Visit Amsterdam, and Google Flights."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a one-page travel flyer using Canva to promote a weekend getaway to Amsterdam, Netherlands. The flyer must include a 3-day itinerary, estimated costs, and top attractions. The agent must source data from GetYourGuide, Visit Amsterdam, and Google Flights.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Canva, configure a one-page travel flyer promoting a weekend getaway to Amsterdam, Netherlands. Include a 3-day itinerary section, estimated costs, and top attractions. Select a free travel flyer template and complete the population with data sourced from GetYourGuide, Visit Amsterdam, and Google Flights.

## Task-Specific Constraints
- Must use Canva to create the flyer and select a free template.
- Must source itinerary and attraction details from Visit Amsterdam and GetYourGuide.
- Must source flight cost estimates from Google Flights.
- Flyer must include a structured 3-day itinerary section.
- Flyer must include estimated costs for flights, accommodations, and activities.
- Flyer must include at least 3 top attractions in Amsterdam.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Canva and select a free flyer template?
- Did the agent source itinerary and attraction details from Visit Amsterdam and GetYourGuide?
- Did the agent source flight cost estimates from Google Flights?
- Is the flyer organized with a clear 3-day itinerary section, estimated costs, and top attractions?
- Are the estimated costs and itinerary details accurate and sourced correctly?

### Step 2: Dimension Scoring

#### A. Deliverable Completeness (0.35)
Measures whether the flyer includes all required sections (3-day itinerary, estimated costs, and top attractions).

5 — All required sections are present, complete, and well-integrated.
4 — All required sections are present but lack minor details or integration.
3 — Most required sections are present but incomplete or poorly integrated.
2 — Few required sections are present, and they are incomplete.
1 — No required sections are present.

#### B. Source Utilization (0.30)
Measures whether the agent used all specified platforms (Canva, GetYourGuide, Visit Amsterdam, Google Flights).

5 — All specified platforms were used, and data was accurately sourced.
4 — Most specified platforms were used, with minor sourcing gaps.
3 — Some specified platforms were used, but sourcing was incomplete.
2 — Few specified platforms were used, and sourcing was poor.
1 — No specified platforms were used.

#### C. Detail Accuracy (0.25)
Measures the accuracy and specificity of the itinerary, costs, and attractions included.

5 — All details are accurate, specific, and sourced correctly.
4 — Most details are accurate, with minor inaccuracies or omissions.
3 — Some details are accurate, but there are notable inaccuracies or omissions.
2 — Few details are accurate, and sourcing is unreliable.
1 — No accurate details are present.

#### D. Formatting and Presentation (0.10)
Measures the organization and visual appeal of the flyer.

5 — Flyer is well-organized, visually appealing, and easy to read.
4 — Flyer is organized and readable, with minor formatting issues.
3 — Flyer is somewhat organized but has notable formatting issues.
2 — Flyer is disorganized and difficult to read.
1 — Flyer is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_completeness": <1-5>,
  "source_utilization": <1-5>,
  "detail_accuracy": <1-5>,
  "formatting_and_presentation": <1-5>,
  "dimension_reasoning": {{
    "deliverable_completeness": "<one sentence citing specific evidence>",
    "source_utilization": "<one sentence citing specific evidence>",
    "detail_accuracy": "<one sentence citing specific evidence>",
    "formatting_and_presentation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_completeness": 0.35,
    "source_utilization": 0.30,
    "detail_accuracy": 0.25,
    "formatting_and_presentation": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())