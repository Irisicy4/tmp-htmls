"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Create a mood board for a coffee brand with a rustic aesthetic using Canva.
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


TASK_INSTRUCTION = """Create a mood board for a coffee brand with a rustic aesthetic using Canva. Include at least five elements: a color palette, one texture (e.g., wood), one typography suggestion, two images reflecting the brand style, and an icon or logo placeholder."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a mood board for a coffee brand with a rustic aesthetic using Canva. The deliverable must include five specific elements: a color palette, one texture, one typography suggestion, two images reflecting the brand style, and an icon or logo placeholder. The task is in the domain of design and requires using multiple platforms to gather resources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a mood board for a coffee brand with a rustic aesthetic using Canva. Include at least five elements: a color palette, one texture (e.g., wood), one typography suggestion, two images reflecting the brand style, and an icon or logo placeholder.

## Task-Specific Constraints
- Must visit Canva, Pexels, and Color Hunt to gather resources.
- Must include a color palette with at least three colors.
- Must include one texture relevant to the rustic aesthetic (e.g., wood or burlap).
- Must include two images that align with the coffee brand's rustic style.
- Must suggest one typography style suitable for the brand.
- Must include a placeholder for an icon or logo.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Canva, Pexels, and Color Hunt? Which platforms were actually visited?
- Does the response include all five required elements (color palette, texture, typography, images, icon/logo placeholder)?
- Is the color palette described with at least three colors?
- Are the images and texture relevant to the rustic aesthetic?
- Is the typography suggestion appropriate for a coffee brand?

### Step 2: Dimension Scoring

#### A. Deliverable Completeness (0.35)
Measures whether all required elements are present and correctly implemented.

5 — All five elements are present, correctly implemented, and align with the rustic aesthetic.
4 — Four elements are present and correctly implemented; one element may be incomplete or slightly misaligned.
3 — Three elements are present and usable; others are missing or poorly implemented.
2 — One or two elements are present but incomplete or irrelevant.
1 — None of the required elements are present or usable.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and utilized them effectively.

5 — Agent visited Canva, Pexels, and Color Hunt, and utilized resources from all three.
4 — Agent visited two platforms and utilized resources effectively; one platform may be missing.
3 — Agent visited one platform and utilized resources; others are missing.
2 — Agent visited one platform but did not utilize resources effectively.
1 — Agent did not visit any required platforms.

#### C. Aesthetic Relevance (0.25)
Measures whether the chosen elements align with the rustic aesthetic.

5 — All elements (color palette, texture, images, typography) strongly align with the rustic aesthetic.
4 — Most elements align with the rustic aesthetic; one may be slightly off.
3 — Some elements align with the rustic aesthetic; others may be irrelevant or generic.
2 — Few elements align with the rustic aesthetic; most are irrelevant.
1 — No elements align with the rustic aesthetic.

#### D. Output Structure and Clarity (0.10)
Measures whether the response is well-organized and easy to interpret.

5 — Response is highly organized, with clear descriptions and structured formatting.
4 — Response is organized but may lack minor clarity or formatting.
3 — Response is partially organized; some parts may be unclear or unstructured.
2 — Response is poorly organized and difficult to interpret.
1 — Response is completely unorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "aesthetic_relevance": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "deliverable_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "aesthetic_relevance": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_completeness": 0.35,
    "platform_coverage": 0.30,
    "aesthetic_relevance": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())