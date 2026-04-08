"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Create a promotional image for a furniture store sale using Photopea and Unsplash.
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


TASK_INSTRUCTION = """Using Photopea, create a promotional image for a fictional furniture store sale. Add text overlay '50% Off All Sofas!' and include a background image of a showroom sourced from Unsplash. Export the image in PNG format and report the final design details."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a promotional image for a fictional furniture store sale. The agent must use Photopea to design the image, include a showroom background sourced from Unsplash, add the text overlay '50% Off All Sofas!', and export the image in PNG format. A successful completion includes a valid PNG image and a report of the design details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Photopea, create a promotional image for a fictional furniture store sale. Add text overlay '50% Off All Sofas!' and include a background image of a showroom sourced from Unsplash. Export the image in PNG format and report the final design details.

## Task-Specific Constraints
- Must use Photopea to create the image.
- Must source the background image from Unsplash.
- Must include the text overlay '50% Off All Sofas!' in the design.
- Must export the final image in PNG format.
- Must provide a report detailing the design choices (e.g., font used, colors, etc.).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both Photopea and Unsplash as required?
- Did the agent include the text overlay '50% Off All Sofas!' in the design?
- Was the background image sourced from Unsplash, and does it resemble a showroom?
- Was the final image exported in PNG format?
- Did the agent provide a detailed report of the design choices?

### Step 2: Dimension Scoring

#### A. Image Design Accuracy (0.35)
Measures whether the final image meets the design requirements.

5 — Includes the correct text overlay, showroom background, and is exported in PNG format.
4 — Minor issues with one element (e.g., text placement or background relevance).
3 — Partial completion (e.g., missing text or incorrect export format).
2 — Major issues or incomplete image.
1 — No valid image produced.

#### B. Platform Usage Compliance (0.30)
Measures whether the agent used the required platforms (Photopea and Unsplash).

5 — Both platforms used correctly and as intended.
4 — Both platforms used, but with minor issues.
3 — Only one platform used or significant errors in usage.
2 — Attempted but failed to use platforms correctly.
1 — Did not use the required platforms.

#### C. Design Specificity and Detail (0.20)
Measures the quality and detail of the design choices.

5 — Provides a detailed report with font, color, and layout information.
4 — Provides most details but misses minor aspects.
3 — Provides minimal details (e.g., only font or color mentioned).
2 — Provides vague or unclear details.
1 — No details provided.

#### D. Output Organization and Clarity (0.15)
Measures the clarity and organization of the agent's response.

5 — Response is clear, well-structured, and easy to follow.
4 — Response is mostly clear but has minor organizational issues.
3 — Response is somewhat clear but lacks structure.
2 — Response is unclear or poorly structured.
1 — Response is incoherent or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "image_design_accuracy": <1-5>,
  "platform_usage_compliance": <1-5>,
  "design_specificity_and_detail": <1-5>,
  "output_organization_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "image_design_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_compliance": "<one sentence citing specific evidence>",
    "design_specificity_and_detail": "<one sentence citing specific evidence>",
    "output_organization_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "image_design_accuracy": 0.35,
    "platform_usage_compliance": 0.30,
    "design_specificity_and_detail": 0.20,
    "output_organization_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())