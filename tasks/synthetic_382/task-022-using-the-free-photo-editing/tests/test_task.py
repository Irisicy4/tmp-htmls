"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Create a banner-sized graphic for a fictional online coffee shop using Photopea and public image resources.
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


TASK_INSTRUCTION = """Using the free photo editing tool Photopea, create a banner-sized graphic (1200x300px) for a fictional online coffee shop. Include the shop’s fictional name ('Java Paradise'), a tagline ('Wake Up to Paradise'), and an image of a coffee cup overlaid on a tropical background. Use publicly available image resources for the assets."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a banner-sized graphic (1200x300px) for a fictional online coffee shop using the free photo editing tool Photopea. The graphic must include the shop’s fictional name ('Java Paradise'), a tagline ('Wake Up to Paradise'), and an image of a coffee cup overlaid on a tropical background. The agent must use publicly available image resources for the assets.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using the free photo editing tool Photopea, create a banner-sized graphic (1200x300px) for a fictional online coffee shop. Include the shop’s fictional name ('Java Paradise'), a tagline ('Wake Up to Paradise'), and an image of a coffee cup overlaid on a tropical background. Use publicly available image resources for the assets.

## Task-Specific Constraints
- Must use Photopea to create the graphic.
- The graphic must be exactly 1200x300px in size.
- The name 'Java Paradise' must be clearly visible.
- The tagline 'Wake Up to Paradise' must be included and legible.
- The graphic must include a coffee cup image overlaid on a tropical background.
- All images used must come from publicly available resources (e.g., Pexels, Unsplash).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Photopea to create the graphic?
- Is the graphic exactly 1200x300px in size?
- Does the graphic include the shop name 'Java Paradise' and the tagline 'Wake Up to Paradise'?
- Does the graphic include a coffee cup image overlaid on a tropical background?
- Are all images sourced from publicly available resources?

### Step 2: Dimension Scoring

#### A. Graphic Accuracy (0.35)
Measures whether the graphic meets all specified requirements.

5 — The graphic is exactly 1200x300px, includes the shop name, tagline, coffee cup, and tropical background.
4 — The graphic meets most requirements but has minor issues (e.g., slight size deviation or missing minor elements).
3 — The graphic is partially complete but missing key elements (e.g., no tagline or incorrect size).
2 — The graphic is mostly incorrect or incomplete.
1 — The graphic is completely missing or irrelevant.

#### B. Coverage of Requirements (0.30)
Measures whether all required elements are present in the graphic.

5 — All required elements (shop name, tagline, coffee cup, tropical background) are present.
4 — One required element is missing or incomplete.
3 — Two required elements are missing or incomplete.
2 — Three required elements are missing or incomplete.
1 — None of the required elements are present.

#### C. Image Sourcing (0.20)
Measures whether the agent used publicly available resources for all images.

5 — All images are sourced from publicly available platforms like Pexels or Unsplash.
4 — Most images are sourced correctly, with one minor issue.
3 — Some images are sourced correctly, but others are missing or incorrectly sourced.
2 — Few images are sourced correctly.
1 — No images are sourced correctly or sources are unclear.

#### D. Design Quality (0.15)
Measures the overall aesthetic and readability of the graphic.

5 — The graphic is visually appealing, well-organized, and all text is legible.
4 — The graphic is mostly appealing but has minor readability or design issues.
3 — The graphic is somewhat appealing but has noticeable design flaws.
2 — The graphic is poorly designed or hard to read.
1 — The graphic is completely unappealing or illegible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "graphic_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "image_sourcing": <1-5>,
  "design_quality": <1-5>,
  "dimension_reasoning": {{
    "graphic_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "image_sourcing": "<one sentence citing specific evidence>",
    "design_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "graphic_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "image_sourcing": 0.20,
    "design_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())