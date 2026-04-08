"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Create a thumbnail mockup for a YouTube video about 'AI and Content Creation' using Canva, including a title overlay, stock image/icon, and contrasting colors, and download it as a PNG file.
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


TASK_INSTRUCTION = """Use Canva (no login required) to create a thumbnail mockup for a YouTube video about 'AI and Content Creation.' Include a title overlay, stock image/icon, and use contrasting colors for visibility. Download the mockup as a PNG file."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Canva to create a thumbnail mockup for a YouTube video about 'AI and Content Creation.' The thumbnail must include a title overlay, a stock image or icon, and contrasting colors for visibility. The final deliverable must be downloaded as a PNG file.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Canva (no login required) to create a thumbnail mockup for a YouTube video about 'AI and Content Creation.' Include a title overlay, stock image/icon, and use contrasting colors for visibility. Download the mockup as a PNG file.

## Task-Specific Constraints
- Must use Canva to create the thumbnail.
- Thumbnail must include a title overlay related to 'AI and Content Creation.'
- Thumbnail must include at least one stock image or icon sourced from Canva, Unsplash, or Pexels.
- Thumbnail must use contrasting colors for visibility.
- Final deliverable must be downloaded as a PNG file.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Canva to create the thumbnail?
- Does the thumbnail include a title overlay related to 'AI and Content Creation'?
- Does the thumbnail include at least one stock image or icon sourced from Canva, Unsplash, or Pexels?
- Are contrasting colors used for visibility in the thumbnail?
- Was the thumbnail downloaded as a PNG file?

### Step 2: Dimension Scoring

#### A. Thumbnail Accuracy (0.35)
Measures whether the thumbnail meets the core requirements of the task.

5 — Thumbnail includes a relevant title overlay, a stock image/icon, and contrasting colors, and is downloaded as a PNG file.
4 — Thumbnail includes most required elements but is missing minor details (e.g., weak color contrast).
3 — Thumbnail includes some required elements but lacks critical components (e.g., no title overlay or stock image).
2 — Thumbnail is mostly incomplete or incorrect.
1 — Thumbnail is absent or entirely wrong.

#### B. Platform Usage (0.30)
Measures whether the agent used the required platforms appropriately.

5 — Agent used Canva and sourced stock images/icons from Unsplash or Pexels as needed.
4 — Agent used Canva but did not source stock images/icons from other platforms.
3 — Agent used Canva but did not fully utilize platform features (e.g., no stock images/icons).
2 — Agent attempted platform usage but failed to produce a usable thumbnail.
1 — Agent did not use Canva or other specified platforms.

#### C. Visual Design Quality (0.20)
Measures the aesthetic quality and visibility of the thumbnail.

5 — Thumbnail design is visually appealing, with excellent color contrast and layout.
4 — Thumbnail design is good but has minor flaws in color contrast or layout.
3 — Thumbnail design is acceptable but lacks polish or visibility.
2 — Thumbnail design is poor, with major issues in color contrast or layout.
1 — Thumbnail design is absent or unusable.

#### D. Deliverable Format (0.15)
Measures whether the final deliverable meets the required format.

5 — Thumbnail was downloaded as a PNG file and confirmed in the response.
4 — Thumbnail was downloaded but format confirmation is unclear.
3 — Thumbnail was created but not downloaded as a PNG file.
2 — Thumbnail was attempted but no deliverable was produced.
1 — No deliverable was created.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "thumbnail_accuracy": <1-5>,
  "platform_usage": <1-5>,
  "visual_design_quality": <1-5>,
  "deliverable_format": <1-5>,
  "dimension_reasoning": {{
    "thumbnail_accuracy": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "visual_design_quality": "<one sentence citing specific evidence>",
    "deliverable_format": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "thumbnail_accuracy": 0.35,
    "platform_usage": 0.30,
    "visual_design_quality": 0.20,
    "deliverable_format": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())