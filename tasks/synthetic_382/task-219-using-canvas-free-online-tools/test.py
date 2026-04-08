"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Design a YouTube channel thumbnail promoting a documentary about AI ethics using Canva.
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


TASK_INSTRUCTION = """Using Canva's free online tools, design a mockup for a YouTube channel thumbnail promoting a new documentary about AI ethics. Incorporate text overlays, an AI-themed background, and a call-to-action (e.g., 'Watch Now!'). Share details on the design choices you made."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to design a YouTube thumbnail using Canva's free online tools. The thumbnail must promote a documentary about AI ethics and include text overlays, an AI-themed background, and a call-to-action like 'Watch Now!'. A successful completion involves creating a visually appealing and relevant thumbnail while adhering to the specified requirements.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Canva's free online tools, design a mockup for a YouTube channel thumbnail promoting a new documentary about AI ethics. Incorporate text overlays, an AI-themed background, and a call-to-action (e.g., 'Watch Now!'). Share details on the design choices you made.

## Task-Specific Constraints
- Must use Canva's free online tools to create the thumbnail.
- The thumbnail must include text overlays relevant to AI ethics.
- The background must be AI-themed (e.g., futuristic, technological, or abstract AI imagery).
- The thumbnail must include a clear call-to-action, such as 'Watch Now!'.
- The agent must describe the design choices made, including why specific elements were chosen.
- The final output must be visually appealing and relevant to the documentary's theme.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Canva's free online tools to create the thumbnail?
- Does the thumbnail include text overlays relevant to AI ethics?
- Is the background AI-themed and visually appropriate?
- Is there a clear call-to-action included in the thumbnail?
- Are the design choices described in sufficient detail?

### Step 2: Dimension Scoring

#### A. Thumbnail Design Accuracy (0.35)
Measures whether the thumbnail meets the task requirements for text overlays, background, and call-to-action.

5 — All required elements are present, accurate, and visually appealing.
4 — All required elements are present but slightly less polished or cohesive.
3 — Most required elements are present but incomplete or poorly executed.
2 — Few required elements are present or poorly executed.
1 — No required elements are present.

#### B. Coverage of Requirements (0.30)
Measures whether the agent addressed all specified constraints.

5 — All constraints are fully addressed.
4 — Most constraints are addressed with minor omissions.
3 — Some constraints are addressed but with significant omissions.
2 — Few constraints are addressed.
1 — No constraints are addressed.

#### C. Design Specificity (0.20)
Measures the depth and specificity of the agent's design choices and explanations.

5 — Design choices are thoroughly explained with clear reasoning for all elements.
4 — Design choices are explained but with minor gaps in reasoning.
3 — Design choices are partially explained with significant gaps.
2 — Design choices are minimally explained.
1 — No explanation of design choices is provided.

#### D. Output Structure and Clarity (0.15)
Measures the clarity and organization of the agent's response.

5 — Response is well-organized, clear, and easy to follow.
4 — Response is organized but slightly less clear or detailed.
3 — Response is partially organized but lacks clarity in some areas.
2 — Response is poorly organized and unclear.
1 — Response is completely disorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "thumbnail_design_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "design_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "thumbnail_design_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "design_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "thumbnail_design_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "design_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())