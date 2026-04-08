"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Create a YouTube thumbnail template with specific design elements using Canva or Photopea.
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


TASK_INSTRUCTION = """Visit Canva or Photopea and create a thumbnail template for a YouTube video titled 'Top AI Trends in 2024'. Ensure the template includes bold text, an AI-themed background, and placeholder sections for an image and a subtitle."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a YouTube thumbnail template using Canva or Photopea. The template must include bold text, an AI-themed background, and placeholder sections for an image and a subtitle. Success is defined by the presence of these elements in the final design.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit Canva or Photopea and create a thumbnail template for a YouTube video titled 'Top AI Trends in 2024'. Ensure the template includes bold text, an AI-themed background, and placeholder sections for an image and a subtitle.

## Task-Specific Constraints
- Must use either Canva or Photopea as the design platform.
- The thumbnail must include bold text prominently displayed.
- The background must be AI-themed (e.g., futuristic, digital patterns, or AI-related imagery).
- Placeholder sections for an image and a subtitle must be clearly visible.
- The final output must be formatted as a reusable template.
- The agent must describe the steps taken to create the design.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Canva or Photopea? Which platform was used?
- Does the thumbnail include bold text prominently displayed?
- Is the background AI-themed and relevant to the task?
- Are placeholder sections for an image and a subtitle clearly visible?
- Is the final output described as a reusable template?

### Step 2: Dimension Scoring

#### A. Template Design Accuracy (0.35)
Measures whether the thumbnail template includes all required elements.

5 — Includes bold text, AI-themed background, and placeholder sections for both an image and a subtitle.
4 — Includes most required elements but misses one minor detail.
3 — Includes some required elements but lacks significant details.
2 — Includes few required elements; mostly incomplete.
1 — Does not include any required elements.

#### B. Platform Usage and Execution (0.30)
Measures whether the agent correctly used Canva or Photopea to complete the task.

5 — Successfully used Canva or Photopea and provided a clear description of the steps taken.
4 — Used Canva or Photopea but with minor execution errors or unclear descriptions.
3 — Used Canva or Photopea but with significant execution errors or vague descriptions.
2 — Attempted to use Canva or Photopea but failed to complete the task.
1 — Did not use Canva or Photopea.

#### C. Background and Theme Relevance (0.20)
Measures the relevance and quality of the AI-themed background.

5 — Background is highly relevant, visually appealing, and clearly AI-themed.
4 — Background is relevant but lacks visual appeal or clarity.
3 — Background is somewhat relevant but generic or unclear.
2 — Background is mostly irrelevant or poorly executed.
1 — Background is absent or completely irrelevant.

#### D. Output Structure and Reusability (0.15)
Measures whether the final output is well-organized and reusable.

5 — Output is well-organized, clearly reusable, and described in detail.
4 — Output is reusable but lacks detailed organization or description.
3 — Output is minimally reusable and lacks clarity.
2 — Output is poorly organized and not reusable.
1 — Output is absent or completely unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_design_accuracy": <1-5>,
  "platform_usage_and_execution": <1-5>,
  "background_and_theme_relevance": <1-5>,
  "output_structure_and_reusability": <1-5>,
  "dimension_reasoning": {{
    "template_design_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_and_execution": "<one sentence citing specific evidence>",
    "background_and_theme_relevance": "<one sentence citing specific evidence>",
    "output_structure_and_reusability": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_design_accuracy": 0.35,
    "platform_usage_and_execution": 0.30,
    "background_and_theme_relevance": 0.20,
    "output_structure_and_reusability": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())