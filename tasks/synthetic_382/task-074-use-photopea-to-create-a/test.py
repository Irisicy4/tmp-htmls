"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Create a reusable shopping bag mockup using Photopea, Pixabay, and Canva.
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


TASK_INSTRUCTION = """Use Photopea to create a product mockup for a reusable shopping bag. Start with a blank canvas and add a high-resolution image of the bag, then overlay a custom logo and text using publicly available assets from Pixabay and Canva templates."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a reusable shopping bag mockup. The agent must use Photopea to design the mockup, starting with a blank canvas, adding a high-resolution image of a bag, and overlaying a custom logo and text. The assets for the logo and text must be sourced from Pixabay and Canva.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Photopea to create a product mockup for a reusable shopping bag. Start with a blank canvas and add a high-resolution image of the bag, then overlay a custom logo and text using publicly available assets from Pixabay and Canva templates.

## Task-Specific Constraints
- Must use Photopea to create the mockup.
- Must source a high-resolution image of a bag from Pixabay.
- Must source a custom logo or text template from Canva.
- The final mockup must include both the logo and text overlayed on the bag.
- The response must describe the steps taken and the sources used for the assets.
- The output must be clear and structured.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Photopea, Pixabay, and Canva? Which platforms were actually used?
- Did the agent source a high-resolution image of a bag from Pixabay?
- Did the agent source a custom logo or text template from Canva?
- Does the final mockup include both the logo and text overlayed on the bag?
- Is the agent's response clear and structured, describing the steps taken?

### Step 2: Dimension Scoring

#### A. Mockup Accuracy (0.35)
Measures whether the final mockup meets the task requirements.

5 — The mockup includes a high-resolution bag image, a logo, and text, all correctly overlayed.
4 — The mockup includes most required elements but with minor issues (e.g., low resolution).
3 — The mockup is partially complete but missing key elements (e.g., no logo or text).
2 — The mockup is mostly incorrect or incomplete.
1 — No mockup was created.

#### B. Platform Usage (0.30)
Measures whether the agent used the required platforms (Photopea, Pixabay, Canva).

5 — All three platforms were used correctly.
4 — Two platforms were used correctly; one was partially used.
3 — At least one platform was used correctly.
2 — Platforms were used incorrectly or not at all.
1 — No platforms were used.

#### C. Asset Sourcing (0.20)
Measures whether the agent sourced appropriate assets (bag image, logo, text).

5 — All assets were sourced appropriately and described clearly.
4 — Most assets were sourced appropriately, with minor omissions.
3 — Some assets were sourced, but key elements are missing.
2 — Few assets were sourced, and most are missing.
1 — No assets were sourced.

#### D. Response Clarity (0.15)
Measures the clarity and structure of the agent's response.

5 — The response is clear, well-structured, and describes all steps taken.
4 — The response is mostly clear but has minor structural issues.
3 — The response is somewhat clear but lacks detail or structure.
2 — The response is unclear or poorly structured.
1 — The response is incomprehensible or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "mockup_accuracy": <1-5>,
  "platform_usage": <1-5>,
  "asset_sourcing": <1-5>,
  "response_clarity": <1-5>,
  "dimension_reasoning": {{
    "mockup_accuracy": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "asset_sourcing": "<one sentence citing specific evidence>",
    "response_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "mockup_accuracy": 0.35,
    "platform_usage": 0.30,
    "asset_sourcing": 0.20,
    "response_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())