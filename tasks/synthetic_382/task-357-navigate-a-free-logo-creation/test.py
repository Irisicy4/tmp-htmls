"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Create a logo for a fictional brand 'EcoZen' using a free logo creation tool.
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


TASK_INSTRUCTION = """Navigate a free logo creation tool and create a logo for a fictional brand called 'EcoZen'. Choose the color green for the logo's theme, select an appropriate font, and include an icon that represents 'sustainability'. Report the final logo preview screen and download the design."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a logo for a fictional brand called 'EcoZen' using a free logo creation tool. The logo must have a green theme, an appropriate font, and an icon representing sustainability. The agent must report the final logo preview screen and download the design.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate a free logo creation tool and create a logo for a fictional brand called 'EcoZen'. Choose the color green for the logo's theme, select an appropriate font, and include an icon that represents 'sustainability'. Report the final logo preview screen and download the design.

## Task-Specific Constraints
- Must use at least one of the specified platforms: logomaker.com, canva.com, hatchful.shopify.com.
- The logo must prominently feature the color green.
- The font must be appropriate for a sustainability-focused brand.
- The icon must visually represent sustainability (e.g., leaves, trees, recycling symbols).
- The agent must report the final logo preview screen.
- The agent must download the logo design.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least one of the required platforms? Which ones were actually visited?
- Does the logo prominently feature the color green?
- Is the font appropriate for a sustainability-focused brand?
- Does the icon visually represent sustainability?
- Did the agent report the final logo preview screen and download the design?

### Step 2: Dimension Scoring

#### A. Logo Accuracy (0.35)
Measures whether the logo meets the task requirements for color, font, and icon.

5 — Logo prominently features green, uses an appropriate font, and includes a sustainability-themed icon.
4 — Logo meets most requirements but has minor issues (e.g., font slightly inappropriate).
3 — Logo partially meets requirements (e.g., missing one key element like color or icon).
2 — Logo is mostly incorrect or incomplete (e.g., missing multiple elements).
1 — Logo does not meet any requirements.

#### B. Platform Usage (0.30)
Measures whether the agent used the required platforms.

5 — Agent used at least one of the specified platforms and completed the task successfully.
4 — Agent used a specified platform but with minor issues in execution.
3 — Agent used a specified platform but did not fully complete the task.
2 — Agent attempted to use a platform but failed to execute the task.
1 — Agent did not use any specified platform.

#### C. Execution Specificity (0.20)
Measures the level of detail and specificity in the agent's response.

5 — Response includes detailed descriptions of the logo design process and choices.
4 — Response includes most details but lacks minor specifics.
3 — Response includes basic details but lacks depth.
2 — Response is vague and lacks significant details.
1 — Response provides no meaningful details.

#### D. Output Structure (0.15)
Measures the organization and clarity of the agent's response.

5 — Response is well-organized, clearly structured, and easy to follow.
4 — Response is mostly organized but has minor clarity issues.
3 — Response is partially organized but lacks some clarity.
2 — Response is poorly organized and hard to follow.
1 — Response is completely disorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "logo_accuracy": <1-5>,
  "platform_usage": <1-5>,
  "execution_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "logo_accuracy": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "execution_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "logo_accuracy": 0.35,
    "platform_usage": 0.30,
    "execution_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())