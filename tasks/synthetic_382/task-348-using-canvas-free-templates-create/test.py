"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Create a mood board for an eco-friendly beverage brand using Canva, download it as a JPG, and describe its elements.
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
    aggregated["evidence_summary"] = valid[0].get("evidence_summary", "")
    aggregated["dimension_reasoning"] = valid[0].get("dimension_reasoning", {})
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


TASK_INSTRUCTION = """Using Canva's free templates, create a mood board for a new eco-friendly beverage brand. Include elements such as a green color palette, nature-inspired imagery, and an elegant font pairing. Download the mood board as a JPG file and describe its elements in a summary."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a mood board for a new eco-friendly beverage brand using Canva's free templates. The mood board must include a green color palette, nature-inspired imagery, and an elegant font pairing. The agent must download the mood board as a JPG file and provide a summary describing its elements.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Canva's free templates, create a mood board for a new eco-friendly beverage brand. Include elements such as a green color palette, nature-inspired imagery, and an elegant font pairing. Download the mood board as a JPG file and describe its elements in a summary.

## Task-Specific Constraints
- Must use Canva's free templates to create the mood board.
- The mood board must include a green color palette.
- The mood board must include nature-inspired imagery.
- The mood board must include an elegant font pairing.
- The final mood board must be downloaded as a JPG file.
- A summary describing the elements of the mood board must be provided.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Canva to create the mood board?
- Does the mood board include a green color palette?
- Does the mood board include nature-inspired imagery?
- Does the mood board include an elegant font pairing?
- Was the mood board downloaded as a JPG file?
- Does the agent's summary accurately describe the elements of the mood board?

### Step 2: Dimension Scoring

#### A. Mood Board Content Accuracy (0.35)
Measures whether the mood board includes all required elements (green color palette, nature-inspired imagery, elegant font pairing).

5 — Includes all required elements with excellent execution.
4 — Includes all required elements but with minor flaws.
3 — Includes most required elements but misses one or more.
2 — Includes some elements but misses key components.
1 — Does not include any required elements.

#### B. Summary Quality (0.30)
Measures the accuracy and detail of the agent's summary describing the mood board.

5 — Summary is detailed, accurate, and covers all elements.
4 — Summary is mostly accurate but lacks detail in some areas.
3 — Summary is partially accurate but misses key details.
2 — Summary is vague or mostly incorrect.
1 — No summary provided or completely incorrect.

#### C. Platform Usage and Execution (0.20)
Measures whether the agent used Canva and followed the required steps to create and download the mood board.

5 — Used Canva correctly and downloaded the mood board as a JPG.
4 — Used Canva but with minor execution issues.
3 — Used Canva but with significant execution issues.
2 — Attempted to use Canva but failed to complete the task.
1 — Did not use Canva or complete the task.

#### D. Creativity and Presentation (0.15)
Measures the creativity and overall presentation quality of the mood board.

5 — Mood board is highly creative and visually appealing.
4 — Mood board is creative but lacks polish in some areas.
3 — Mood board is acceptable but lacks creativity or polish.
2 — Mood board is poorly designed or unappealing.
1 — Mood board is absent or completely unappealing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "mood_board_content_accuracy": <1-5>,
  "summary_quality": <1-5>,
  "platform_usage_and_execution": <1-5>,
  "creativity_and_presentation": <1-5>,
  "dimension_reasoning": {{
    "mood_board_content_accuracy": "<one sentence citing specific evidence>",
    "summary_quality": "<one sentence citing specific evidence>",
    "platform_usage_and_execution": "<one sentence citing specific evidence>",
    "creativity_and_presentation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "mood_board_content_accuracy": 0.35,
    "summary_quality": 0.30,
    "platform_usage_and_execution": 0.20,
    "creativity_and_presentation": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())