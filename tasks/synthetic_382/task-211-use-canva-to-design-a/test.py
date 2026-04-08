"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Design a social media post template for promoting a content creator's weekly schedule using Canva.
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


TASK_INSTRUCTION = """Use Canva to design a social media post template for promoting a content creator's weekly schedule. Include placeholders for the creator's profile picture, day-by-day activity descriptions, and links to their social media accounts. Save the design and provide a preview link."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves using Canva to design a social media post template for promoting a content creator's weekly schedule. The template must include placeholders for the creator's profile picture, day-by-day activity descriptions, and links to their social media accounts. A successful completion requires saving the design and providing a preview link.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Canva to design a social media post template for promoting a content creator's weekly schedule. Include placeholders for the creator's profile picture, day-by-day activity descriptions, and links to their social media accounts. Save the design and provide a preview link.

## Task-Specific Constraints
- Must use Canva to create the design.
- The design must include placeholders for the creator's profile picture.
- The design must include placeholders for day-by-day activity descriptions.
- The design must include placeholders for links to social media accounts.
- The agent must save the design and provide a valid preview link.
- The preview link must be functional and accessible.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Canva to create the design?
- Does the design include placeholders for the creator's profile picture?
- Does the design include placeholders for day-by-day activity descriptions?
- Does the design include placeholders for links to social media accounts?
- Did the agent save the design and provide a valid preview link? Is the link functional?

### Step 2: Dimension Scoring

#### A. Template Completeness (0.35)
Measures whether the design includes all required placeholders and elements.

5 — Includes all placeholders (profile picture, day-by-day activities, social media links) and is visually coherent.
4 — Includes most placeholders but may miss minor elements or have slight visual issues.
3 — Includes some placeholders but is incomplete or visually disorganized.
2 — Includes few placeholders and is mostly incomplete.
1 — No placeholders or design provided.

#### B. Platform Usage Accuracy (0.30)
Measures whether Canva was used correctly and the design was saved appropriately.

5 — Canva was used correctly, and the design was saved with a valid preview link.
4 — Canva was used correctly, but the preview link has minor issues.
3 — Canva was used, but the design or link is incomplete or partially incorrect.
2 — Canva usage was attempted but largely incorrect or incomplete.
1 — Canva was not used or the design was not saved.

#### C. Placeholder Specificity (0.20)
Measures the clarity and usability of placeholders for profile picture, activities, and social media links.

5 — Placeholders are clear, labeled, and easy to use.
4 — Placeholders are mostly clear but lack minor labeling or usability features.
3 — Placeholders are present but lack clarity or usability.
2 — Placeholders are vague or poorly implemented.
1 — Placeholders are absent or unusable.

#### D. Output Accessibility (0.15)
Measures whether the preview link is functional and accessible.

5 — Preview link is fully functional and accessible.
4 — Preview link is functional but has minor accessibility issues.
3 — Preview link is partially functional or difficult to access.
2 — Preview link is mostly non-functional or inaccessible.
1 — No preview link provided or completely inaccessible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_completeness": <1-5>,
  "platform_usage_accuracy": <1-5>,
  "placeholder_specificity": <1-5>,
  "output_accessibility": <1-5>,
  "dimension_reasoning": {{
    "template_completeness": "<one sentence citing specific evidence>",
    "platform_usage_accuracy": "<one sentence citing specific evidence>",
    "placeholder_specificity": "<one sentence citing specific evidence>",
    "output_accessibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_completeness": 0.35,
    "platform_usage_accuracy": 0.30,
    "placeholder_specificity": 0.20,
    "output_accessibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())