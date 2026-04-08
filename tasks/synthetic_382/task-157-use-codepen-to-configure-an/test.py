"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Configure an interactive HTML/JavaScript code snippet for a search bar that dynamically filters a list of items based on user input, including basic styling and functionality.
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


TASK_INSTRUCTION = """Use CodePen to configure an interactive HTML/JavaScript code snippet for a search bar that dynamically filters a list of items based on user input. Include basic styling and functionality for the search field and list."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating an interactive HTML/JavaScript code snippet for a search bar that dynamically filters a list of items based on user input. The deliverable must include basic styling and functionality for the search field and list. The agent must use CodePen and may reference w3schools.com and developer.mozilla.org for guidance.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use CodePen to configure an interactive HTML/JavaScript code snippet for a search bar that dynamically filters a list of items based on user input. Include basic styling and functionality for the search field and list.

## Task-Specific Constraints
- Must use CodePen as the platform to develop the code snippet.
- Must reference w3schools.com and developer.mozilla.org for guidance or examples.
- The search bar must dynamically filter the list based on user input.
- The list must contain at least 5 items with meaningful labels.
- Basic styling must be applied to the search field and list (e.g., borders, colors, spacing).
- The functionality must work without errors when tested.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to CodePen to create the code snippet?
- Did the agent reference w3schools.com and developer.mozilla.org for guidance or examples?
- Does the search bar dynamically filter the list based on user input?
- Does the list contain at least 5 items with meaningful labels?
- Is basic styling applied to the search field and list?

### Step 2: Dimension Scoring

#### A. Functionality Accuracy (0.35)
Measures whether the search bar correctly filters the list based on user input.

5 — The search bar dynamically filters the list correctly and handles edge cases (e.g., empty input, special characters).
4 — The search bar filters the list correctly but may miss edge cases.
3 — The search bar filters the list but has noticeable errors or missing functionality.
2 — The search bar is mostly non-functional or incorrect.
1 — The search bar functionality is absent.

#### B. Platform and Reference Usage (0.30)
Measures whether the agent used the required platforms and references.

5 — The agent used CodePen and referenced both w3schools.com and developer.mozilla.org effectively.
4 — The agent used CodePen and referenced at least one of the required platforms.
3 — The agent used CodePen but did not reference the required platforms.
2 — The agent attempted to use CodePen but failed to complete the task.
1 — The agent did not use CodePen.

#### C. Styling Quality (0.20)
Measures the quality and completeness of the styling applied to the search field and list.

5 — Styling is complete, visually appealing, and enhances usability.
4 — Styling is complete but lacks polish or minor usability enhancements.
3 — Styling is present but minimal or inconsistent.
2 — Styling is mostly absent or poorly implemented.
1 — Styling is completely absent.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-structured and organized.

5 — The code is well-organized, readable, and includes comments for clarity.
4 — The code is organized but lacks comments or minor readability improvements.
3 — The code is functional but poorly organized or difficult to read.
2 — The code is mostly disorganized or incomplete.
1 — The code is completely disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "functionality_accuracy": <1-5>,
  "platform_and_reference_usage": <1-5>,
  "styling_quality": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "functionality_accuracy": "<one sentence citing specific evidence>",
    "platform_and_reference_usage": "<one sentence citing specific evidence>",
    "styling_quality": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "functionality_accuracy": 0.35,
    "platform_and_reference_usage": 0.30,
    "styling_quality": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())