"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Create a responsive todo list web component with add, edit, and delete functionality using HTML/CSS/JavaScript.
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


TASK_INSTRUCTION = """Use CodePen to set up a simple HTML/CSS/JavaScript web component for a dynamic todo list. Implement features for adding, editing, and deleting tasks, and ensure the layout uses a responsive design. Share the final CodePen URL and explain how the component works."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a responsive web component for a dynamic todo list using HTML, CSS, and JavaScript. The component must include functionality for adding, editing, and deleting tasks. A successful completion includes sharing a valid CodePen URL and providing a clear explanation of how the component works.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use CodePen to set up a simple HTML/CSS/JavaScript web component for a dynamic todo list. Implement features for adding, editing, and deleting tasks, and ensure the layout uses a responsive design. Share the final CodePen URL and explain how the component works.

## Task-Specific Constraints
- Must include functionality for adding, editing, and deleting tasks.
- Must use responsive design principles to ensure usability on different screen sizes.
- Must provide a valid CodePen URL with working code.
- Must explain the structure and functionality of the component clearly.
- Must use at least one external platform (developer.mozilla.org or w3schools.com) for reference during implementation.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms (developer.mozilla.org, w3schools.com)? Which ones were actually visited?
- Does the CodePen URL provided contain a working web component?
- Are the required features (add, edit, delete tasks) implemented and functional?
- Is the layout responsive and usable across different screen sizes?
- Is the explanation of the component's structure and functionality clear and accurate?

### Step 2: Dimension Scoring

#### A. Functionality Implementation (0.35)
Measures whether the todo list component includes all required features (add, edit, delete tasks).

5 — All three features (add, edit, delete) are implemented and fully functional.
4 — Two features are fully functional, and the third is partially implemented.
3 — At least one feature is fully functional, and others are partially implemented.
2 — Only one feature is partially implemented.
1 — No features are implemented.

#### B. Responsiveness (0.30)
Measures whether the layout adapts well to different screen sizes.

5 — Fully responsive design with no layout issues on any screen size.
4 — Mostly responsive design with minor layout issues on some screen sizes.
3 — Partially responsive design with noticeable layout issues.
2 — Poorly responsive design with significant layout issues.
1 — Not responsive at all.

#### C. CodePen Accuracy (0.20)
Measures whether the CodePen URL is valid and contains working code.

5 — CodePen URL is valid and contains fully functional code.
4 — CodePen URL is valid but contains partially functional code.
3 — CodePen URL is valid but contains incomplete or non-functional code.
2 — CodePen URL is invalid or leads to unrelated content.
1 — No CodePen URL provided.

#### D. Explanation Clarity (0.15)
Measures whether the explanation of the component's structure and functionality is clear and accurate.

5 — Explanation is clear, detailed, and accurate.
4 — Explanation is mostly clear and accurate but lacks some detail.
3 — Explanation is partially clear or contains minor inaccuracies.
2 — Explanation is unclear or contains significant inaccuracies.
1 — No explanation provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "functionality_implementation": <1-5>,
  "responsiveness": <1-5>,
  "codepen_accuracy": <1-5>,
  "explanation_clarity": <1-5>,
  "dimension_reasoning": {{
    "functionality_implementation": "<one sentence citing specific evidence>",
    "responsiveness": "<one sentence citing specific evidence>",
    "codepen_accuracy": "<one sentence citing specific evidence>",
    "explanation_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "functionality_implementation": 0.35,
    "responsiveness": 0.30,
    "codepen_accuracy": 0.20,
    "explanation_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())