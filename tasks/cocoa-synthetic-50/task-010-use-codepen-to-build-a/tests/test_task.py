"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Build a functional drag-and-drop Kanban board snippet using CodePen, referencing MDN documentation and open-source examples.
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


TASK_INSTRUCTION = """Use CodePen to build a functional snippet that demonstrates a simple drag-and-drop interface for a Kanban board using JavaScript and CSS. Reference the official MDN documentation on drag-and-drop APIs and open-source examples for inspiration."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use CodePen to create a functional drag-and-drop Kanban board snippet. The snippet must utilize JavaScript and CSS, and the agent must reference the MDN documentation on drag-and-drop APIs as well as open-source examples for inspiration. A successful completion involves producing a working CodePen snippet with clear drag-and-drop functionality for Kanban board items.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use CodePen to build a functional snippet that demonstrates a simple drag-and-drop interface for a Kanban board using JavaScript and CSS. Reference the official MDN documentation on drag-and-drop APIs and open-source examples for inspiration.

## Task-Specific Constraints
- The agent must use CodePen to create the snippet.
- The snippet must include at least two Kanban columns and draggable items.
- The drag-and-drop functionality must work correctly (e.g., items can be moved between columns).
- The agent must reference MDN documentation on drag-and-drop APIs.
- The agent must provide a CodePen link to the final snippet.
- The snippet must include both JavaScript and CSS code.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use CodePen to create the snippet? Is a valid CodePen link provided?
- Does the snippet include at least two Kanban columns and draggable items?
- Does the drag-and-drop functionality work as intended?
- Did the agent reference MDN documentation or open-source examples in their response?
- Is the snippet implemented using both JavaScript and CSS?

### Step 2: Dimension Scoring

#### A. Functionality and Accuracy (0.35)
Measures whether the drag-and-drop Kanban board works as intended.

5 — The snippet is fully functional, with drag-and-drop working flawlessly across multiple columns.
4 — The snippet is functional but has minor issues (e.g., occasional bugs or limited functionality).
3 — The snippet is partially functional but incomplete (e.g., drag-and-drop works in only one direction).
2 — The snippet is mostly non-functional, with significant issues.
1 — The snippet does not work at all.

#### B. Coverage of Requirements (0.30)
Measures whether all task-specific constraints are satisfied.

5 — All constraints are fully satisfied (e.g., CodePen used, MDN referenced, JavaScript and CSS included).
4 — Most constraints are satisfied, with minor omissions.
3 — Some constraints are satisfied, but key elements are missing.
2 — Few constraints are satisfied.
1 — None of the constraints are satisfied.

#### C. Code Quality and Specificity (0.20)
Measures the quality and clarity of the code provided.

5 — Code is clean, well-documented, and follows best practices.
4 — Code is mostly clean but lacks some documentation or has minor issues.
3 — Code is functional but messy or poorly documented.
2 — Code is difficult to understand or poorly structured.
1 — Code is unreadable or absent.

#### D. Evidence and References (0.15)
Measures whether the agent provided credible references and evidence.

5 — Clear references to MDN documentation and/or open-source examples are provided.
4 — References are provided but lack specificity or clarity.
3 — Minimal references are provided.
2 — References are vague or irrelevant.
1 — No references are provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "functionality_and_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "code_quality_and_specificity": <1-5>,
  "evidence_and_references": <1-5>,
  "dimension_reasoning": {{
    "functionality_and_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "code_quality_and_specificity": "<one sentence citing specific evidence>",
    "evidence_and_references": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "functionality_and_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "code_quality_and_specificity": 0.20,
    "evidence_and_references": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())