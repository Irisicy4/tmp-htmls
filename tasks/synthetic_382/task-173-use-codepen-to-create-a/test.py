"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Configure a CodePen project with a CSS Grid layout and provide the link.
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


TASK_INSTRUCTION = """Use CodePen to create a new pen for experimenting with CSS Grid layouts. Configure the pen with a basic HTML structure containing a flexible grid layout, and include three example grid items with custom styles. Provide the link to the configured pen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves using CodePen to create a new pen that demonstrates a CSS Grid layout. The pen must include a basic HTML structure with a flexible grid layout and three example grid items styled with custom CSS. A successful completion requires providing a valid link to the configured pen.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use CodePen to create a new pen for experimenting with CSS Grid layouts. Configure the pen with a basic HTML structure containing a flexible grid layout, and include three example grid items with custom styles. Provide the link to the configured pen.

## Task-Specific Constraints
- The agent must use CodePen to create the pen.
- The HTML structure must include a parent container with a CSS Grid layout.
- The grid must have at least three items styled with custom CSS.
- The agent must provide a valid link to the configured pen.
- The pen must be publicly accessible and functional.
- The CSS must demonstrate flexibility (e.g., responsive behavior or dynamic sizing).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use CodePen to create the pen?
- Does the HTML structure include a parent container with a CSS Grid layout?
- Are there at least three grid items with custom CSS styles?
- Is the provided link valid and does it lead to a publicly accessible pen?
- Does the CSS demonstrate flexibility (e.g., responsive behavior or dynamic sizing)?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent successfully created the required CSS Grid layout and provided a valid link.

5 — The pen contains a valid CSS Grid layout, three styled grid items, and the link is functional.
4 — The pen contains a valid CSS Grid layout and three styled grid items, but the link has minor issues.
3 — The pen contains a CSS Grid layout but is missing key elements or the link is partially functional.
2 — The pen is incomplete or the link is invalid.
1 — No pen was created or the link is absent.

#### B. Coverage of Requirements (0.30)
Measures whether all task-specific constraints were met.

5 — All constraints are fully satisfied (CodePen used, valid HTML structure, custom CSS, etc.).
4 — Most constraints are satisfied, with minor omissions.
3 — Some constraints are satisfied, but key elements are missing.
2 — Few constraints are satisfied.
1 — None of the constraints are satisfied.

#### C. CSS Flexibility Demonstration (0.20)
Measures whether the CSS demonstrates flexibility (e.g., responsiveness or dynamic sizing).

5 — CSS demonstrates excellent flexibility with responsive behavior and dynamic sizing.
4 — CSS demonstrates good flexibility with minor issues.
3 — CSS demonstrates basic flexibility but lacks advanced features.
2 — CSS demonstrates minimal flexibility.
1 — CSS flexibility is absent.

#### D. Output Structure and Accessibility (0.15)
Measures whether the pen is well-organized and publicly accessible.

5 — The pen is well-organized, visually clear, and fully accessible.
4 — The pen is organized and accessible with minor issues.
3 — The pen is accessible but poorly organized.
2 — The pen is disorganized or partially inaccessible.
1 — The pen is inaccessible or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "The agent created a pen on CodePen with a CSS Grid layout, provided a link, and included three styled grid items. The link is functional and the CSS demonstrates flexibility.",
  "deliverable_accuracy": 5,
  "coverage_of_requirements": 5,
  "css_flexibility_demonstration": 4,
  "output_structure_and_accessibility": 5,
  "dimension_reasoning": {{
    "deliverable_accuracy": "The pen contains a valid CSS Grid layout, three styled grid items, and the link is functional.",
    "coverage_of_requirements": "All task-specific constraints were satisfied.",
    "css_flexibility_demonstration": "The CSS demonstrates responsive behavior and dynamic sizing.",
    "output_structure_and_accessibility": "The pen is well-organized, visually clear, and fully accessible."
  }},
  "overall_score": 4.85,
  "passed": true
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "css_flexibility_demonstration": 0.20,
    "output_structure_and_accessibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())