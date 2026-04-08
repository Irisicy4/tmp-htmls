"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Create a responsive navigation bar with HTML, CSS, and JavaScript on CodePen.
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


TASK_INSTRUCTION = """Open CodePen and create a simple responsive navigation bar using HTML, CSS, and JavaScript. The navigation bar should include three links (Home, About, Contact) with a hover effect and a dropdown menu under 'About'."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a responsive navigation bar using HTML, CSS, and JavaScript on CodePen. The navigation bar must include three links (Home, About, Contact) with a hover effect and a dropdown menu under 'About'. A successful completion involves correctly implementing the required functionality and design elements.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open CodePen and create a simple responsive navigation bar using HTML, CSS, and JavaScript. The navigation bar should include three links (Home, About, Contact) with a hover effect and a dropdown menu under 'About'.

## Task-Specific Constraints
- Must include three links: Home, About, Contact.
- Must implement hover effects for the links.
- Must include a dropdown menu under the 'About' link.
- Must use responsive design principles (e.g., media queries).
- Must use CodePen for implementation.
- Output must be functional and visually correct.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use CodePen for implementation?
- Are the three required links (Home, About, Contact) present in the response?
- Is there a hover effect implemented for the links?
- Is there a functional dropdown menu under the 'About' link?
- Does the design use responsive principles (e.g., media queries)?

### Step 2: Dimension Scoring

#### A. Functionality and Accuracy (0.35)
Measures whether the navigation bar works as intended and includes all required features.

5 — All required features (links, hover effects, dropdown menu) are fully functional and correct.
4 — Most features are functional, but one minor issue exists.
3 — Some features are functional, but major issues exist.
2 — Few features are functional, with significant omissions.
1 — No features are functional or implemented.

#### B. Coverage of Requirements (0.30)
Measures whether all task requirements are addressed.

5 — All requirements (links, hover effects, dropdown menu, responsive design) are fully implemented.
4 — Most requirements are implemented, but one is missing or incomplete.
3 — Some requirements are implemented, but multiple are missing or incomplete.
2 — Few requirements are implemented.
1 — No requirements are implemented.

#### C. Responsive Design Quality (0.20)
Measures the quality and correctness of the responsive design implementation.

5 — Responsive design is fully implemented with appropriate media queries and layout adjustments.
4 — Responsive design is mostly implemented, but minor issues exist.
3 — Responsive design is partially implemented, with noticeable flaws.
2 — Responsive design is poorly implemented or mostly missing.
1 — No responsive design elements are present.

#### D. Code Quality and Organization (0.15)
Measures the readability, structure, and organization of the code.

5 — Code is well-organized, readable, and follows best practices.
4 — Code is mostly well-organized, with minor readability issues.
3 — Code is partially organized, with noticeable readability issues.
2 — Code is poorly organized and difficult to follow.
1 — Code is completely disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "functionality_and_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "responsive_design_quality": <1-5>,
  "code_quality_and_organization": <1-5>,
  "dimension_reasoning": {{
    "functionality_and_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "responsive_design_quality": "<one sentence citing specific evidence>",
    "code_quality_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "functionality_and_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "responsive_design_quality": 0.20,
    "code_quality_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())