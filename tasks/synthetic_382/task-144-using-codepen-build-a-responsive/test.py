"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Build a responsive navigation bar template in HTML/CSS/JS with dropdown menus and mobile compatibility.
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


TASK_INSTRUCTION = """Using CodePen, build a responsive navigation bar template in HTML/CSS/JS that includes dropdown menus and mobile compatibility. Use W3Schools or MDN Web Docs for reference if needed."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a responsive navigation bar template using HTML, CSS, and JavaScript. The navigation bar must include dropdown menus, work on mobile devices, and demonstrate responsiveness. The agent may use CodePen for implementation and W3Schools or MDN Web Docs for reference.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using CodePen, build a responsive navigation bar template in HTML/CSS/JS that includes dropdown menus and mobile compatibility. Use W3Schools or MDN Web Docs for reference if needed.

## Task-Specific Constraints
- Must use CodePen for implementation.
- Must visit at least one of W3Schools or MDN Web Docs for reference.
- The navigation bar must include dropdown menus.
- The navigation bar must demonstrate mobile responsiveness.
- The output must include HTML, CSS, and JavaScript code.
- The final response must explain how the code meets the requirements.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use CodePen for implementation?
- Did the agent visit at least one of W3Schools or MDN Web Docs for reference?
- Does the navigation bar include dropdown menus?
- Does the navigation bar demonstrate mobile responsiveness?
- Is the output structured with HTML, CSS, and JavaScript code?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the navigation bar meets the functional requirements.

5 — Fully functional navigation bar with dropdown menus and mobile responsiveness demonstrated.
4 — Functional navigation bar with minor issues in dropdown menus or responsiveness.
3 — Navigation bar is partially functional but missing key features like dropdown menus or responsiveness.
2 — Navigation bar is mostly non-functional or missing major features.
1 — No navigation bar or completely incorrect implementation.

#### B. Platform Usage and Coverage (0.30)
Measures whether the agent used the required platforms and references appropriately.

5 — CodePen used for implementation and both W3Schools and MDN Web Docs visited for reference.
4 — CodePen used and at least one reference platform visited.
3 — CodePen used but no reference platforms visited.
2 — CodePen not used or unclear platform usage.
1 — No evidence of platform usage.

#### C. Code Quality and Specificity (0.25)
Measures the quality and specificity of the provided code.

5 — Code is well-structured, includes comments, and is highly specific to the requirements.
4 — Code is structured and mostly specific but lacks comments or minor details.
3 — Code is functional but lacks structure or specificity.
2 — Code is poorly written or mostly incorrect.
1 — No code provided or completely incorrect.

#### D. Explanation and Organization (0.10)
Measures the clarity and organization of the response.

5 — Response is clear, well-organized, and explains how the code meets requirements.
4 — Response is mostly clear and organized with minor omissions.
3 — Response is somewhat clear but lacks organization or explanation.
2 — Response is unclear or poorly organized.
1 — No explanation or completely disorganized response.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "platform_usage_and_coverage": <1-5>,
  "code_quality_and_specificity": <1-5>,
  "explanation_and_organization": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_and_coverage": "<one sentence citing specific evidence>",
    "code_quality_and_specificity": "<one sentence citing specific evidence>",
    "explanation_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "platform_usage_and_coverage": 0.30,
    "code_quality_and_specificity": 0.25,
    "explanation_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())