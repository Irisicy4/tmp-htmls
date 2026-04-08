"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract and summarize breaking changes between specific versions of React and Vue.js from their changelogs.
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


TASK_INSTRUCTION = """Navigate the changelogs for React and Vue.js and extract all breaking changes introduced between their respective versions 17.0 and 18.0 for React, and 2.6 and 3.0 for Vue.js. Provide a summary of these changes categorized by feature or functionality."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate the changelogs for React and Vue.js and extract all breaking changes introduced between specific versions (React: 17.0 to 18.0, Vue.js: 2.6 to 3.0). The deliverable is a categorized summary of these breaking changes by feature or functionality. This task is in the domain of Software Engineering.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate the changelogs for React and Vue.js and extract all breaking changes introduced between their respective versions 17.0 and 18.0 for React, and 2.6 and 3.0 for Vue.js. Provide a summary of these changes categorized by feature or functionality.

## Task-Specific Constraints
- Must visit the changelogs on github.com/facebook/react and github.com/vuejs/vue.
- Must use reactjs.org for additional context on React breaking changes.
- Must extract breaking changes for both React and Vue.js.
- Output must categorize breaking changes by feature or functionality.
- Must provide accurate version ranges for each breaking change.
- Must ensure the summary is clear and well-structured.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms (github.com/facebook/react, github.com/vuejs/vue, reactjs.org)?
- Are breaking changes for both React and Vue.js included in the response?
- Are the breaking changes categorized by feature or functionality?
- Are the version ranges for each breaking change accurate and clearly stated?
- Is the summary clear, structured, and free of major errors?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the breaking changes extracted are correct and complete.

5 — All breaking changes are accurate, complete, and correctly categorized.
4 — Most breaking changes are accurate and categorized, with minor omissions.
3 — Some breaking changes are present but incomplete or partially incorrect.
2 — Few breaking changes are present, with major inaccuracies or omissions.
1 — No breaking changes are correctly extracted.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and extracted data from them.

5 — All specified platforms were visited and data extracted from each.
4 — Most platforms were visited, but one may have been missed or underutilized.
3 — At least one platform was visited, but others were missed.
2 — Minimal platform usage; most were ignored.
1 — No evidence of platform usage.

#### C. Depth of Categorization (0.25)
Measures the depth and specificity of categorization of breaking changes.

5 — Breaking changes are deeply categorized by feature or functionality with detailed descriptions.
4 — Breaking changes are categorized with reasonable depth, but some details are missing.
3 — Categorization is present but shallow or incomplete.
2 — Minimal categorization; most changes are uncategorized.
1 — No categorization provided.

#### D. Output Structure and Clarity (0.10)
Measures the organization and clarity of the summary provided.

5 — Summary is well-structured, clear, and free of errors.
4 — Summary is mostly clear and structured, with minor issues.
3 — Summary is usable but contains noticeable structural or clarity issues.
2 — Summary is poorly structured or unclear.
1 — Summary is disorganized or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_categorization": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_categorization": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_categorization": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())