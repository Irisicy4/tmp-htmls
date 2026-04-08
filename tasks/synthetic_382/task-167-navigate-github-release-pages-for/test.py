"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract breaking changes from GitHub release pages for TensorFlow, PyTorch, and Scikit-learn.
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


TASK_INSTRUCTION = """Navigate GitHub release pages for three popular machine learning libraries (TensorFlow, PyTorch, and Scikit-learn) and extract breaking changes introduced between their last minor version updates (e.g., 2.x.x to 2.y.x). Ensure extracted details focus on API changes affecting backward compatibility."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to navigate GitHub release pages for TensorFlow, PyTorch, and Scikit-learn and extract breaking changes introduced between their last minor version updates. The task is in the domain of software engineering, specifically focused on API changes affecting backward compatibility. A successful completion requires the agent to provide accurate, structured, and complete information about breaking changes from all three libraries.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate GitHub release pages for three popular machine learning libraries (TensorFlow, PyTorch, and Scikit-learn) and extract breaking changes introduced between their last minor version updates (e.g., 2.x.x to 2.y.x). Ensure extracted details focus on API changes affecting backward compatibility.

## Task-Specific Constraints
- Must visit the release pages of TensorFlow, PyTorch, and Scikit-learn.
- Must extract breaking changes specifically related to API backward compatibility.
- Must include changes for the last minor version update for each library.
- Output must be structured as a list or table for clarity.
- Must provide library names and version numbers for each set of breaking changes.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the release pages of TensorFlow, PyTorch, and Scikit-learn? Which ones were actually visited?
- Did the agent extract breaking changes specifically related to API backward compatibility?
- Are the breaking changes organized in a structured format (e.g., list or table)?
- Are the library names and version numbers included for each set of breaking changes?
- Are the extracted breaking changes accurate and relevant to the task?

### Step 2: Dimension Scoring

#### A. Breaking Changes Accuracy (0.35)
Measures whether the extracted breaking changes are accurate and relevant to API backward compatibility.

5 — Extracts accurate breaking changes for all three libraries, with no errors or irrelevant details.
4 — Extracts mostly accurate breaking changes, with minor errors or omissions.
3 — Extracts partially accurate breaking changes, with some errors or missing details.
2 — Extracts mostly inaccurate or irrelevant breaking changes.
1 — Fails to extract any accurate breaking changes.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included breaking changes for each.

5 — Includes breaking changes for all three libraries (TensorFlow, PyTorch, Scikit-learn).
4 — Includes breaking changes for two libraries, with minor omissions.
3 — Includes breaking changes for at least one library, but misses others.
2 — Attempts to visit platforms but fails to extract breaking changes.
1 — Does not visit any required platforms or extract breaking changes.

#### C. Detail and Specificity (0.20)
Measures the level of detail and specificity in the extracted breaking changes.

5 — Provides detailed breaking changes, including specific APIs, methods, or parameters affected.
4 — Provides mostly detailed breaking changes, with minor omissions.
3 — Provides some details, but lacks specificity in many cases.
2 — Provides vague or generic descriptions of breaking changes.
1 — Provides no meaningful details.

#### D. Output Organization (0.15)
Measures the clarity and structure of the output.

5 — Output is well-organized in a clear list or table format, with library names and version numbers.
4 — Output is mostly clear, with minor formatting issues.
3 — Output is partially clear, but lacks consistent structure.
2 — Output is poorly organized and difficult to follow.
1 — Output is unstructured or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "breaking_changes_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "breaking_changes_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "breaking_changes_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())