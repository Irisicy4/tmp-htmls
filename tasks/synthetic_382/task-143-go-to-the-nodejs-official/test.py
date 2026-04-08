"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract breaking changes between Node.js versions 18.0.0 and 18.15.0 from the official changelog.
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


TASK_INSTRUCTION = """Go to the Node.js official changelog, apply filters to view updates between versions 18.0.0 and 18.15.0, and extract a list of all breaking changes. Include feature names and affected modules for each change."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate to the official Node.js changelog, filter updates between versions 18.0.0 and 18.15.0, and extract all breaking changes. The deliverable is a structured list containing feature names and affected modules for each breaking change.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the Node.js official changelog, apply filters to view updates between versions 18.0.0 and 18.15.0, and extract a list of all breaking changes. Include feature names and affected modules for each change.

## Task-Specific Constraints
- Must navigate to the official Node.js changelog page.
- Must apply filters to view updates between versions 18.0.0 and 18.15.0.
- Must extract all breaking changes listed in the changelog.
- Must include both feature names and affected modules for each breaking change.
- Output must be structured as a list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the official Node.js changelog page?
- Did the agent apply filters correctly to view updates between versions 18.0.0 and 18.15.0?
- Are all breaking changes between these versions included in the response?
- Does the response include both feature names and affected modules for each breaking change?
- Is the output structured as a list or table?

### Step 2: Dimension Scoring

#### A. Breaking Changes Accuracy (0.35)
Measures whether the agent correctly identified all breaking changes between versions 18.0.0 and 18.15.0.

5 — Identifies all breaking changes with correct feature names and affected modules.
4 — Identifies most breaking changes with minor omissions or inaccuracies.
3 — Identifies some breaking changes but with notable omissions or inaccuracies.
2 — Identifies few breaking changes and misses key details.
1 — Fails to identify any breaking changes.

#### B. Filtering Correctness (0.30)
Measures whether the agent applied filters correctly to view updates between the specified versions.

5 — Filters applied correctly and updates between 18.0.0 and 18.15.0 are accurately retrieved.
4 — Filters mostly applied correctly but with minor errors.
3 — Filters partially applied, resulting in incomplete or inaccurate updates.
2 — Filters applied incorrectly, resulting in mostly irrelevant updates.
1 — Filters not applied at all.

#### C. Detail Specificity (0.20)
Measures whether the agent included sufficient detail, such as feature names and affected modules.

5 — Includes detailed feature names and affected modules for all breaking changes.
4 — Includes most details but with minor omissions.
3 — Includes some details but lacks specificity for several breaking changes.
2 — Includes few details and lacks specificity.
1 — Includes no details.

#### D. Output Structure and Organization (0.15)
Measures whether the response is well-organized and presented in a structured format.

5 — Output is well-organized as a structured list or table.
4 — Output is mostly organized but with minor formatting issues.
3 — Output is partially organized but lacks clarity.
2 — Output is poorly organized and difficult to interpret.
1 — Output is unstructured or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "breaking_changes_accuracy": <1-5>,
  "filtering_correctness": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "breaking_changes_accuracy": "<one sentence citing specific evidence>",
    "filtering_correctness": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "breaking_changes_accuracy": 0.35,
    "filtering_correctness": 0.30,
    "detail_specificity": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())