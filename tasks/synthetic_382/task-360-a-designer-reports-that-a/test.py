"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Investigate performance issues in a Figma file and suggest optimizations.
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


TASK_INSTRUCTION = """A designer reports that a Figma file with multiple nested components is causing significant lag during editing. Investigate the issue by reviewing Figma's performance documentation and community forums for similar cases. Identify the root cause and suggest optimizations."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating performance issues in a Figma file with nested components. The agent must review Figma's performance documentation and community forums to identify the root cause and suggest optimizations. This task is in the domain of design and requires actionable recommendations based on credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A designer reports that a Figma file with multiple nested components is causing significant lag during editing. Investigate the issue by reviewing Figma's performance documentation and community forums for similar cases. Identify the root cause and suggest optimizations.

## Task-Specific Constraints
- Must visit figma.com, forum.figma.com, and uxdesign.cc during investigation.
- Must identify at least one root cause of performance lag based on credible sources.
- Must suggest at least two actionable optimizations for improving performance.
- Output must be structured as a clear list or table.
- Recommendations must cite specific evidence or examples from the sources used.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to figma.com, forum.figma.com, and uxdesign.cc? Which ones were actually visited?
- Did the agent identify at least one root cause of performance lag? Is it credible?
- Are at least two actionable optimizations suggested? Are they specific and feasible?
- Is the output organized as a clear list or table?
- Are the recommendations supported by evidence or examples from the sources used?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identifies the root cause of performance lag.

5 — Identifies 2 or more root causes with credible evidence from sources.
4 — Identifies 1 root cause with credible evidence from sources.
3 — Identifies 1 root cause but lacks credible evidence or specificity.
2 — Identifies a vague or incorrect cause.
1 — Fails to identify any root cause.

#### B. Optimization Suggestions Completeness (0.30)
Measures whether the agent provides actionable and specific optimizations.

5 — Provides 3 or more actionable optimizations with supporting evidence.
4 — Provides 2 actionable optimizations with supporting evidence.
3 — Provides 2 actionable optimizations but lacks evidence or specificity.
2 — Provides 1 vague or incomplete optimization.
1 — Fails to provide any optimizations.

#### C. Platform Coverage (0.20)
Measures whether the agent visits all required platforms and uses them effectively.

5 — Uses all 3 specified platforms and extracts relevant information from each.
4 — Uses 2 specified platforms and extracts relevant information.
3 — Uses 2 specified platforms but extracts limited or irrelevant information.
2 — Uses 1 specified platform or extracts irrelevant information.
1 — Fails to use any specified platform.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and supported by credible sources.

5 — Output is structured as a clear list or table with credible citations.
4 — Output is structured but lacks some citations or clarity.
3 — Output is partially structured but lacks clarity or citations.
2 — Output is disorganized or lacks credibility.
1 — Output is absent or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "optimization_suggestions_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "optimization_suggestions_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification_accuracy": 0.35,
    "optimization_suggestions_completeness": 0.30,
    "platform_coverage": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())