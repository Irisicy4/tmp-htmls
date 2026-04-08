"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Diagnose the root cause of a package installation conflict and provide a workaround or fix.
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


TASK_INSTRUCTION = """Diagnose the root cause of a problem where installing 'requests==2.31' alongside 'certifi==2024.2' fails. Check PyPI package compatibility notes, GitHub issues, and Stack Overflow threads to find the affected version range and the workaround or fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to diagnose the root cause of a package installation conflict involving 'requests==2.31' and 'certifi==2024.2'. The agent must investigate compatibility notes on PyPI, relevant GitHub issues, and Stack Overflow threads to identify the affected version range and provide a workaround or fix. This is a Software Engineering task requiring technical accuracy and thorough investigation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose the root cause of a problem where installing 'requests==2.31' alongside 'certifi==2024.2' fails. Check PyPI package compatibility notes, GitHub issues, and Stack Overflow threads to find the affected version range and the workaround or fix.

## Task-Specific Constraints
- Must investigate at least one compatibility note on PyPI.
- Must check at least one relevant GitHub issue.
- Must review at least one Stack Overflow thread.
- Must identify the affected version range for both 'requests' and 'certifi'.
- Must provide a clear workaround or fix for the conflict.
- Output must be structured as a clear explanation with steps or code snippets.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify the affected version range for both 'requests' and 'certifi'?
- Did the agent provide a clear workaround or fix for the conflict?
- Is the output structured as a clear explanation with steps or code snippets?
- Are the claims made by the agent accurate and sourced?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the root cause, affected version range, and provided a valid workaround or fix.

5 — Identifies the exact version range for both packages and provides a valid workaround or fix with supporting evidence.
4 — Identifies the version range for both packages and provides a workaround or fix, but lacks supporting evidence.
3 — Identifies partial version range or provides an incomplete workaround or fix.
2 — Incorrect or incomplete diagnosis with no valid workaround or fix.
1 — No diagnosis or workaround provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent investigated all required platforms (PyPI, GitHub, Stack Overflow).

5 — Investigates all three platforms and cites relevant findings from each.
4 — Investigates at least two platforms and cites relevant findings.
3 — Investigates at least one platform and cites findings.
2 — Investigates platforms but cites no relevant findings.
1 — No investigation of required platforms.

#### C. Depth of Analysis (0.25)
Measures the level of detail in the agent's response, including technical specifics and clarity.

5 — Provides detailed technical analysis, including code snippets or step-by-step instructions.
4 — Provides clear analysis but lacks technical specifics or code snippets.
3 — Provides basic analysis with minimal technical details.
2 — Provides vague or unclear analysis.
1 — No meaningful analysis provided.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the response.

5 — Response is well-organized, credible, and includes properly formatted evidence.
4 — Response is organized and credible but lacks formatting.
3 — Response is somewhat organized but lacks credibility or formatting.
2 — Response is disorganized or lacks credibility.
1 — Response is completely disorganized and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())