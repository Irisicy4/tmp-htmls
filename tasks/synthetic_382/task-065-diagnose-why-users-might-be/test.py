"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Diagnose delays in shipping for an Ikea desk by analyzing user reviews and Q&A sections across multiple platforms.
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


TASK_INSTRUCTION = """Diagnose why users might be reporting delays in the shipping of an Ikea desk (ID: 'LAGKAPTEN/ADILS') purchased on Ikea.com. Browse user reviews on Trustpilot.com, Reddit.com, and Ikea's Q&A section to identify the root cause, affected regions, and suggested solutions."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to investigate shipping delays for a specific Ikea desk by analyzing user reviews and Q&A sections across multiple platforms. The domain is e-commerce and customer feedback analysis. A successful completion involves identifying the root cause, affected regions, and suggested solutions, with evidence sourced from the required platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why users might be reporting delays in the shipping of an Ikea desk (ID: 'LAGKAPTEN/ADILS') purchased on Ikea.com. Browse user reviews on Trustpilot.com, Reddit.com, and Ikea's Q&A section to identify the root cause, affected regions, and suggested solutions.

## Task-Specific Constraints
- Must visit Ikea.com, Trustpilot.com, and Reddit.com.
- Must identify the root cause of delays with specific evidence.
- Must specify affected regions based on user feedback.
- Must provide suggested solutions based on credible sources.
- Output must be organized as a structured list or table.
- Must cite sources for all claims made.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Ikea.com, Trustpilot.com, and Reddit.com? Which platforms were actually visited?
- Does the response identify the root cause of shipping delays with specific evidence?
- Are affected regions clearly specified based on user feedback?
- Are suggested solutions provided, and are they credible?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent accurately identifies the root cause of shipping delays.

5 — Clearly identifies the root cause with specific evidence from at least 3 platforms.
4 — Identifies the root cause with evidence from 2 platforms.
3 — Identifies the root cause but lacks specificity or evidence.
2 — Attempts to identify the cause but is mostly incorrect or unsupported.
1 — Fails to identify the root cause.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and utilized their data.

5 — Uses data from all 3 platforms (Ikea.com, Trustpilot.com, Reddit.com).
4 — Uses data from 2 platforms.
3 — Uses data from 1 platform.
2 — Attempts to use data but fails to extract meaningful insights.
1 — Does not use any platform data.

#### C. Specificity of Findings (0.25)
Measures the depth and specificity of the agent's findings.

5 — Provides detailed findings, including affected regions and solutions, with specific evidence.
4 — Provides findings with some detail but lacks full specificity.
3 — Provides findings but lacks depth or specificity.
2 — Findings are vague or unsupported.
1 — Findings are absent or completely incorrect.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the agent's output.

5 — Output is well-organized, structured, and cites credible sources.
4 — Output is organized but lacks full credibility or structure.
3 — Output is partially organized but lacks clarity or citations.
2 — Output is disorganized or lacks credibility.
1 — Output is completely unstructured or unsupported.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification": <1-5>,
  "coverage_of_platforms": <1-5>,
  "specificity_of_findings": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "specificity_of_findings": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "coverage_of_platforms": 0.30,
    "specificity_of_findings": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())