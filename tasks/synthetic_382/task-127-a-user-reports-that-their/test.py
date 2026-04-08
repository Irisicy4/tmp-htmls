"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Diagnose the cause of a customs clearance delay for an international UPS package and provide resolution steps.
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


TASK_INSTRUCTION = """A user reports that their international package shipped via UPS is stuck in customs and shows a 'Clearance Delay' status. Diagnose the issue by reviewing the UPS tracking help page and CBP’s guidelines on restricted items. Provide the likely cause and resolution steps."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires diagnosing the cause of a customs clearance delay for an international UPS package by reviewing UPS tracking help pages and CBP guidelines on restricted items. A successful completion involves identifying the likely cause and providing actionable resolution steps based on credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports that their international package shipped via UPS is stuck in customs and shows a 'Clearance Delay' status. Diagnose the issue by reviewing the UPS tracking help page and CBP’s guidelines on restricted items. Provide the likely cause and resolution steps.

## Task-Specific Constraints
- Must visit both ups.com and cbp.gov platforms.
- Must identify the likely cause of the delay based on credible information from the platforms.
- Must provide actionable resolution steps that address the identified cause.
- Output must be structured as a clear list or table.
- Must reference specific guidelines or policies from CBP and UPS.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to ups.com and cbp.gov? Were both platforms used?
- Does the response identify the likely cause of the customs delay based on credible information?
- Are actionable resolution steps provided, and do they address the identified cause?
- Is the output structured as a clear list or table?
- Are specific guidelines or policies from CBP and UPS referenced in the response?

### Step 2: Dimension Scoring

#### A. Cause Identification Accuracy (0.35)
Measures whether the agent correctly identifies the likely cause of the customs delay.

5 — Identifies the cause with specific references to CBP and UPS policies.
4 — Identifies the cause but lacks specific references or minor inaccuracies.
3 — Provides a plausible cause but lacks credibility or specificity.
2 — Provides an incorrect or vague cause.
1 — Fails to identify any cause.

#### B. Resolution Steps Completeness (0.30)
Measures whether the agent provides actionable and complete resolution steps.

5 — Provides clear, actionable steps addressing the identified cause.
4 — Provides actionable steps but with minor omissions or ambiguities.
3 — Provides steps but lacks detail or partially addresses the cause.
2 — Provides vague or incomplete steps.
1 — Fails to provide any resolution steps.

#### C. Platform Usage Coverage (0.20)
Measures whether the agent uses both ups.com and cbp.gov effectively.

5 — Uses both platforms and extracts relevant information from each.
4 — Uses both platforms but misses minor relevant details.
3 — Uses one platform effectively but the other minimally or incorrectly.
2 — Uses only one platform with minimal relevance.
1 — Fails to use either platform effectively.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and references credible sources.

5 — Output is structured clearly and references credible sources from both platforms.
4 — Output is structured but lacks minor clarity or credibility.
3 — Output is usable but lacks organization or credible references.
2 — Output is poorly structured or lacks credibility.
1 — Output is disorganized and lacks credible references.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "cause_identification_accuracy": <1-5>,
  "resolution_steps_completeness": <1-5>,
  "platform_usage_coverage": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "cause_identification_accuracy": "<one sentence citing specific evidence>",
    "resolution_steps_completeness": "<one sentence citing specific evidence>",
    "platform_usage_coverage": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "cause_identification_accuracy": 0.35,
    "resolution_steps_completeness": 0.30,
    "platform_usage_coverage": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())