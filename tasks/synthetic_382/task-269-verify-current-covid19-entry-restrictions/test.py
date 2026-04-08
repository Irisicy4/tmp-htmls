"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Verify current COVID-19 entry restrictions for travelers arriving in Australia using specified platforms.
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


TASK_INSTRUCTION = """Verify current COVID-19 entry restrictions for travelers arriving in Australia as of this week. Check official Australian government sites, IATA Travel Center, and an airline advisory page to confirm whether testing or quarantine requirements are in place."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires verifying current COVID-19 entry restrictions for travelers arriving in Australia. The agent must consult official Australian government sites, the IATA Travel Center, and an airline advisory page to confirm whether testing or quarantine requirements are in place. A successful completion includes accurate and complete information sourced from the specified platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify current COVID-19 entry restrictions for travelers arriving in Australia as of this week. Check official Australian government sites, IATA Travel Center, and an airline advisory page to confirm whether testing or quarantine requirements are in place.

## Task-Specific Constraints
- Must visit health.gov.au, iatatravelcentre.com, and qantas.com.
- Must accurately report testing and quarantine requirements for travelers arriving in Australia.
- Must include information sourced from each platform visited.
- Output must be organized as a structured list or table.
- Must specify the date or week of the information retrieved.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to health.gov.au, iatatravelcentre.com, and qantas.com? Which ones were actually visited?
- Are testing and quarantine requirements clearly stated in the response?
- Is the output organized as a structured list or table?
- Is the information sourced from each platform accurate and complete?
- Does the response specify the date or week of the information retrieved?

### Step 2: Dimension Scoring

#### A. Accuracy of Restrictions Information (0.35)
Measures whether the agent correctly identified and reported testing and quarantine requirements.

5 — All restrictions are correctly identified and reported with no errors.
4 — Most restrictions are correctly identified, with minor omissions or inaccuracies.
3 — Some restrictions are identified, but the response is incomplete or partially inaccurate.
2 — Few restrictions are identified, with significant errors or omissions.
1 — No restrictions are identified or the information is entirely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and sourced information from them.

5 — Information is sourced from all three platforms (health.gov.au, iatatravelcentre.com, qantas.com).
4 — Information is sourced from two platforms, with minor omissions.
3 — Information is sourced from one platform, or partially sourced from multiple platforms.
2 — Minimal platform coverage, with significant omissions.
1 — No platform coverage or no information sourced.

#### C. Depth of Information (0.25)
Measures the level of detail in the response, including dates, specifics of restrictions, and structured output.

5 — Response includes detailed restrictions, dates, and is well-structured.
4 — Response includes most details, with minor omissions or lack of structure.
3 — Response includes basic details, but lacks depth or structure.
2 — Response is shallow, with significant missing details.
1 — Response lacks meaningful detail or structure.

#### D. Source Credibility and Output Organization (0.10)
Measures whether the sources used are credible and the output is well-organized.

5 — Sources are credible, and the output is highly organized and professional.
4 — Sources are credible, but the output has minor organizational issues.
3 — Sources are mostly credible, but the output lacks clarity or structure.
2 — Sources are questionable, and the output is poorly organized.
1 — Sources are not credible, and the output is disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "accuracy_of_restrictions_information": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_information": <1-5>,
  "source_credibility_and_output_organization": <1-5>,
  "dimension_reasoning": {{
    "accuracy_of_restrictions_information": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "source_credibility_and_output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "accuracy_of_restrictions_information": 0.35,
    "platform_coverage": 0.30,
    "depth_of_information": 0.25,
    "source_credibility_and_output_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())