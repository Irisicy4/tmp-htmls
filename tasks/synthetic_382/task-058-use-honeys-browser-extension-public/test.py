"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Configure a price-drop alert for a PlayStation 5 console on Amazon and Walmart using Honey's browser extension.
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


TASK_INSTRUCTION = """Use Honey's browser extension (public site interface) to configure a price-drop alert for a PlayStation 5 console on Amazon and Walmart. Set the target price to $450 and specify email notifications for when the prices drop below the threshold."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task is in the Shopping domain and involves using Honey's browser extension to configure price-drop alerts for a PlayStation 5 console on Amazon and Walmart. A successful completion requires setting the target price to $450 and enabling email notifications for price drops below the threshold.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Honey's browser extension (public site interface) to configure a price-drop alert for a PlayStation 5 console on Amazon and Walmart. Set the target price to $450 and specify email notifications for when the prices drop below the threshold.

## Task-Specific Constraints
- Must navigate to both Amazon and Walmart platforms.
- Must use Honey's browser extension to configure the alerts.
- Target price must be set to exactly $450.
- Email notifications must be enabled for price drops below $450.
- Output must confirm the alert configuration for both platforms.
- Response must include evidence of platform navigation and alert setup.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon and Walmart as required?
- Did the agent use Honey's browser extension to configure the alerts?
- Was the target price set to $450 for both platforms?
- Were email notifications enabled for price drops below $450?
- Does the response confirm the alert setup for both platforms?

### Step 2: Dimension Scoring

#### A. Alert Configuration Accuracy (0.35)
Measures whether the agent correctly configured price-drop alerts with the specified parameters.

5 — Alerts configured correctly on both platforms with target price $450 and email notifications enabled.
4 — Alerts configured correctly on one platform; minor issues on the other.
3 — Partial configuration; missing one key parameter (e.g., target price or email notifications).
2 — Attempted configuration but mostly incorrect or incomplete.
1 — No alerts configured or completely wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent navigated to both Amazon and Walmart and performed actions on both.

5 — Successfully navigated and performed actions on both platforms.
4 — Navigated to both platforms but completed actions on only one.
3 — Navigated to only one platform but performed actions there.
2 — Attempted navigation but failed to perform actions.
1 — No platform navigation or actions performed.

#### C. Evidence Detail (0.20)
Measures the specificity and clarity of evidence provided in the response.

5 — Includes detailed evidence of navigation, alert setup, and confirmation for both platforms.
4 — Includes evidence for one platform and partial evidence for the other.
3 — Includes minimal evidence; lacks clarity or specificity.
2 — Evidence mostly absent or unclear.
1 — No evidence provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and credible.

5 — Response is well-structured, clear, and credible.
4 — Response is mostly clear but has minor structural issues.
3 — Response is usable but disorganized or lacks credibility.
2 — Response is poorly structured or mostly unclear.
1 — Response is completely disorganized or not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "alert_configuration_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "evidence_detail": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "alert_configuration_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "evidence_detail": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "alert_configuration_accuracy": 0.35,
    "platform_coverage": 0.30,
    "evidence_detail": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())