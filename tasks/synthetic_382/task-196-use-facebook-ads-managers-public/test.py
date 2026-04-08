"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Create a sample Facebook Ads campaign targeting people aged 18-34 interested in AI and technology in the United States, with a daily budget of $20.
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


TASK_INSTRUCTION = """Use Facebook Ads Manager’s public demo tool to create a sample campaign targeting people aged 18-34 interested in AI and technology in the United States. Choose the demographics, interest-based targeting, and set a daily budget of $20. Report back with the final screen showing the campaign setup summary."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Facebook Ads Manager’s public demo tool to create a sample campaign targeting people aged 18-34 interested in AI and technology in the United States. The campaign must include demographics, interest-based targeting, and a daily budget of $20. A successful completion includes reporting back with the final screen showing the campaign setup summary.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Facebook Ads Manager’s public demo tool to create a sample campaign targeting people aged 18-34 interested in AI and technology in the United States. Choose the demographics, interest-based targeting, and set a daily budget of $20. Report back with the final screen showing the campaign setup summary.

## Task-Specific Constraints
- Must use Facebook Ads Manager’s public demo tool.
- Must select demographics targeting people aged 18-34.
- Must select interest-based targeting for AI and technology.
- Must set a daily budget of $20.
- Must report back with the final screen showing the campaign setup summary.
- Must include evidence of platform navigation in the execution trace.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Facebook Ads Manager’s public demo tool?
- Did the agent select demographics targeting people aged 18-34?
- Did the agent include interest-based targeting for AI and technology?
- Did the agent set the daily budget to $20?
- Did the agent provide the final campaign setup summary screen?

### Step 2: Dimension Scoring

#### A. Campaign Setup Accuracy (0.35)
Measures whether the campaign setup matches the task requirements.

5 — All required elements (demographics, interests, budget, summary screen) are correct and complete.
4 — One minor error or omission in the setup.
3 — Partial completion with multiple errors or missing elements.
2 — Significant errors or omissions in the setup.
1 — No meaningful attempt at the campaign setup.

#### B. Platform Navigation Coverage (0.30)
Measures whether the agent navigated to the correct platforms and tools.

5 — Successfully navigated Facebook Ads Manager and included evidence of tool usage.
4 — Navigated Facebook Ads Manager but evidence is incomplete.
3 — Partial navigation with missing evidence or skipped steps.
2 — Minimal navigation or incorrect platform usage.
1 — No navigation or completely wrong platforms.

#### C. Targeting Specificity (0.20)
Measures the accuracy and specificity of targeting parameters.

5 — Demographics and interests are highly specific and match the task requirements perfectly.
4 — Demographics and interests are mostly correct but slightly vague or incomplete.
3 — Partial targeting with noticeable errors or omissions.
2 — Minimal targeting or mostly incorrect parameters.
1 — No targeting or completely wrong parameters.

#### D. Output Organization and Clarity (0.15)
Measures the structure and clarity of the agent’s final response.

5 — Final response is well-organized, clear, and includes the required summary screen.
4 — Response is mostly clear but slightly disorganized or missing minor details.
3 — Response is partially clear but lacks organization or key details.
2 — Response is poorly organized or missing major elements.
1 — Response is completely unclear or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "campaign_setup_accuracy": <1-5>,
  "platform_navigation_coverage": <1-5>,
  "targeting_specificity": <1-5>,
  "output_organization_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "campaign_setup_accuracy": "<one sentence citing specific evidence>",
    "platform_navigation_coverage": "<one sentence citing specific evidence>",
    "targeting_specificity": "<one sentence citing specific evidence>",
    "output_organization_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "campaign_setup_accuracy": 0.35,
    "platform_navigation_coverage": 0.30,
    "targeting_specificity": 0.20,
    "output_organization_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())