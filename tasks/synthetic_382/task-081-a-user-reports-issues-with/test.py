"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Investigate and resolve payment failure issues for purchasing a refurbished laptop on eBay.
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


TASK_INSTRUCTION = """A user reports issues with purchasing a refurbished laptop on eBay where the payment keeps failing. Investigate the problem by checking eBay's help center, forums, and live updates, and determine the root cause (e.g., payment gateway issues, seller restrictions). Provide the recommended resolution and step-by-step guidance."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating and resolving payment failure issues for purchasing a refurbished laptop on eBay. The agent must identify the root cause of the issue by consulting eBay's help center, forums, and live updates, and provide a clear resolution with step-by-step guidance. A successful completion includes accurate identification of the issue, use of the required platforms, and a clear, actionable resolution.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports issues with purchasing a refurbished laptop on eBay where the payment keeps failing. Investigate the problem by checking eBay's help center, forums, and live updates, and determine the root cause (e.g., payment gateway issues, seller restrictions). Provide the recommended resolution and step-by-step guidance.

## Task-Specific Constraints
- Must visit eBay's help center, forums (community.ebay.com), and downdetector.com.
- Must identify the root cause of the payment failure.
- Must provide a clear, actionable resolution with step-by-step guidance.
- Must summarize findings from all three platforms in the response.
- Output must be well-organized and easy to follow.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to eBay's help center, forums, and downdetector.com? Which ones were actually visited?
- Did the agent identify the root cause of the payment failure?
- Does the response provide a clear, actionable resolution with step-by-step guidance?
- Are findings from all three platforms summarized in the response?
- Is the output well-organized and easy to follow?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identified the root cause of the payment failure.

5 — Correctly identifies the root cause with clear evidence from all three platforms.
4 — Identifies the root cause with evidence from at least two platforms.
3 — Identifies a plausible root cause but lacks sufficient evidence.
2 — Incorrect or vague identification of the root cause.
1 — No attempt to identify the root cause.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms and incorporated findings into the response.

5 — Uses all three platforms and incorporates findings from each.
4 — Uses at least two platforms and incorporates findings.
3 — Uses one platform or partially incorporates findings.
2 — Minimal use of platforms or findings.
1 — No use of required platforms.

#### C. Resolution Clarity and Specificity (0.25)
Measures the clarity and specificity of the resolution provided.

5 — Provides a clear, actionable resolution with step-by-step guidance.
4 — Provides a mostly clear resolution with minor omissions.
3 — Provides a resolution but lacks clarity or specificity.
2 — Provides a vague or incomplete resolution.
1 — No resolution provided.

#### D. Output Organization and Credibility (0.10)
Measures the organization and credibility of the response.

5 — Well-organized, easy to follow, and cites credible sources.
4 — Mostly organized and credible with minor issues.
3 — Somewhat organized but lacks credibility or clarity.
2 — Poorly organized or lacks credibility.
1 — Disorganized and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "resolution_clarity_and_specificity": <1-5>,
  "output_organization_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "resolution_clarity_and_specificity": "<one sentence citing specific evidence>",
    "output_organization_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "resolution_clarity_and_specificity": 0.25,
    "output_organization_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())