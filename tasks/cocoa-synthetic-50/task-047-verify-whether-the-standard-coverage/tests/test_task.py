"""
LLM-as-judge evaluator for EvolveBench task.

Category: Insurance & Actuarial
Task: Verify whether flood insurance policies include reimbursement for temporary housing expenses during home repairs.
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


TASK_INSTRUCTION = """Verify whether the standard coverage for flood insurance includes reimbursement for temporary housing expenses during home repairs. Visit the FEMA National Flood Insurance Program (https://www.floodsmart.gov/), Progressive (https://www.progressive.com/insurance/flood/), and Allstate (https://www.allstate.com/flood-insurance.aspx) to review their policy details. Check each source for any mention of temporary housing coverage and record whether it is explicitly included or excluded. Produce a verification report summarizing whether this coverage is available and provide evidence from the reviewed sources."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify whether flood insurance policies include reimbursement for temporary housing expenses during home repairs. The domain is Insurance & Actuarial, and successful completion requires reviewing policy details from three specified platforms and producing a summary report with evidence.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the standard coverage for flood insurance includes reimbursement for temporary housing expenses during home repairs. Visit the FEMA National Flood Insurance Program (https://www.floodsmart.gov/), Progressive (https://www.progressive.com/insurance/flood/), and Allstate (https://www.allstate.com/flood-insurance.aspx) to review their policy details. Check each source for any mention of temporary housing coverage and record whether it is explicitly included or excluded. Produce a verification report summarizing whether this coverage is available and provide evidence from the reviewed sources.

## Task-Specific Constraints
- Must visit all three specified platforms: floodsmart.gov, progressive.com, and allstate.com.
- Must explicitly state whether temporary housing coverage is included or excluded for each platform.
- Must provide direct evidence (e.g., quotes or screenshots) from the reviewed sources.
- Must summarize findings in a structured format (e.g., table or bullet points).
- Must ensure factual accuracy in claims about policy coverage.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response explicitly state whether temporary housing coverage is included or excluded for each platform?
- Are direct quotes or evidence from the reviewed sources included in the response?
- Is the output organized in a structured format (e.g., table or bullet points)?
- Are all factual claims about policy coverage accurate and sourced?

### Step 2: Dimension Scoring

#### A. Coverage Verification Accuracy (0.35)
Measures whether the agent correctly identified whether temporary housing coverage is included or excluded.

5 — Correctly identifies inclusion/exclusion for all three platforms with evidence.
4 — Correctly identifies inclusion/exclusion for two platforms with evidence.
3 — Correctly identifies inclusion/exclusion for one platform with evidence.
2 — Incorrect or missing identification for most platforms.
1 — No attempt made to verify coverage.

#### B. Platform Coverage Completeness (0.30)
Measures whether the agent visited and reviewed all required platforms.

5 — All three platforms reviewed and cited in the response.
4 — Two platforms reviewed and cited in the response.
3 — One platform reviewed and cited in the response.
2 — No platforms reviewed, or only partial navigation.
1 — No attempt made to visit platforms.

#### C. Evidence Specificity (0.20)
Measures whether the agent provided specific evidence (e.g., quotes or screenshots) to support claims.

5 — Provides specific evidence for all three platforms.
4 — Provides specific evidence for two platforms.
3 — Provides specific evidence for one platform.
2 — Evidence is vague or missing for most platforms.
1 — No evidence provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and uses credible sources.

5 — Response is structured (e.g., table or bullet points) and sources are credible.
4 — Response is mostly structured and sources are credible.
3 — Response is partially structured or sources are somewhat credible.
2 — Response is disorganized or sources lack credibility.
1 — No structure or credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent reviewed all three platforms and provided evidence for coverage inclusion/exclusion. Some claims lacked specificity or were partially incomplete.",
  "coverage_verification_accuracy": 4,
  "platform_coverage_completeness": 5,
  "evidence_specificity": 3,
  "output_structure_and_credibility": 4,
  "dimension_reasoning": {
    "coverage_verification_accuracy": "Two platforms were correctly analyzed with evidence, but one lacked clarity.",
    "platform_coverage_completeness": "All three platforms were reviewed and cited.",
    "evidence_specificity": "Specific evidence was provided for two platforms but missing for one.",
    "output_structure_and_credibility": "The response was structured and sources were credible."
  },
  "overall_score": 4.05,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "coverage_verification_accuracy": 0.35,
    "platform_coverage_completeness": 0.30,
    "evidence_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())