"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Verify if the Dell XPS 13 laptop has consistent specifications across Amazon.com, Dell.com, and BestBuy.com.
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


TASK_INSTRUCTION = """Verify if a specific laptop model (Dell XPS 13) listed on Amazon.com, Dell.com, and BestBuy.com has the same specifications across all platforms. Check CPU model, RAM size, and storage capacity. Report any inconsistencies."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying if the Dell XPS 13 laptop has consistent specifications across Amazon.com, Dell.com, and BestBuy.com. The agent must compare CPU model, RAM size, and storage capacity across these platforms and report any inconsistencies. A successful completion requires accurate data collection and a structured report highlighting any discrepancies.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify if a specific laptop model (Dell XPS 13) listed on Amazon.com, Dell.com, and BestBuy.com has the same specifications across all platforms. Check CPU model, RAM size, and storage capacity. Report any inconsistencies.

## Task-Specific Constraints
- Must visit Amazon.com, Dell.com, and BestBuy.com.
- Must include CPU model, RAM size, and storage capacity for the Dell XPS 13 from each platform.
- Output must be organized as a structured table or list for comparison.
- Must highlight any inconsistencies in specifications across platforms.
- Must provide evidence or references for the specifications collected.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon.com, Dell.com, and BestBuy.com? Which platforms were actually visited?
- Are CPU model, RAM size, and storage capacity included for the Dell XPS 13 from each platform?
- Is the output organized as a structured table or list for comparison?
- Are any inconsistencies in specifications highlighted in the response?
- Are the specifications sourced or referenced accurately?

### Step 2: Dimension Scoring

#### A. Specification Accuracy (0.35)
Measures whether the specifications provided for the Dell XPS 13 are correct and complete.

5 — Specifications for CPU, RAM, and storage are accurate and complete for all three platforms.
4 — Specifications are mostly accurate but contain minor errors or omissions.
3 — Specifications are partially accurate but missing key details for one or more platforms.
2 — Specifications are mostly incorrect or incomplete for multiple platforms.
1 — No specifications provided or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included data from each.

5 — Data from Amazon.com, Dell.com, and BestBuy.com is included.
4 — Data from two platforms is included, with minor omissions.
3 — Data from only one platform is included, or data is incomplete.
2 — Minimal data included, missing most platforms.
1 — No data from any platform included.

#### C. Inconsistency Identification (0.20)
Measures whether the agent correctly identified and reported inconsistencies in specifications.

5 — All inconsistencies are identified and clearly reported.
4 — Most inconsistencies are identified, with minor omissions.
3 — Some inconsistencies are identified, but others are missed.
2 — Few inconsistencies are identified, or reporting is unclear.
1 — No inconsistencies identified.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a clear table or list, with credible sourcing.
4 — Output is mostly structured, with minor formatting issues or unclear sourcing.
3 — Output is partially structured but lacks clarity or credible sourcing.
2 — Output is poorly structured and lacks credible sourcing.
1 — Output is unstructured and lacks any credible sourcing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "specification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "inconsistency_identification": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "specification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "inconsistency_identification": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "specification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "inconsistency_identification": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())