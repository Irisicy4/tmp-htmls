"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Research and compare monetization terms for creators on TikTok, YouTube, and Patreon.
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


TASK_INSTRUCTION = """Research the monetization terms for creators on TikTok, YouTube, and Patreon. Compare their payout rates, eligibility requirements, and revenue share percentages, and create a summary table with key metrics such as minimum follower thresholds, payout frequency, and types of supported income streams."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research monetization terms for creators on TikTok, YouTube, and Patreon. The agent must compare payout rates, eligibility requirements, and revenue share percentages, and produce a summary table containing key metrics such as minimum follower thresholds, payout frequency, and types of supported income streams.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research the monetization terms for creators on TikTok, YouTube, and Patreon. Compare their payout rates, eligibility requirements, and revenue share percentages, and create a summary table with key metrics such as minimum follower thresholds, payout frequency, and types of supported income streams.

## Task-Specific Constraints
- Must visit TikTok, YouTube, and Patreon websites.
- Must include payout rates, eligibility requirements, and revenue share percentages for all three platforms.
- Output must be organized as a summary table with clear metrics.
- Must include minimum follower thresholds, payout frequency, and types of supported income streams.
- Must provide accurate and sourced data for comparisons.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TikTok, YouTube, and Patreon websites? Which ones were actually visited?
- Are payout rates, eligibility requirements, and revenue share percentages present for all three platforms?
- Is the output organized as a summary table with clear metrics?
- Are minimum follower thresholds, payout frequency, and types of supported income streams included?
- Are the claims made in the response accurate and sourced?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (summary table) is correct and complete.

5 — Includes all required metrics for all three platforms with accurate data.
4 — Includes most required metrics for all three platforms; minor inaccuracies.
3 — Includes some metrics; incomplete or partially inaccurate.
2 — Includes few metrics; mostly incorrect or missing.
1 — No usable metrics; completely absent or wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent researched all required platforms and included their data.

5 — Covers TikTok, YouTube, and Patreon comprehensively.
4 — Covers TikTok, YouTube, and Patreon but with minor omissions.
3 — Covers at least two platforms with partial data.
2 — Covers only one platform or mostly incomplete data.
1 — No platform data included.

#### C. Depth and Specificity (0.20)
Measures the level of detail in the comparisons and metrics provided.

5 — Provides detailed comparisons with specific numbers and sourced data.
4 — Provides comparisons with numbers but lacks some sourcing or detail.
3 — Provides basic comparisons; lacks depth or specificity.
2 — Provides minimal comparisons; vague or generic.
1 — No meaningful comparisons provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and uses credible sources.

5 — Output is well-organized as a table and uses credible sources.
4 — Output is mostly organized; minor issues with sourcing or format.
3 — Output is somewhat organized; lacks clarity or credible sources.
2 — Output is poorly organized or missing sourcing.
1 — Output is disorganized and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())