"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Calculate average engagement rate for 'tech review' YouTube creators and project earnings for a creator with 1M subscribers.
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


TASK_INSTRUCTION = """Using data from Crowdtangle, Social Blade, and Hootsuite’s free tools, find the average engagement rate for creators in the 'tech review' niche on YouTube. Calculate the projected earnings for a creator with 1 million subscribers, assuming YouTube’s typical CPM range of $4-10. Provide a detailed calculation and recommendation for their expected monthly income."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to calculate the average engagement rate for YouTube creators in the 'tech review' niche using data from Crowdtangle, Social Blade, and Hootsuite. It also requires the agent to project earnings for a creator with 1 million subscribers based on YouTube’s CPM range of $4-10. A successful completion includes accurate calculations, use of all specified platforms, and a structured recommendation for monthly income.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using data from Crowdtangle, Social Blade, and Hootsuite’s free tools, find the average engagement rate for creators in the 'tech review' niche on YouTube. Calculate the projected earnings for a creator with 1 million subscribers, assuming YouTube’s typical CPM range of $4-10. Provide a detailed calculation and recommendation for their expected monthly income.

## Task-Specific Constraints
- Must visit Crowdtangle, Social Blade, and Hootsuite platforms.
- Must include engagement rate data for 'tech review' creators.
- Must calculate projected earnings using CPM range of $4-10.
- Output must include a structured calculation and recommendation.
- Must provide evidence of platform usage in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Crowdtangle, Social Blade, and Hootsuite? Which ones were actually visited?
- Is engagement rate data for 'tech review' creators present in the response?
- Are earnings calculations based on the CPM range of $4-10 included?
- Is the output structured with clear calculations and recommendations?
- Are all quantitative claims accurate and sourced?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's calculations and recommendations are correct and complete.

5 — All calculations are accurate, complete, and include clear recommendations.
4 — Minor inaccuracies or omissions in calculations or recommendations.
3 — Calculations are partially complete but usable; recommendations are vague.
2 — Major inaccuracies or incomplete calculations; recommendations missing.
1 — No usable calculations or recommendations.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms and included relevant data.

5 — Data from Crowdtangle, Social Blade, and Hootsuite is fully utilized.
4 — Data from at least 2 platforms is utilized; minor omissions.
3 — Data from 1 platform is utilized; significant omissions.
2 — No meaningful data from required platforms; mostly missing.
1 — No platform usage evident.

#### C. Depth of Analysis (0.25)
Measures the level of detail and specificity in the agent's calculations and recommendations.

5 — Includes detailed engagement rates, CPM breakdowns, and income projections.
4 — Includes engagement rates and income projections but lacks some detail.
3 — Includes basic calculations but lacks depth or specificity.
2 — Calculations are vague and lack meaningful detail.
1 — No meaningful calculations or analysis.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and uses credible sources.

5 — Response is well-organized, structured, and cites credible sources.
4 — Response is mostly organized but lacks some clarity or citations.
3 — Response is usable but poorly structured or missing citations.
2 — Response is disorganized and lacks credibility.
1 — Response is incoherent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
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
    "platform_coverage": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())