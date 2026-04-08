"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Evaluate subscriber growth and revenue potential across YouTube, Twitch, and Facebook Gaming to recommend the best platform.
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


TASK_INSTRUCTION = """Fetch monthly subscriber growth data and average earnings per subscriber for creators on YouTube, Twitch, and Facebook Gaming for the last 6 months. Use this data to calculate which platform offers the highest revenue potential per subscriber and recommend the top choice."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to fetch subscriber growth data and average earnings per subscriber for creators on YouTube, Twitch, and Facebook Gaming over the last 6 months. The agent must use this data to calculate revenue potential per subscriber for each platform and recommend the best choice.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch monthly subscriber growth data and average earnings per subscriber for creators on YouTube, Twitch, and Facebook Gaming for the last 6 months. Use this data to calculate which platform offers the highest revenue potential per subscriber and recommend the top choice.

## Task-Specific Constraints
- Must visit YouTube, Twitch, and Facebook Gaming to collect data.
- Must include subscriber growth data and average earnings per subscriber for each platform.
- Output must include a calculation of revenue potential per subscriber for each platform.
- Recommendation must be based on the calculated revenue potential.
- Output must be organized as a structured table or list.
- Must provide sources or evidence for all data used.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube, Twitch, and Facebook Gaming? Which platforms were actually visited?
- Are subscriber growth data and average earnings per subscriber present for all three platforms?
- Is the revenue potential calculation present and correct for each platform?
- Is the recommendation clearly based on the calculated revenue potential?
- Is the output organized as a structured table or list?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (revenue potential calculation and recommendation) is correct and complete.

5 — Provides accurate revenue potential calculations and a clear recommendation based on them.
4 — Minor inaccuracies in calculations or recommendation logic.
3 — Partial calculations or unclear recommendation.
2 — Major inaccuracies or missing calculations/recommendation.
1 — No calculations or recommendation present.

#### B. Coverage of Platforms (0.30)
Measures whether data was collected from all required platforms (YouTube, Twitch, Facebook Gaming).

5 — Includes data from all three platforms with no omissions.
4 — Includes data from two platforms, with minor omissions.
3 — Includes data from one platform, or incomplete data from multiple platforms.
2 — Minimal data collected, missing key platforms.
1 — No data collected from any platform.

#### C. Depth of Analysis (0.20)
Measures the specificity and detail of the data and calculations provided.

5 — Provides detailed subscriber growth and earnings data, with clear calculations.
4 — Provides detailed data but lacks clarity in calculations or comparisons.
3 — Provides partial data or lacks specificity in calculations.
2 — Minimal data or vague calculations.
1 — No specific data or calculations provided.

#### D. Output Structure and Credibility (0.15)
Measures the organization of the output and credibility of sources used.

5 — Output is well-organized (e.g., table) and sources are credible.
4 — Output is organized but sources lack credibility or are incomplete.
3 — Output is partially organized or sources are unclear.
2 — Output is disorganized or sources are missing.
1 — Output is completely disorganized and lacks sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited YouTube, Twitch, and Facebook Gaming but omitted key data for Twitch. Subscriber growth and earnings data were present for YouTube and Facebook Gaming. Revenue potential calculations were partially correct, but the recommendation lacked clarity.",
  "deliverable_accuracy": 3,
  "coverage_of_platforms": 3,
  "depth_of_analysis": 3,
  "output_structure_and_credibility": 2,
  "dimension_reasoning": {
    "deliverable_accuracy": "Revenue calculations were partially correct but lacked clarity in the recommendation.",
    "coverage_of_platforms": "Data was collected from YouTube and Facebook Gaming but omitted Twitch.",
    "depth_of_analysis": "Subscriber growth and earnings data were present but lacked specificity for Twitch.",
    "output_structure_and_credibility": "Output was partially organized but lacked credible sources."
  },
  "overall_score": 2.95,
  "passed": false
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())