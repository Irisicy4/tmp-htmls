"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Research and compare monetization options for creators on YouTube, Twitch, and Substack.
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


TASK_INSTRUCTION = """Research and compare the monetization options for creators on YouTube, Twitch, and Substack. Include information on revenue-sharing percentages, subscription models, ad revenue possibilities, and any notable restrictions or eligibility criteria for each platform."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare monetization options for creators on YouTube, Twitch, and Substack. This involves gathering information about revenue-sharing percentages, subscription models, ad revenue possibilities, and notable restrictions or eligibility criteria. Successful completion requires accurate, structured, and comprehensive information about all three platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare the monetization options for creators on YouTube, Twitch, and Substack. Include information on revenue-sharing percentages, subscription models, ad revenue possibilities, and any notable restrictions or eligibility criteria for each platform.

## Task-Specific Constraints
- Must visit YouTube, Twitch, and Substack to gather information.
- Must include revenue-sharing percentages for all three platforms.
- Must describe subscription models and ad revenue possibilities for each platform.
- Must identify notable restrictions or eligibility criteria for each platform.
- Output must be organized as a structured list or table for easy comparison.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube, Twitch, and Substack? Which platforms were actually visited?
- Are revenue-sharing percentages included for all three platforms?
- Are subscription models and ad revenue possibilities described for each platform?
- Are notable restrictions or eligibility criteria identified for each platform?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's output is correct and complete.

5 — Includes accurate revenue-sharing percentages, subscription models, ad revenue possibilities, and restrictions for all three platforms.
4 — Minor inaccuracies or missing details for one platform.
3 — Partial completion with significant omissions or inaccuracies for two platforms.
2 — Major inaccuracies or missing details for all platforms.
1 — No meaningful information provided.

#### B. Coverage of Platforms (0.30)
Measures whether the agent researched all required platforms.

5 — Successfully gathered information from YouTube, Twitch, and Substack.
4 — Information from two platforms is complete; minor omissions for the third.
3 — Information from two platforms is incomplete or missing.
2 — Information from only one platform is provided.
1 — No platform-specific information is provided.

#### C. Depth and Specificity (0.25)
Measures the level of detail and specificity in the response.

5 — Includes detailed comparisons with specific numbers, eligibility criteria, and examples.
4 — Comparisons are present but lack some specifics or examples.
3 — Basic information is provided but lacks depth or specificity.
2 — Minimal information with little to no detail.
1 — No meaningful detail provided.

#### D. Output Organization (0.10)
Measures the structure and readability of the output.

5 — Output is well-organized as a structured list or table with clear comparisons.
4 — Output is organized but lacks clarity in some areas.
3 — Output is partially organized but difficult to follow.
2 — Output is disorganized and hard to interpret.
1 — No discernible structure in the output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_and_specificity": 0.25,
    "output_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())