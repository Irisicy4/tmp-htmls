"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Research and compare monetization options for creators on YouTube, TikTok, and Instagram, and present findings in a table.
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


TASK_INSTRUCTION = """Research and compare three social media platforms (YouTube, TikTok, and Instagram) on their monetization options for creators. Focus on revenue streams like ad revenue, brand collaborations, and subscription features. Create a table with the key differences in terms of eligibility, payout structure, and additional available tools for monetization."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research and compare monetization options for creators on YouTube, TikTok, and Instagram. The agent must identify key differences in eligibility, payout structure, and additional tools for monetization. The deliverable must be presented as a table.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three social media platforms (YouTube, TikTok, and Instagram) on their monetization options for creators. Focus on revenue streams like ad revenue, brand collaborations, and subscription features. Create a table with the key differences in terms of eligibility, payout structure, and additional available tools for monetization.

## Task-Specific Constraints
- Must visit YouTube, TikTok, and Instagram to gather information.
- Must include eligibility criteria for monetization on each platform.
- Must include payout structures for ad revenue, brand collaborations, and subscriptions.
- Output must be organized as a table.
- Must include at least one additional monetization tool per platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube, TikTok, and Instagram? Which platforms were actually visited?
- Does the response include eligibility criteria for monetization on each platform?
- Does the response include payout structures for ad revenue, brand collaborations, and subscriptions?
- Is the output organized as a table?
- Are additional monetization tools mentioned for each platform?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the output accurately and completely fulfills the task requirements.

5 — All required monetization details (eligibility, payout structure, tools) are accurate and complete for all three platforms.
4 — Minor inaccuracies or omissions in one platform's details.
3 — Some inaccuracies or omissions, but most key details are present.
2 — Significant inaccuracies or omissions in multiple platforms' details.
1 — Response is mostly or entirely incorrect.

#### B. Coverage of Platforms (0.30)
Measures whether the agent covered all three platforms as required.

5 — All three platforms (YouTube, TikTok, Instagram) are fully covered.
4 — All three platforms are mentioned, but one is missing minor details.
3 — All three platforms are mentioned, but two or more are missing minor details.
2 — Only two platforms are mentioned or covered.
1 — Only one or no platforms are mentioned.

#### C. Depth and Specificity (0.20)
Measures whether the response includes detailed and specific information.

5 — Includes specific eligibility criteria, payout structures, and tools with examples or numbers for all platforms.
4 — Includes most details, but lacks some specificity or examples.
3 — Includes some details, but lacks depth or specificity in multiple areas.
2 — Includes very few details or is overly vague.
1 — Includes no meaningful details.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-organized and presented as a table.

5 — Output is fully organized as a table with clear headers and logical structure.
4 — Output is mostly organized as a table but has minor formatting issues.
3 — Output is partially organized as a table but lacks clarity or consistency.
2 — Output is poorly organized or not in table format.
1 — Output is completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())