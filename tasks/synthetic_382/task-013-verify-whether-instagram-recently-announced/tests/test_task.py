"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether Instagram recently announced or implemented a change to its algorithm affecting content visibility by cross-checking information on Instagram's official blog, Twitter account, and a reputable tech news site like TechCrunch.
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


TASK_INSTRUCTION = """Verify whether Instagram recently announced or implemented a change to its algorithm affecting content visibility. Cross-check this information on Instagram’s official blog, the company's Twitter account, and a reputable tech news site like TechCrunch."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to determine if Instagram recently announced or implemented a change to its algorithm affecting content visibility. The agent must cross-check this information using Instagram's official blog, Twitter account, and a reputable tech news site like TechCrunch. A successful completion requires accurate identification of any algorithm changes, supported by evidence from the required platforms.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Instagram recently announced or implemented a change to its algorithm affecting content visibility. Cross-check this information on Instagram’s official blog, the company's Twitter account, and a reputable tech news site like TechCrunch.

## Task-Specific Constraints
- Must visit Instagram's official blog, Twitter account, and TechCrunch.
- Must clearly identify whether a change to Instagram's algorithm was announced or implemented.
- Must provide evidence or quotes from each platform visited.
- Output must summarize findings in a clear and structured format.
- Must explicitly state if no relevant information was found on any of the platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Instagram's official blog, Twitter account, and TechCrunch? Which ones were actually visited?
- Did the agent identify whether a change to Instagram's algorithm was announced or implemented?
- Did the agent provide evidence or quotes from each platform visited?
- Is the output structured clearly and does it summarize findings effectively?
- Did the agent explicitly state if no relevant information was found on any platform?

### Step 2: Dimension Scoring

#### A. Algorithm Change Identification Accuracy (0.35)
Measures whether the agent correctly identified whether Instagram announced or implemented an algorithm change.

5 — Correctly identifies algorithm change and provides evidence from all three platforms.
4 — Correctly identifies algorithm change with evidence from two platforms.
3 — Correctly identifies algorithm change but provides evidence from only one platform.
2 — Incorrectly identifies algorithm change or lacks sufficient evidence.
1 — Does not address algorithm change at all.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and used all required platforms.

5 — Uses all three platforms (Instagram blog, Twitter, TechCrunch) with evidence from each.
4 — Uses two platforms with evidence from both.
3 — Uses one platform with evidence.
2 — Navigates to platforms but provides no evidence.
1 — Does not navigate to any required platforms.

#### C. Evidence Depth (0.20)
Measures the depth and specificity of evidence provided.

5 — Provides detailed quotes or data from all platforms.
4 — Provides detailed evidence from two platforms.
3 — Provides some evidence but lacks depth or specificity.
2 — Provides minimal evidence with no depth.
1 — Provides no evidence.

#### D. Output Clarity and Structure (0.15)
Measures how well the response is organized and presented.

5 — Clear, structured summary with findings from all platforms.
4 — Mostly clear and structured but with minor issues.
3 — Understandable but lacks clarity or structure.
2 — Poorly organized and difficult to follow.
1 — No structure or clarity.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "algorithm_change_identification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "evidence_depth": <1-5>,
  "output_clarity_and_structure": <1-5>,
  "dimension_reasoning": {{
    "algorithm_change_identification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "evidence_depth": "<one sentence citing specific evidence>",
    "output_clarity_and_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "algorithm_change_identification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "evidence_depth": 0.20,
    "output_clarity_and_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())