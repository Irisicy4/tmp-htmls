"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Investigate sizing discrepancies for an Adidas jacket across Amazon, Zappos, and Adidas.com, and provide a diagnosis and recommended workaround.
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


TASK_INSTRUCTION = """A user reports that sizing information for a specific Adidas jacket (Product ID: HZ9876) appears inconsistent across listings on Amazon, Zappos, and Adidas.com. Investigate the root cause of the discrepancy by checking size charts, user reviews, and product descriptions across these listings. Provide a diagnosis and recommended workaround or clarification path."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating sizing discrepancies for an Adidas jacket across Amazon, Zappos, and Adidas.com. The agent must analyze size charts, user reviews, and product descriptions to identify the root cause and provide a clear diagnosis and workaround. A successful completion includes accurate findings and actionable recommendations.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports that sizing information for a specific Adidas jacket (Product ID: HZ9876) appears inconsistent across listings on Amazon, Zappos, and Adidas.com. Investigate the root cause of the discrepancy by checking size charts, user reviews, and product descriptions across these listings. Provide a diagnosis and recommended workaround or clarification path.

## Task-Specific Constraints
- Must visit all three platforms: Amazon, Zappos, and Adidas.com.
- Must analyze size charts, user reviews, and product descriptions for the jacket.
- Must identify the root cause of the sizing discrepancy.
- Must provide actionable recommendations or a workaround for users.
- Output must be organized as a structured list or table.
- Must cite specific evidence from each platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (Amazon, Zappos, Adidas.com)?
- Did the agent analyze size charts, user reviews, and product descriptions for the jacket?
- Is the output organized as a structured list or table?
- Did the agent identify the root cause of the sizing discrepancy?
- Are actionable recommendations or workarounds provided and supported by evidence?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the diagnosis and recommendations are correct and complete.

5 — Identifies the root cause with clear evidence and provides actionable recommendations.
4 — Identifies the root cause but recommendations are slightly less actionable or clear.
3 — Partially identifies the root cause or provides incomplete recommendations.
2 — Incorrect diagnosis or vague recommendations.
1 — No diagnosis or recommendations provided.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and analyzed the necessary data.

5 — Analyzes size charts, user reviews, and product descriptions from all three platforms.
4 — Analyzes most required data but misses minor elements.
3 — Visits all platforms but analyzes only partial data.
2 — Visits fewer than three platforms or analyzes very limited data.
1 — Does not visit required platforms or analyze relevant data.

#### C. Depth of Analysis (0.20)
Measures the level of detail in the analysis, including comparisons and evidence.

5 — Provides detailed comparisons, cites specific evidence, and includes numerical data.
4 — Provides comparisons and evidence but lacks minor details.
3 — Provides basic analysis with limited comparisons or evidence.
2 — Analysis is shallow or lacks meaningful comparisons.
1 — No meaningful analysis provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and cites credible sources.

5 — Output is structured as a clear list or table and cites credible sources.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is usable but disorganized or lacks source citations.
2 — Output is poorly organized or lacks credibility.
1 — Output is absent or completely unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
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
    "coverage_of_platforms": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())