"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Analyze the SEO performance of a sample video titled 'AI for Beginners' using TubeBuddy or similar tools, and extract the on-page SEO score and keyword suggestions for optimization.
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


TASK_INSTRUCTION = """Use TubeBuddy's public demo or similar tools to complete a workflow that analyzes the SEO performance of a sample video titled 'AI for Beginners'. Extract the on-page SEO score and specific keyword suggestions for optimization."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to analyze the SEO performance of a sample video titled 'AI for Beginners' using TubeBuddy or similar tools. A successful completion involves extracting the on-page SEO score and providing specific keyword suggestions for optimization.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use TubeBuddy's public demo or similar tools to complete a workflow that analyzes the SEO performance of a sample video titled 'AI for Beginners'. Extract the on-page SEO score and specific keyword suggestions for optimization.

## Task-Specific Constraints
- Must use TubeBuddy or a similar SEO analysis tool.
- Must extract the on-page SEO score for the video.
- Must provide at least 3 specific keyword suggestions for optimization.
- Must include a structured output (e.g., a table or list) for the extracted data.
- Must provide evidence of tool usage in the response (e.g., screenshots or detailed descriptions).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to TubeBuddy or a similar tool? Was the platform used correctly?
- Did the agent extract the on-page SEO score for the video?
- Are at least 3 specific keyword suggestions present in the response?
- Is the output organized in a structured format (e.g., table or list)?
- Does the response provide evidence of tool usage (e.g., screenshots or detailed descriptions)?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly extracted the on-page SEO score and provided keyword suggestions.

5 — Extracts the SEO score and provides 3+ specific keyword suggestions.
4 — Extracts the SEO score and provides 2 specific keyword suggestions.
3 — Extracts the SEO score but provides only 1 keyword suggestion or lacks specificity.
2 — Fails to extract the SEO score or provides no meaningful keyword suggestions.
1 — No attempt to extract the SEO score or provide keyword suggestions.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent used TubeBuddy or a similar tool as required.

5 — Clearly uses TubeBuddy or a similar tool and provides evidence of usage.
4 — Uses TubeBuddy or a similar tool but provides limited evidence.
3 — Uses TubeBuddy or a similar tool but provides no evidence.
2 — Attempts to use a tool but fails to complete the task.
1 — Does not attempt to use TubeBuddy or a similar tool.

#### C. Depth and Specificity (0.20)
Measures the level of detail and specificity in the response.

5 — Provides highly detailed keyword suggestions with specific optimization strategies.
4 — Provides moderately detailed keyword suggestions with some optimization strategies.
3 — Provides basic keyword suggestions with minimal detail.
2 — Provides vague or generic keyword suggestions.
1 — Provides no meaningful detail or specificity.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the output.

5 — Output is well-structured (e.g., table or list) and includes credible evidence (e.g., screenshots).
4 — Output is structured but lacks credible evidence.
3 — Output is minimally structured and lacks evidence.
2 — Output is poorly organized or unclear.
1 — Output is absent or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())