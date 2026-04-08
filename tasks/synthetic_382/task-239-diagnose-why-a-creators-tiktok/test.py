"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Diagnose why a creator’s TikTok video views dropped significantly and suggest a resolution strategy.
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


TASK_INSTRUCTION = """Diagnose why a creator’s TikTok video views dropped significantly over the past week. Explore recent changes to TikTok’s algorithm, public complaints on forums such as Reddit, and official documentation updates. Identify the likely cause and suggest a resolution strategy."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to diagnose why a TikTok creator's video views dropped significantly. The agent must investigate TikTok algorithm changes, public complaints on forums like Reddit, and official documentation updates. A successful completion includes identifying the likely cause and providing a resolution strategy.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why a creator’s TikTok video views dropped significantly over the past week. Explore recent changes to TikTok’s algorithm, public complaints on forums such as Reddit, and official documentation updates. Identify the likely cause and suggest a resolution strategy.

## Task-Specific Constraints
- Must visit tiktok.com, reddit.com, and socialmediatoday.com.
- Must identify at least one recent TikTok algorithm change.
- Must summarize at least two public complaints or trends from Reddit.
- Must cite official documentation or credible sources from socialmediatoday.com.
- Output must include both the likely cause and a resolution strategy.
- Output must be structured as a clear list or paragraph format.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tiktok.com, reddit.com, and socialmediatoday.com? Which ones were actually visited?
- Did the agent identify a recent TikTok algorithm change? Was it clearly explained?
- Did the agent summarize at least two public complaints or trends from Reddit?
- Did the agent cite official documentation or credible sources from socialmediatoday.com?
- Is the output structured as a clear list or paragraph format?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the cause of the drop in views and provided a resolution strategy.

5 — Identifies a clear cause and provides a detailed resolution strategy addressing the issue.
4 — Identifies a likely cause and provides a reasonable resolution strategy, but lacks detail.
3 — Identifies a plausible cause but resolution strategy is vague or incomplete.
2 — Cause is unclear or resolution strategy is mostly missing.
1 — No cause or resolution strategy provided.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Uses all three platforms (tiktok.com, reddit.com, socialmediatoday.com) and extracts relevant information from each.
4 — Uses all three platforms but misses minor details or relevance.
3 — Uses at least two platforms with partial relevance or missing details.
2 — Uses only one platform or extracts mostly irrelevant information.
1 — Does not use any required platforms.

#### C. Depth and Specificity (0.20)
Measures the level of detail and specificity in the findings and resolution strategy.

5 — Provides highly detailed findings with specific examples, trends, or citations.
4 — Provides detailed findings but lacks some specificity or examples.
3 — Provides basic findings with minimal detail or examples.
2 — Findings are vague or lack meaningful detail.
1 — Findings are absent or completely generic.

#### D. Source Credibility and Output Structure (0.15)
Measures whether the agent cited credible sources and presented the output in a clear, structured format.

5 — Cites credible sources and organizes output in a clear, professional format.
4 — Cites credible sources but output structure is slightly unclear or inconsistent.
3 — Cites some credible sources but structure is minimally acceptable.
2 — Sources are mostly missing or output is poorly structured.
1 — No credible sources cited and output is disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
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
    "source_credibility_and_output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())