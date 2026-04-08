"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Investigate reasons for a fictional YouTube channel's video view drop by analyzing platform-specific discussions and optimization strategies.
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


TASK_INSTRUCTION = """Investigate why a fictional YouTube channel's video views drastically dropped despite consistent posting. Find possible causes by checking YouTube's Creator Blog for algorithm updates, Reddit 'r/YouTube' discussions, and articles on video optimization from trusted sources like Ahrefs."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to investigate reasons for a fictional YouTube channel's video view drop by analyzing platform-specific discussions and optimization strategies. The agent must gather insights from YouTube's Creator Blog, Reddit 'r/YouTube' discussions, and articles on video optimization from Ahrefs.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Investigate why a fictional YouTube channel's video views drastically dropped despite consistent posting. Find possible causes by checking YouTube's Creator Blog for algorithm updates, Reddit 'r/YouTube' discussions, and articles on video optimization from trusted sources like Ahrefs.

## Task-Specific Constraints
- Must visit and extract information from YouTube's Creator Blog, Reddit 'r/YouTube', and Ahrefs.
- Must identify at least 3 distinct reasons for the view drop.
- Must provide evidence or examples for each reason cited.
- Output must be organized as a structured list or table.
- Must include references to specific posts, articles, or updates from each platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube's Creator Blog, Reddit 'r/YouTube', and Ahrefs? Which platforms were actually visited?
- Are at least 3 distinct reasons for the view drop identified?
- Does the response provide evidence or examples for each reason cited?
- Is the output organized as a structured list or table?
- Are references to specific posts, articles, or updates from each platform included?

### Step 2: Dimension Scoring

#### A. Reason Identification Accuracy (0.35)
Measures whether the agent correctly identified at least 3 distinct reasons for the view drop.

5 — Identifies 3 or more distinct reasons with supporting evidence/examples for each.
4 — Identifies 3 reasons but lacks evidence/examples for 1.
3 — Identifies 2 reasons with evidence/examples.
2 — Identifies 1 reason or lacks evidence/examples entirely.
1 — No reasons identified or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and extracted information from all required platforms.

5 — Extracts relevant information from YouTube's Creator Blog, Reddit 'r/YouTube', and Ahrefs.
4 — Extracts relevant information from 2 platforms but misses 1.
3 — Extracts relevant information from 1 platform.
2 — Visits platforms but extracts irrelevant or insufficient information.
1 — No platform visits or completely irrelevant information.

#### C. Depth of Analysis (0.20)
Measures the level of detail and specificity in the agent's response.

5 — Provides detailed analysis with examples, references, and specific insights for all reasons.
4 — Provides detailed analysis for most reasons but lacks depth for 1.
3 — Provides basic analysis with minimal examples or references.
2 — Provides shallow analysis with no examples or references.
1 — No analysis or completely incorrect.

#### D. Output Structure and Source Credibility (0.15)
Measures whether the response is well-organized and uses credible sources.

5 — Output is structured as a clear list or table with references to credible sources.
4 — Output is structured but lacks references for 1 platform or source credibility is questionable.
3 — Output is partially structured but lacks references or credible sources.
2 — Output is disorganized and lacks credible sources.
1 — Output is completely disorganized and lacks references entirely.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "reason_identification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "reason_identification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "reason_identification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_source_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())