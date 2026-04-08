"""
LLM-as-judge evaluator for EvolveBench task-74.

Category: Marketing & Analytics
Task: Collect and analyze the top YouTube Shorts accounts most watched by people aged 50 and above in Korea. I want to create 
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


TASK_INSTRUCTION = """Collect and analyze the top YouTube Shorts accounts most watched by people aged 50 and above in Korea. I want to create a new channel, so please gather very detailed information."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching the YouTube Shorts landscape in South Korea for the 50+ demographic, gathering channel data, content strategy insights, and actionable findings for a new creator.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: YouTube Shorts specifically (not general YouTube)
- Demographic: viewers aged 50+ in Korea
- Depth: detailed channel info — subscriber count, posting frequency, content style, engagement
- Goal: inform creation of a new channel — must include strategic insights

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Korean YouTube Shorts channels popular with 50+ viewers?
- What specific channels were identified (names, subscriber counts, content types)?
- Was engagement data (views, likes) collected?
- Are content strategy insights provided?
- Are recommendations for a new channel included?

### Step 2: Dimension Scoring

#### A. Channel Research (0.3)
Did the agent find specific, relevant Korean YouTube Shorts channels?

5 — 5+ specific channels identified with names, subscriber counts, and content focus targeting 50+ Koreans.
4 — 3-4 channels with most details.
3 — 1-2 channels or channels without demographic focus.
2 — General YouTube Shorts landscape without Korea/50+ focus.
1 — No channel research.

#### B. Data Depth (0.25)
Was detailed channel data collected?

5 — For each channel: subscribers, avg views per short, posting frequency, content themes, engagement rate.
4 — Most metrics present but one or two missing.
3 — Basic data (name + subscriber count) only.
2 — Very surface-level data.
1 — No channel data.

#### C. Demographic Relevance (0.25)
Is content specifically relevant to the 50+ Korean demographic?

5 — Clear evidence that identified channels target 50+ Koreans (content topics, language, cultural references).
4 — Channels plausibly relevant but demographic targeting not confirmed.
3 — Korean channels but not specifically 50+ focused.
2 — Generic popular channels without demographic filter.
1 — No demographic relevance.

#### D. Strategic Insights (0.2)
Are actionable insights for a new channel provided?

5 — Specific recommendations: content niches, posting schedule, style, hooks that work for the demographic.
4 — Some strategic guidance but not fully actionable.
3 — Generic advice not grounded in the research.
2 — Research presented without strategic conclusions.
1 — No strategic insights.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "channel_research": <1-5>,
  "data_depth": <1-5>,
  "demographic_relevance": <1-5>,
  "strategic_insights": <1-5>,
  "dimension_reasoning": {{
    "channel_research": "<one sentence citing specific evidence>",
    "data_depth": "<one sentence citing specific evidence>",
    "demographic_relevance": "<one sentence citing specific evidence>",
    "strategic_insights": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "channel_research": 0.3,
    "data_depth": 0.25,
    "demographic_relevance": 0.25,
    "strategic_insights": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())