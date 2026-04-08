"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Extract top 10 trending music videos from YouTube and analyze audience sentiment on Twitter.
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


TASK_INSTRUCTION = """Go to YouTube's trending section, filter for videos in the 'Music' category, and extract the top 10 trending videos with their title, channel name, and views. Additionally, visit Twitter and check any tweets mentioning these videos for audience sentiment keywords (e.g., 'love', 'hate', 'favorite')."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to extract the top 10 trending music videos from YouTube, including their title, channel name, and view count, and then analyze audience sentiment on Twitter by checking for specific keywords related to these videos. A successful completion includes accurate extraction of YouTube data and meaningful sentiment analysis from Twitter.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to YouTube's trending section, filter for videos in the 'Music' category, and extract the top 10 trending videos with their title, channel name, and views. Additionally, visit Twitter and check any tweets mentioning these videos for audience sentiment keywords (e.g., 'love', 'hate', 'favorite').

## Task-Specific Constraints
- Must navigate to YouTube's trending section and filter for the 'Music' category.
- Must extract exactly 10 videos with their title, channel name, and view count.
- Must visit Twitter and analyze tweets mentioning these videos.
- Must identify sentiment keywords ('love', 'hate', 'favorite') in the tweets.
- Output must be structured as a table or list with clear labels for each data field.
- Must provide evidence of platform navigation in the tool-call trace.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube and Twitter as required? Was this evident in the tool-call trace?
- Did the agent extract exactly 10 videos with their title, channel name, and view count?
- Did the agent analyze tweets mentioning these videos for sentiment keywords ('love', 'hate', 'favorite')?
- Is the output organized as a structured table or list with clear labels?
- Are the extracted data and sentiment analysis accurate and meaningful?

### Step 2: Dimension Scoring

#### A. Data Extraction Accuracy (0.35)
Measures whether the agent accurately extracted the required YouTube data.

5 — Extracts all 10 videos with correct titles, channel names, and view counts.
4 — Extracts 8-9 videos with mostly correct data.
3 — Extracts 6-7 videos or has significant inaccuracies in data.
2 — Extracts fewer than 6 videos or mostly incorrect data.
1 — Fails to extract any meaningful data.

#### B. Sentiment Analysis Coverage (0.30)
Measures whether the agent analyzed tweets for all 10 videos and identified sentiment keywords.

5 — Analyzes tweets for all 10 videos and identifies all sentiment keywords.
4 — Analyzes tweets for 8-9 videos and identifies most sentiment keywords.
3 — Analyzes tweets for 6-7 videos or misses some keywords.
2 — Analyzes tweets for fewer than 6 videos or misses most keywords.
1 — Fails to analyze tweets or identify sentiment keywords.

#### C. Output Structure and Clarity (0.20)
Measures whether the output is well-organized and easy to interpret.

5 — Output is a clear, structured table or list with all required fields labeled.
4 — Output is mostly clear with minor formatting issues.
3 — Output is partially structured but lacks clarity or labels.
2 — Output is poorly structured and difficult to interpret.
1 — Output is unstructured or incomprehensible.

#### D. Platform Navigation Evidence (0.15)
Measures whether the agent's tool-call trace shows clear evidence of platform navigation.

5 — Tool-call trace clearly shows navigation to YouTube and Twitter.
4 — Tool-call trace shows navigation to both platforms but lacks some detail.
3 — Tool-call trace shows partial navigation evidence for one platform.
2 — Tool-call trace shows minimal or unclear navigation evidence.
1 — Tool-call trace shows no evidence of platform navigation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_extraction_accuracy": <1-5>,
  "sentiment_analysis_coverage": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "platform_navigation_evidence": <1-5>,
  "dimension_reasoning": {{
    "data_extraction_accuracy": "<one sentence citing specific evidence>",
    "sentiment_analysis_coverage": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>",
    "platform_navigation_evidence": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_extraction_accuracy": 0.35,
    "sentiment_analysis_coverage": 0.30,
    "output_structure_and_clarity": 0.20,
    "platform_navigation_evidence": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())