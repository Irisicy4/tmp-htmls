"""
LLM-as-judge evaluator for EvolveBench task-88.

Category: Daily Activities
Task: Search online to find information about the current foot traffic and popularity of Jewel Changi Airport shopping mall in Singapore. Look for recent visitor reviews, social media posts, and any available foot traffic data. Summarize your findings.
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


TASK_INSTRUCTION = """Search online to find information about the current foot traffic and popularity of Jewel Changi Airport shopping mall in Singapore. Look for recent visitor reviews, social media posts, and any available foot traffic data. Summarize your findings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves searching for current or recent foot traffic information about Jewel Changi Airport shopping mall in Singapore.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Location: Jewel Changi Airport, Singapore
- Data: foot traffic, visitor numbers, or crowd levels — current or very recent
- Sources: search across multiple platforms (social media, news, reviews, data sites)
- Output: clear assessment of current crowd levels

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for Jewel Changi Airport foot traffic information?
- What sources were used (Google reviews, TripAdvisor, social media, news, etc.)?
- What specific data or impressions were found?
- Is the information recent?
- Is a clear assessment of crowd level provided?

### Step 2: Dimension Scoring

#### A. Search Execution (0.25)
Did the agent search for foot traffic at Jewel Changi Airport?

5 — Searched multiple platforms (Google reviews, TripAdvisor, Instagram, news) for Jewel Changi visitor info.
4 — Searched 2-3 sources.
3 — Only general web search without platform-specific sources.
2 — Described what to search without searching.
1 — No search.

#### B. Data Recency (0.3)
Is the information current or recent?

5 — Data from within the past week or clearly current (real-time crowd indicator or recent post).
4 — Data from past month.
3 — Data is recent but exact date unclear.
2 — Historical data without recency context.
1 — No temporal context.

#### C. Crowd Assessment (0.35)
Is a clear foot traffic assessment provided?

5 — Specific crowd level given (e.g. 'very busy, estimated X visitors on weekends', 'peak hours are...') with evidence.
4 — Good assessment with some evidence.
3 — General impression without specific data.
2 — 'It may be busy' without evidence.
1 — No assessment.

#### D. Source Quality (0.1)
Are credible sources cited?

5 — Sources named (Google reviews, TripAdvisor posts, news articles) with recency.
4 — Sources mentioned but vaguely.
3 — Single source only.
2 — Sources not cited.
1 — No sourcing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "search_execution": <1-5>,
  "data_recency": <1-5>,
  "crowd_assessment": <1-5>,
  "source_quality": <1-5>,
  "dimension_reasoning": {{
    "search_execution": "<one sentence citing specific evidence>",
    "data_recency": "<one sentence citing specific evidence>",
    "crowd_assessment": "<one sentence citing specific evidence>",
    "source_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "search_execution": 0.25,
    "data_recency": 0.3,
    "crowd_assessment": 0.35,
    "source_quality": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())