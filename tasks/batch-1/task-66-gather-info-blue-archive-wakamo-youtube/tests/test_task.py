"""
LLM-as-judge evaluator for EvolveBench task-66.

Category: (Self) Media
Task: For the smartphone game Blue Archive's Total Assault mode featuring Wakamo Hovercraft on Torment dif...
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


TASK_INSTRUCTION = """For the smartphone game Blue Archive's Total Assault mode featuring Wakamo Hovercraft on Torment difficulty, gather and organize information about effective team compositions, focusing primarily on YouTube videos."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching a specific mobile game boss raid (Blue Archive Total Assault, Wakamo Hovercraft, Torment difficulty) and compiling team composition guides from YouTube.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Game: Blue Archive specifically
- Mode: Total Assault (总力战)
- Boss: Wakamo Hovercraft on Torment difficulty
- Primary source: YouTube videos — not just text guides
- Output: organized compilation of team compositions with source references

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search YouTube for Wakamo Hovercraft Total Assault guides?
- What specific team compositions were found?
- Are YouTube video sources cited?
- Is information organized by team composition type or ranking?
- Is difficulty (Torment) confirmed in the sources?

### Step 2: Dimension Scoring

#### A. Youtube Research (0.3)
Did the agent find relevant YouTube content?

5 — Multiple YouTube videos found specifically for Wakamo Hovercraft Torment with video titles/channels cited.
4 — YouTube content found but not all specifically Torment difficulty.
3 — YouTube searched but limited results; supplemented with other sources.
2 — General web search instead of YouTube focus.
1 — No YouTube research.

#### B. Team Comp Detail (0.35)
Are specific team compositions reported?

5 — Multiple team compositions with character names, equipment, and roles (DPS/support/healer).
4 — Team compositions listed with character names but limited detail.
3 — Character names mentioned without team context.
2 — Vague descriptions (e.g. 'use strong characters') without specifics.
1 — No team compositions found.

#### C. Source Organization (0.2)
Is content organized and sources cited?

5 — Content organized by composition type with YouTube source links or channel names.
4 — Organized but sources only partially cited.
3 — Information present but not well-organized.
2 — Dump of information without structure.
1 — No organization or sourcing.

#### D. Difficulty Accuracy (0.15)
Is the content specifically for Torment difficulty?

5 — All content confirmed for Torment difficulty explicitly.
4 — Mostly Torment but some content from lower difficulties.
3 — Difficulty level unclear for most content.
2 — Content from wrong difficulty.
1 — Difficulty not addressed.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "youtube_research": <1-5>,
  "team_comp_detail": <1-5>,
  "source_organization": <1-5>,
  "difficulty_accuracy": <1-5>,
  "dimension_reasoning": {{
    "youtube_research": "<one sentence citing specific evidence>",
    "team_comp_detail": "<one sentence citing specific evidence>",
    "source_organization": "<one sentence citing specific evidence>",
    "difficulty_accuracy": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "youtube_research": 0.3,
    "team_comp_detail": 0.35,
    "source_organization": 0.2,
    "difficulty_accuracy": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())