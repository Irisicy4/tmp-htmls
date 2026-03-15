"""
LLM-as-judge evaluator for EvolveBench task-09.

Category: (Self) Media
Task: "go on youtube and find videos of silent hill 3 that explains the story"
"""

import os, json, re

TASK_INSTRUCTION = "go on youtube and find videos of silent hill 3 that explains the story"
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a YouTube video search task.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: YouTube specifically
- Game: Silent Hill 3 specifically — not Silent Hill 1, 2, or other entries in the series
- Content type: videos that explain the story — lore breakdowns, narrative explanations, story summaries; not gameplay walkthroughs without commentary or unrelated content
- Output: the agent should present specific videos (title, channel, URL or description) — not just describe what to search for

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube? Cite evidence from trace or response.
- What specific videos did the agent find? List titles and channels if mentioned.
- Are the videos about Silent Hill 3 specifically (not other SH games)?
- Do the videos explain the story (lore, narrative, plot analysis) rather than just gameplay?
- How many videos did the agent surface?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent actually navigate YouTube as instructed?

5 — Clear evidence agent visited YouTube and retrieved specific video results.
4 — Agent found YouTube videos but with limited detail (titles only, no channels or URLs).
3 — Agent searched YouTube but retrieved only search results page without accessing specific videos.
2 — Agent used a non-YouTube source or only described what to search for.
1 — No search performed; response is from prior knowledge or fabricated.

#### B. Game Specificity
Are the videos specifically about Silent Hill 3?

5 — All videos are clearly about Silent Hill 3 (title/description confirms the specific game).
4 — Most videos are SH3-specific; 1 is ambiguous (e.g. "Silent Hill series lore").
3 — Videos cover the Silent Hill series broadly; SH3 is included but not the focus.
2 — Videos are about a different Silent Hill entry or horror games generally.
1 — No relevant Silent Hill content found or videos are completely off-topic.

#### C. Content Type Match
Do the videos explain the story as requested?

5 — All or most videos are story explanations, lore breakdowns, or narrative analyses of SH3.
4 — Most videos are story-focused; 1 is a gameplay walkthrough with some story commentary.
3 — Videos are SH3-related but are primarily gameplay recordings without story focus.
2 — Videos are tangentially related (e.g. reviews or tier lists mentioning SH3's story briefly).
1 — Videos do not match the requested content type at all.

#### D. Result Usefulness
Are the results presented in a useful, actionable way?

5 — 3+ videos presented with: title, channel name, and brief description of story coverage; easy to choose from.
4 — 3+ videos with titles and channels; descriptions are minimal.
3 — 2–3 videos mentioned with titles only; no channel or description.
2 — 1–2 videos mentioned or results are vague (e.g. "several videos found").
1 — No specific videos presented; agent only described a search or gave generic advice.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_execution": <1-5>,
  "game_specificity": <1-5>,
  "content_type_match": <1-5>,
  "result_usefulness": <1-5>,
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "game_specificity": "<one sentence citing specific evidence>",
    "content_type_match": "<one sentence citing specific evidence>",
    "result_usefulness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_execution":  0.25,
    "game_specificity":    0.30,
    "content_type_match":  0.30,
    "result_usefulness":   0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse_answer_tag(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
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
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }