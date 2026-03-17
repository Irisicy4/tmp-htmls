"""
LLM-as-judge evaluator for EvolveBench task-20.

Category: (Self) Media
Task: Search Baidu Tieba for a cultivation/system progression guide for
      the mobile game 境界刀鸣 (Jingjie Daoming) and compile it.
"""

import os, json, re

TASK_INSTRUCTION = "Search Baidu Tieba to find and compile a cultivation/system progression guide for the mobile game 境界刀鸣 (Jingjie Daoming)."
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully searched Baidu Tieba for a specific mobile game guide and compiled it.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Baidu Tieba (百度贴吧) specifically — not Baidu search, Bilibili, or other platforms
- Game: 境界刀鸣 (Jingjie Daoming) specifically — a Chinese mobile RPG
- Content type: cultivation/system progression guide — covering how the game's character advancement, skill systems, or resource progression works
- Output: a compiled guide (not just a list of links), synthesised from Tieba posts

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Baidu Tieba? Cite evidence from trace or response.
- What game-specific content did the agent find? Describe briefly.
- Is the content about 境界刀鸣 specifically?
- Does the content cover cultivation/progression systems (not just general gameplay tips)?
- Did the agent compile content from multiple posts, or just summarise one?
- Is the output a usable guide or just a description of what was found?

### Step 2: Dimension Scoring

#### A. Platform Execution
Did the agent actually navigate Baidu Tieba as instructed?

5 — Clear evidence agent searched and browsed Baidu Tieba for 境界刀鸣 content; multiple posts accessed.
4 — Agent used Baidu Tieba but accessed only 1–2 posts, or trace confirms platform but response doesn't cite it.
3 — Ambiguous: response mentions Baidu Tieba but trace shows no navigation, or agent used Tieba alongside other platforms.
2 — Agent used Baidu search or a different platform instead of Tieba.
1 — No search performed; content appears fabricated or from prior knowledge.

#### B. Game Specificity
Is the content specifically about 境界刀鸣?

5 — All content is clearly about 境界刀鸣 (game name appears in sources; content references game-specific mechanics).
4 — Most content is game-specific; 1 section is generic mobile RPG advice that could apply to any game.
3 — Content is broadly relevant to Chinese mobile RPG cultivation systems but lacks 境界刀鸣-specific detail.
2 — Content is generic mobile game advice with only superficial mention of the game name.
1 — Content is off-topic or clearly about a different game.

#### C. Guide Depth
Does the compiled content constitute a useful cultivation/progression guide?

5 — Guide covers 3+ distinct progression systems (e.g. character levels, skill trees, equipment upgrade, resource farming) with specific mechanics and tips.
4 — Guide covers 2–3 systems with useful detail; one area is shallow.
3 — Guide covers 1–2 systems; useful but limited in scope.
2 — Content mentions cultivation/progression but provides only surface-level overview without actionable guidance.
1 — No guide content; output is a description of the search process or a list of post titles only.

#### D. Compilation Quality
Is the output a well-organised, readable guide compiled from multiple sources?

5 — Content is synthesised from 2+ Tieba posts into a coherent, structured guide with clear sections.
4 — Content is organised but lightly edited from sources; structure is present but synthesis is minimal.
3 — Content is assembled but reads as disconnected fragments from different posts.
2 — Output is a single post's content without compilation or a list of URLs without content.
1 — No compiled output; agent only described what it searched for.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_execution": <1-5>,
  "game_specificity": <1-5>,
  "guide_depth": <1-5>,
  "compilation_quality": <1-5>,
  "dimension_reasoning": {{
    "platform_execution": "<one sentence citing specific evidence>",
    "game_specificity": "<one sentence citing specific evidence>",
    "guide_depth": "<one sentence citing specific evidence>",
    "compilation_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_execution":  0.25,
    "game_specificity":    0.25,
    "guide_depth":         0.30,
    "compilation_quality": 0.20,
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