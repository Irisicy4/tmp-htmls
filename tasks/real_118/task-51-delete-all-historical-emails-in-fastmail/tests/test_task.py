"""
LLM-as-judge evaluator for EvolveBench task-51.

Category: Daily Activities
Task: Delete all historical emails related to verification codes in Fastmail.
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

def _call(agent_response, execution_summary, system_prompt, user_prompt_template, task_instruction):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(
                    task_instruction=task_instruction,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes, dimensions, weights, pass_threshold):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dimensions)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in dimensions}
    overall = sum(aggregated[d] * weights[d] for d in dimensions)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= pass_threshold
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated


TASK_INSTRUCTION = """Delete all historical emails in Fastmail that are related to verification codes."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent successfully found and deleted verification code emails from a Fastmail inbox."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Fastmail specifically
- Scope: all historical verification code emails — the agent should be thorough
- Action: actual deletion required, not just identification
- Search strategy: agent should use iterative search refinement to catch all variants

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Fastmail?
- What search queries were used to find verification code emails?
- How many emails were found and deleted?
- Did the agent use multiple search iterations to ensure completeness?
- Was deletion confirmed?

### Step 2: Dimension Scoring

#### A. Platform Access (0.2)
Did the agent access Fastmail?

5 — Agent navigated to Fastmail and accessed the inbox.
4 — Agent accessed Fastmail but with some difficulty.
3 — Agent attempted Fastmail but had login or access issues.
2 — Agent described Fastmail without navigating to it.
1 — No Fastmail navigation.

#### B. Search Strategy (0.3)
Did the agent use effective search to find verification code emails?

5 — Used multiple search queries covering various keywords (verification, 验证码, code, OTP, etc.) with iterative refinement.
4 — Used 2-3 good search queries.
3 — Used only one broad search query.
2 — Browsed inbox manually without searching.
1 — No search strategy.

#### C. Deletion Execution (0.35)
Were emails actually deleted?

5 — Agent selected and deleted emails with confirmation of deletion count or empty results after.
4 — Agent deleted emails but without clear confirmation of completeness.
3 — Agent deleted some emails but process was incomplete.
2 — Agent identified emails without deleting them.
1 — No deletion performed.

#### D. Completeness (0.15)
Did the agent ensure thorough coverage?

5 — Agent explicitly checked for remaining emails after deletion and confirmed inbox clear of verification emails.
4 — Agent did multiple passes but did not verify completeness.
3 — Agent did one pass only.
2 — Agent acknowledged potential incompleteness without addressing it.
1 — No completeness consideration.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_access": <1-5>,
  "search_strategy": <1-5>,
  "deletion_execution": <1-5>,
  "completeness": <1-5>,
  "dimension_reasoning": {{
    "platform_access": "<one sentence citing specific evidence>",
    "search_strategy": "<one sentence citing specific evidence>",
    "deletion_execution": "<one sentence citing specific evidence>",
    "completeness": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_access": 0.2,
    "search_strategy": 0.3,
    "deletion_execution": 0.35,
    "completeness": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first,
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION),
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)],
                           DIMENSIONS, DIMENSION_WEIGHTS, PASS_THRESHOLD)
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