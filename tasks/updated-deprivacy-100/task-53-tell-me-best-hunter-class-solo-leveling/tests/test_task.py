"""
LLM-as-judge evaluator for EvolveBench task-53.

Category: (Self) Media
Task: Recommend the best hunter class to develop in Solo Leveling ARISE and explain why.
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


TASK_INSTRUCTION = """Tell me the best hunter class to develop in the Netmarble game Solo Leveling: ARISE and explain why you chose it."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent provided an accurate, well-reasoned recommendation for the best hunter class in Solo Leveling: ARISE."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Game: Solo Leveling: ARISE by Netmarble specifically
- Must recommend a specific hunter class (not a vague 'it depends')
- Recommendation must be based on current meta (tier lists, community consensus)
- Reasoning must address growth potential, versatility, and/or end-game viability

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for current meta information on Solo Leveling ARISE?
- What specific hunter class was recommended?
- What reasoning was provided (tier ranking, versatility, growth potential)?
- Is the recommendation consistent with current community consensus?

### Step 2: Dimension Scoring

#### A. Research Quality (0.25)
Did the agent search for current meta data?

5 — Agent searched for current tier lists, guides, or community posts about Solo Leveling ARISE hunter classes.
4 — Agent found some meta info but sources are limited.
3 — Agent relied on general knowledge without specific search.
2 — Agent described classes generally without meta research.
1 — No research performed.

#### B. Recommendation Specificity (0.25)
Did the agent recommend a specific class with clear reasoning?

5 — Specific class recommended with clear reasons: tier ranking, versatility, growth rate, end-game value.
4 — Specific class recommended but reasoning covers only 1-2 factors.
3 — Class recommended but reasoning is vague or generic.
2 — Multiple classes mentioned without a clear winner.
1 — No specific recommendation.

#### C. Meta Accuracy (0.35)
Is the recommendation consistent with current game meta?

5 — Recommended class is S-tier or widely considered top-tier in current meta with supporting evidence.
4 — Recommended class is strong but not universally top-tier; reasoning compensates.
3 — Recommended class is reasonable but not optimal.
2 — Recommended class is sub-optimal or outdated.
1 — Recommendation is clearly wrong or the class doesn't exist.

#### D. Explanation Depth (0.15)
Is the explanation detailed and useful?

5 — Covers tier ranking, growth mechanics, skill synergies, and end-game role.
4 — Covers 2-3 of the above factors.
3 — Basic explanation with only 1 factor.
2 — Very thin explanation.
1 — No explanation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "research_quality": <1-5>,
  "recommendation_specificity": <1-5>,
  "meta_accuracy": <1-5>,
  "explanation_depth": <1-5>,
  "dimension_reasoning": {{
    "research_quality": "<one sentence citing specific evidence>",
    "recommendation_specificity": "<one sentence citing specific evidence>",
    "meta_accuracy": "<one sentence citing specific evidence>",
    "explanation_depth": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "research_quality": 0.25,
    "recommendation_specificity": 0.25,
    "meta_accuracy": 0.35,
    "explanation_depth": 0.15,
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