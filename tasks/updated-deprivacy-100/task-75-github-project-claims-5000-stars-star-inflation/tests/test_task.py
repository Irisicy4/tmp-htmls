"""
LLM-as-judge evaluator for EvolveBench task-75.

Category: Software Engineering
Task: This GitHub project claims to have reached 5,000 stars in 48 hours. I'm certain the data is fake. Please compile a recor
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
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

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


TASK_INSTRUCTION = """This GitHub project claims to have reached 5,000 stars in 48 hours. I'm certain the data is fake. Please compile a record of star inflation and also analyze the forks from the same angle."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating a GitHub repository for signs of artificial star and fork inflation, using GitHub's API or tools like Star History to detect suspicious growth patterns.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target: a specific GitHub repository (must be identified from context or searched)
- Analysis: star growth timeline — look for unnatural spikes, bot-like patterns
- Forks: same analysis applied to fork count
- Evidence: must use data (star history charts, API data, timestamps) not just assertion

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent access GitHub star history data for the repository?
- What growth patterns were found (spikes, timing, geographic concentration)?
- Was fork analysis also performed?
- What specific evidence of inflation was identified?
- Were external tools (star-history.com, GitHub API) used?

### Step 2: Dimension Scoring

#### A. Star History Analysis (0.35)
Did the agent analyze star growth timeline?

5 — Star growth timeline retrieved (via star-history.com or GitHub API) showing specific dates, volumes, and suspicious spikes.
4 — Star growth analyzed but with less precise data.
3 — Growth pattern described without timeline data.
2 — Agent asserted inflation without data.
1 — No star history analysis.

#### B. Inflation Evidence (0.3)
Was specific evidence of star inflation identified?

5 — Specific red flags cited: unnatural spike timing, accounts with no other activity, geographic clustering, similar account creation dates.
4 — 2-3 specific indicators identified.
3 — 1 specific indicator identified.
2 — Vague inflation claims without evidence.
1 — No inflation evidence.

#### C. Fork Analysis (0.25)
Was fork analysis also performed?

5 — Fork count and growth pattern analyzed; empty/inactive forks identified as additional evidence.
4 — Forks analyzed but less thoroughly than stars.
3 — Forks mentioned but not analyzed.
2 — Fork analysis described without data.
1 — No fork analysis.

#### D. Report Quality (0.1)
Is the report well-organized and evidence-based?

5 — Structured report with timeline, evidence, and conclusion; data cited.
4 — Good report but organization could be clearer.
3 — Information present but loosely organized.
2 — Disorganized or hard to follow.
1 — No report structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "star_history_analysis": <1-5>,
  "inflation_evidence": <1-5>,
  "fork_analysis": <1-5>,
  "report_quality": <1-5>,
  "dimension_reasoning": {{
    "star_history_analysis": "<one sentence citing specific evidence>",
    "inflation_evidence": "<one sentence citing specific evidence>",
    "fork_analysis": "<one sentence citing specific evidence>",
    "report_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "star_history_analysis": 0.35,
    "inflation_evidence": 0.3,
    "fork_analysis": 0.25,
    "report_quality": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())