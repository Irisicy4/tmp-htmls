"""
LLM-as-judge evaluator for EvolveBench task-97.

Category: Data & ML Engineering
Task: Analyze the following student profile data and generate a structured report covering political affiliation, English prof
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


TASK_INSTRUCTION = """Analyze the following student profile data and generate a structured report covering political affiliation, English proficiency exam results, other certification exam results, graduate school recommendations, scholarships and honors, arts and sports, academic competitions, research output, student leadership roles, volunteer hours, and social practice activities."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves analyzing structured student profile data (provided as input) and generating a comprehensive structured report covering all specified categories.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Input: student profile data must be provided (structured data with multiple fields)
- Coverage: all 11 categories must be addressed
- Output: structured report (not just raw data)
- Analysis: each category should have an analytical summary, not just listed facts

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent analyze actual student profile data?
- How many of the 11 categories are covered?
- Is the output structured with clear sections?
- Does each section provide analysis beyond raw data?
- Is the report professionally formatted?

### Step 2: Dimension Scoring

#### A. Data Processing (0.2)
Did the agent process actual student profile data?

5 — Agent clearly read and processed the student data; specific data points referenced.
4 — Data processed but some fields missed.
3 — Data partially processed.
2 — Generic template filled without specific student data.
1 — No data processing.

#### B. Category Coverage (0.35)
How many of the 11 categories are covered?

5 — All 11 categories addressed.
4 — 9-10 categories.
3 — 7-8 categories.
2 — 4-6 categories.
1 — Fewer than 4 categories.

#### C. Analytical Depth (0.3)
Is each category analyzed rather than just listed?

5 — Each category has interpretation: strengths, gaps, comparison to standards, recommendations.
4 — Most categories analyzed; some just listed.
3 — About half analyzed; rest listed.
2 — Mostly raw data without analysis.
1 — No analysis.

#### D. Report Structure (0.15)
Is the report well-structured and professional?

5 — Clear headers for each category, executive summary, consistent formatting.
4 — Good structure but missing summary or inconsistent.
3 — Sections present but formatting rough.
2 — Wall of text.
1 — No structure.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_processing": <1-5>,
  "category_coverage": <1-5>,
  "analytical_depth": <1-5>,
  "report_structure": <1-5>,
  "dimension_reasoning": {{
    "data_processing": "<one sentence citing specific evidence>",
    "category_coverage": "<one sentence citing specific evidence>",
    "analytical_depth": "<one sentence citing specific evidence>",
    "report_structure": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_processing": 0.2,
    "category_coverage": 0.35,
    "analytical_depth": 0.3,
    "report_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())