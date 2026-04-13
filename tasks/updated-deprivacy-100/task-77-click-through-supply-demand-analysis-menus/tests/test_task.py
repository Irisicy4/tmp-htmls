"""
LLM-as-judge evaluator for EvolveBench task-77.

Category: Finance & Economics
Task: Click through all the supply-demand analysis menus, analyze by institution and by foreign investor, and recommend stocks
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


TASK_INSTRUCTION = """Click through all the supply-demand analysis menus, analyze by institution and by foreign investor, and recommend stocks."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating a Korean stock market platform's supply-demand analysis features, reviewing institutional and foreign investor activity, and making stock recommendations based on the data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Korean stock market tool (likely HTS like Kiwoom, Shinhan, or a web platform)
- Coverage: must navigate both 'by institution' AND 'by foreign investor' menus
- Output: stock recommendations grounded in the supply-demand data
- Data: must be current (recent trading data)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate supply-demand analysis menus?
- Were both institutional and foreign investor tabs/sections covered?
- What specific stocks showed significant institutional or foreign buying?
- Were stock recommendations made with data backing?

### Step 2: Dimension Scoring

#### A. Menu Navigation (0.25)
Did the agent navigate the supply-demand analysis menus?

5 — Agent navigated to supply-demand analysis and accessed both institutional and foreign investor sections.
4 — Accessed one section but not the other.
3 — Found supply-demand analysis but navigation was incomplete.
2 — Described the menu structure without navigating.
1 — No navigation.

#### B. Data Extraction (0.3)
Was relevant trading data extracted?

5 — Specific stocks with institutional/foreign net buying amounts and dates extracted.
4 — Stock data found but less precise.
3 — General trends identified without specific stock data.
2 — Data described without being extracted.
1 — No data extracted.

#### C. Analysis Quality (0.3)
Was the data properly analyzed?

5 — Clear analysis: which stocks show consistent institutional accumulation vs distribution; foreign investor trends.
4 — Good analysis but one dimension (inst or foreign) weaker.
3 — Basic pattern identification without depth.
2 — Data presented without analysis.
1 — No analysis.

#### D. Recommendation Quality (0.15)
Were stock recommendations grounded in the data?

5 — Specific stock recommendations with ticker, reasoning from supply-demand data, and risk note.
4 — Recommendations made with some data backing.
3 — Recommendations made but thinly supported.
2 — Generic recommendations not tied to supply-demand data.
1 — No recommendations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "menu_navigation": <1-5>,
  "data_extraction": <1-5>,
  "analysis_quality": <1-5>,
  "recommendation_quality": <1-5>,
  "dimension_reasoning": {{
    "menu_navigation": "<one sentence citing specific evidence>",
    "data_extraction": "<one sentence citing specific evidence>",
    "analysis_quality": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "menu_navigation": 0.25,
    "data_extraction": 0.3,
    "analysis_quality": 0.3,
    "recommendation_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())