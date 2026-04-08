"""
LLM-as-judge evaluator for EvolveBench task-83.

Category: Marketing & Analytics
Task: Analyze the engagement discrepancy in this content performance data. Diagnose breakouts, failures, and baselines, then e
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


TASK_INSTRUCTION = """Analyze the engagement discrepancy in this content performance data. Diagnose breakouts, failures, and baselines, then extract replicable formulas for future newsletters."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves analyzing newsletter or content performance data to identify what drove high-performing vs low-performing pieces, and extracting actionable patterns.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Input: content performance data (provided by user — open rates, click rates, engagement metrics)
- Analysis: must distinguish breakouts (top performers), failures (underperformers), and baseline
- Output: replicable formulas — specific patterns that can be replicated
- Depth: must go beyond averages to identify specific causal factors

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent analyze actual performance data?
- Were breakouts, failures, and baselines clearly identified?
- What specific patterns were found for each category?
- Are replicable formulas extracted with clear rationale?
- Is the output actionable for future newsletter creation?

### Step 2: Dimension Scoring

#### A. Data Analysis (0.3)
Did the agent analyze the actual performance data?

5 — Specific metrics analyzed: open rates, CTR, engagement; compared across content pieces with statistical framing.
4 — Good analysis but less precise numerically.
3 — General trends identified without piece-by-piece analysis.
2 — Described what to look for without analyzing actual data.
1 — No data analysis.

#### B. Pattern Identification (0.35)
Were breakouts, failures, and baselines clearly identified?

5 — Specific pieces categorized as breakout/failure/baseline with metrics to support each classification.
4 — Two of three categories clearly identified.
3 — One category clearly identified.
2 — Vague categorization without metric support.
1 — No categorization.

#### C. Formula Extraction (0.25)
Were replicable formulas extracted?

5 — Specific formulas stated: e.g. 'subject lines with numbers + urgency words outperform by X%'; actionable and testable.
4 — Good formulas but less specific or testable.
3 — Patterns identified but not stated as formulas.
2 — Generic best practices not grounded in the data.
1 — No formulas extracted.

#### D. Actionability (0.1)
Is the output directly usable for future newsletters?

5 — Formulas are ready to apply with clear next steps.
4 — Mostly actionable with minor gaps.
3 — Insights present but require interpretation to apply.
2 — Too abstract to directly apply.
1 — Not actionable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_analysis": <1-5>,
  "pattern_identification": <1-5>,
  "formula_extraction": <1-5>,
  "actionability": <1-5>,
  "dimension_reasoning": {{
    "data_analysis": "<one sentence citing specific evidence>",
    "pattern_identification": "<one sentence citing specific evidence>",
    "formula_extraction": "<one sentence citing specific evidence>",
    "actionability": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_analysis": 0.3,
    "pattern_identification": 0.35,
    "formula_extraction": 0.25,
    "actionability": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())