"""
LLM-as-judge evaluator for EvolveBench task-118.

Category: Medical & Clinical & Bio
Task: Query CDC WONDER for cardiovascular mortality rates by state, cross-reference with
      Commonwealth Fund health system performance scores. Compile a disparity analysis table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to CDC WONDER and use the Underlying Cause of Death database to query age-adjusted "
    "cardiovascular disease mortality rates (ICD-10 codes I00-I99) for all 50 US states for the "
    "most recent available year. Identify the five states with the highest and the five with the "
    "lowest age-adjusted cardiovascular mortality rates; record each state's rate per 100,000 "
    "population. Then go to the Commonwealth Fund's State Health System Performance scorecard "
    "and find each of those ten states' overall health system performance rank and their "
    "cardiovascular care rank specifically. Compile a cardiovascular mortality analysis table "
    "with all five specified columns including a 'disparity note' for states where CV mortality "
    "and health system rank diverge significantly."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a public health data analysis task combining CDC WONDER mortality data with Commonwealth Fund health system rankings.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: CDC WONDER (wonder.cdc.gov) AND Commonwealth Fund scorecard (commonwealthfund.org) — both required
- CDC WONDER: Underlying Cause of Death database; ICD-10 I00-I99; age-adjusted rates; all 50 states; most recent year
- States selected: Top 5 HIGHEST and bottom 5 LOWEST by age-adjusted CVD mortality (10 states total)
- Commonwealth Fund: Overall health system rank AND cardiovascular care rank for each of the 10 states
- Disparity note: Flag states where mortality rank and health system rank diverge significantly
- Output: Five-column table for all 10 states plus disparity notes

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate CDC WONDER and query CVD mortality (ICD-10 I00-I99) by state?
- Are age-adjusted rates per 100,000 population reported for 10 states (5 highest, 5 lowest)?
- Did the agent access the Commonwealth Fund scorecard for state health system ranks?
- Are overall rank and CV care rank reported for all 10 states?
- Is there a five-column table with disparity notes?

### Step 2: Dimension Scoring

#### A. CDC WONDER Navigation & Query
Did the agent navigate CDC WONDER and query cardiovascular mortality data by state?

5 — Navigated wonder.cdc.gov, used the Underlying Cause of Death database, specified ICD-10 I00-I99, grouped by state, and retrieved age-adjusted rates per 100,000 for the most recent available year.
4 — Navigated CDC WONDER and retrieved state-level CVD mortality data but ICD-10 specification or age-adjustment was not confirmed.
3 — Navigated CDC WONDER and retrieved mortality data but the query parameters (ICD codes or age-adjustment) are unclear or approximated.
2 — CDC WONDER mentioned; data appears to be from prior knowledge or secondary sources rather than a direct query.
1 — CDC WONDER not navigated; mortality rates are entirely from prior knowledge or fabricated.

#### B. State Selection Accuracy
Are the 5 highest and 5 lowest CVD mortality states correctly identified with specific rates?

5 — Five highest and five lowest states identified with specific age-adjusted rates per 100,000 (e.g., Mississippi ~380, Hawaii ~150); states are plausible given known regional health disparities.
4 — Nine of ten states identified correctly; one state appears misranked or the rate for one state is missing.
3 — States identified but rates are approximate or the top/bottom split is not clearly based on a ranked query.
2 — Ten states listed but the selection criteria (highest/lowest by CVD mortality) is not clearly applied.
1 — States are random or do not reflect CVD mortality extremes; rates absent or implausible.

#### C. Commonwealth Fund Scorecard Data
Did the agent access the Commonwealth Fund scorecard and retrieve ranks for all 10 states?

5 — Commonwealth Fund scorecard accessed; overall health system rank AND cardiovascular care rank found for all 10 states with specific rank numbers.
4 — Commonwealth Fund data found for eight or nine of ten states; one or two states are missing one rank.
3 — Commonwealth Fund data found for at least seven states; cardiovascular care rank specifically is less often available than overall rank.
2 — Commonwealth Fund referenced for fewer than seven states; most ranks estimated.
1 — Commonwealth Fund not used; all rank data is from prior knowledge or fabricated.

#### D. Output Table & Disparity Notes
Is the five-column table complete with specific disparity notes for divergent states?

5 — Complete five-column table (State, Age-Adjusted CVD Mortality Rate, Data Year, Commonwealth Overall Rank, CV Care Rank) for all 10 states; disparity notes present for at least two states with specific divergence described (e.g., "Oklahoma: high mortality rank 45 but overall health system rank 35 — moderate divergence").
4 — Table with all five columns; disparity notes present but one state that clearly diverges lacks a note.
3 — Table with four columns; or disparity notes are generic for all states.
2 — Table with three or fewer columns; no disparity notes.
1 — No structured table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "cdc_wonder_navigation": <1-5>,
  "state_selection_accuracy": <1-5>,
  "commonwealth_fund_data": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "cdc_wonder_navigation": "<one sentence citing specific evidence>",
    "state_selection_accuracy": "<one sentence citing specific evidence>",
    "commonwealth_fund_data": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "cdc_wonder_navigation":   0.30,
    "state_selection_accuracy": 0.25,
    "commonwealth_fund_data":  0.25,
    "output_table":            0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
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
    except Exception as e:
        return {"error": str(e)}

def _vote(votes):
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
