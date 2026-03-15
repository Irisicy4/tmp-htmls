"""
LLM-as-judge evaluator for EvolveBench task-44.

Category: Data & ML Engineering
Task: Go to https://www.betman.co.kr/, click on game purchase hit results,
      set the period from January 1 2025 to the current date, set the game
      type to soccer toto win/draw/loss, search, and save the results to
      Google Sheets.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to https://www.betman.co.kr/, click on game purchase hit results, "
    "set the period from January 1, 2025 to the current date, set the game "
    "type to soccer toto win/draw/loss, search, and save the results to Google Sheets."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a web data extraction and spreadsheet saving task.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: betman.co.kr specifically — the agent must navigate this site
- Section: "game purchase hit results" (게임구매 적중결과) — correct section must be selected
- Date filter: January 1, 2025 to the current date — both date bounds required
- Game type filter: soccer toto win/draw/loss (축구토토 승무패) — correct game type required
- Output: results must be saved to Google Sheets (not just displayed or saved locally)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to betman.co.kr? Cite evidence.
- Did the agent locate and click the correct "game purchase hit results" section?
- Were both date filters set correctly (Jan 1, 2025 to current date)?
- Was the game type filter set to soccer toto win/draw/loss?
- Was data successfully extracted and saved to Google Sheets?

### Step 2: Dimension Scoring

#### A. Site Navigation & Section Selection
Did the agent navigate betman.co.kr and reach the correct section?

5 — Agent navigated betman.co.kr and reached the game purchase hit results section.
4 — Agent reached betman.co.kr but had difficulty finding the correct section.
3 — Agent reached the site but landed on a different section.
2 — Agent attempted to navigate but failed to load the site or find the section.
1 — No navigation to betman.co.kr; response is from prior knowledge or hallucinated.

#### B. Filter Configuration
Did the agent correctly apply all required filters?

5 — Both date range (Jan 1 2025 to current) and game type (soccer toto win/draw/loss) correctly set.
4 — One filter correctly set; the other is approximate or partially correct.
3 — Filters applied but one is clearly wrong (e.g. wrong date range or wrong game type).
2 — Agent attempted to set filters but could not interact with the form correctly.
1 — No filter configuration attempted.

#### C. Data Extraction Quality
Was the search executed and data successfully retrieved?

5 — Search executed, results retrieved, and data is clearly structured with match results.
4 — Search executed and results retrieved but data is incomplete or poorly structured.
3 — Search attempted but only partial results retrieved or agent encountered errors.
2 — Agent described the data without actually extracting it.
1 — No data extracted.

#### D. Google Sheets Save
Was the extracted data saved to Google Sheets?

5 — Data saved to a new or existing Google Sheet with clear confirmation (sheet name, URL, or screenshot evidence).
4 — Data pasted into Google Sheets but without clear confirmation or with minor issues.
3 — Agent attempted to save to Google Sheets but encountered errors; partial save.
2 — Data saved locally (CSV/file) instead of Google Sheets.
1 — No saving attempted.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "site_navigation": <1-5>,
  "filter_configuration": <1-5>,
  "data_extraction_quality": <1-5>,
  "google_sheets_save": <1-5>,
  "dimension_reasoning": {{
    "site_navigation": "<one sentence citing specific evidence>",
    "filter_configuration": "<one sentence citing specific evidence>",
    "data_extraction_quality": "<one sentence citing specific evidence>",
    "google_sheets_save": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "site_navigation":        0.20,
    "filter_configuration":   0.30,
    "data_extraction_quality": 0.25,
    "google_sheets_save":     0.25,
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