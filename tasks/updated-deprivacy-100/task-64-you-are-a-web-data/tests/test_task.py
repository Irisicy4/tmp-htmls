"""
LLM-as-judge evaluator for EvolveBench task-64.

Category: Data & ML Engineering
Task: You are a web data extraction agent. Go to https://www.jalan.net/en/japan_hotels_ryokan/hokkaido/ and extract all accommodation entries listed on the page. For each entry, extract the hotel name and detail URL, deduplicate, and output as a CSV file with columns: hotel_name, hotel_url.
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


TASK_INSTRUCTION = """You are a web data extraction agent. Go to https://www.jalan.net/en/japan_hotels_ryokan/hokkaido/ and extract all accommodation entries listed on the page. For each entry, extract the hotel name and detail URL, deduplicate, and output as a CSV file with columns: hotel_name, hotel_url."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves web scraping a Japanese hotel listing page on jalan.net and outputting structured CSV data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must visit the exact URL provided (jalan.net Hokkaido hotel listings)
- Fields: hotel_name and hotel_url — both required for every entry
- Deduplication: duplicate entries must be removed
- Output: valid CSV file with correct column headers

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the correct URL?
- How many hotel entries were extracted?
- Are both hotel_name and hotel_url present for each entry?
- Was deduplication performed?
- Was a CSV file produced?

### Step 2: Dimension Scoring

#### A. Site Navigation (0.15)
Did the agent navigate to the correct URL?

5 — Agent visited the exact URL and loaded the hotel listing page.
4 — Agent reached the site but had pagination or loading issues.
3 — Agent found similar content via search rather than direct URL.
2 — Agent visited wrong section of the site.
1 — No navigation to the target URL.

#### B. Extraction Completeness (0.35)
Were all hotel entries extracted?

5 — All visible hotel entries extracted with both name and URL for each.
4 — Most entries extracted but a few missing.
3 — Partial extraction (less than 75% of entries).
2 — Only a few sample entries extracted.
1 — No extraction performed.

#### C. Data Accuracy (0.3)
Are hotel names and URLs accurate?

5 — Names match the page content; URLs are valid and point to correct detail pages.
4 — Names correct; URLs have minor issues (relative vs absolute).
3 — Names mostly correct but some garbled; URLs partially correct.
2 — Significant accuracy issues.
1 — Data is fabricated or completely wrong.

#### D. Csv Output (0.2)
Was valid CSV produced with correct columns?

5 — Valid CSV with hotel_name and hotel_url headers, one row per hotel, no duplicates.
4 — CSV produced but minor issues (e.g. extra columns, inconsistent quoting).
3 — CSV-like output but not proper format.
2 — Data in table or JSON instead of CSV.
1 — No file output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "site_navigation": <1-5>,
  "extraction_completeness": <1-5>,
  "data_accuracy": <1-5>,
  "csv_output": <1-5>,
  "dimension_reasoning": {{
    "site_navigation": "<one sentence citing specific evidence>",
    "extraction_completeness": "<one sentence citing specific evidence>",
    "data_accuracy": "<one sentence citing specific evidence>",
    "csv_output": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "site_navigation": 0.15,
    "extraction_completeness": 0.35,
    "data_accuracy": 0.3,
    "csv_output": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())