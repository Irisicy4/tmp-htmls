"""
LLM-as-judge evaluator for EvolveBench task-73.

Category: Data & ML Engineering
Task: You are a web data extraction agent. Go to https://deriheruhotel.com/hotel/index/hokkaido/sapporo-ch...
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


TASK_INSTRUCTION = """You are a web data extraction agent. Go to https://deriheruhotel.com/hotel/index/hokkaido/sapporo-chuo/3697/ and extract: (1) hotel-level fields including name, address, phone, hotel website, and booking URL, and (2) all paginated reviews including reviewer name, date, score, and review text. Output as structured JSON."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves scraping a specific Japanese hotel detail page including hotel metadata and all paginated reviews, outputting as structured JSON.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- URL: must visit the exact URL (specific hotel in Sapporo-Chuo area)
- Hotel fields: name, address, phone, hotel website, booking URL — all five required
- Reviews: must paginate through ALL reviews — not just first page
- Review fields: reviewer name, date, score, review text — all four required per review
- Output: valid JSON with both hotel_info and reviews sections

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the exact URL?
- Were all 5 hotel-level fields extracted?
- Were reviews paginated (how many pages/reviews found)?
- Were all 4 review fields captured per review?
- Was valid JSON produced?

### Step 2: Dimension Scoring

#### A. Hotel Fields (0.25)
Were all 5 hotel-level fields extracted?

5 — All five fields: name, address, phone, website, booking URL — all present and accurate.
4 — 4 of 5 fields present.
3 — 3 of 5 fields present.
2 — Only 1-2 fields.
1 — No hotel fields extracted.

#### B. Review Pagination (0.3)
Were all paginated reviews collected?

5 — Agent paginated through all review pages and collected all reviews.
4 — Multiple pages collected but not necessarily all.
3 — Only first page of reviews collected.
2 — Only 1-3 sample reviews collected.
1 — No reviews collected.

#### C. Review Field Completeness (0.25)
Were all 4 review fields captured per review?

5 — Reviewer name, date, score, and review text all present for each review.
4 — 3 of 4 fields present consistently.
3 — 2 of 4 fields present.
2 — Only score or text present.
1 — No review fields captured.

#### D. Json Output (0.2)
Was valid structured JSON produced?

5 — Valid JSON with hotel_info and reviews array; parseable without errors.
4 — JSON produced but minor formatting issues.
3 — JSON-like output but not valid JSON.
2 — Data in non-JSON format (CSV, table).
1 — No structured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "hotel_fields": <1-5>,
  "review_pagination": <1-5>,
  "review_field_completeness": <1-5>,
  "json_output": <1-5>,
  "dimension_reasoning": {{
    "hotel_fields": "<one sentence citing specific evidence>",
    "review_pagination": "<one sentence citing specific evidence>",
    "review_field_completeness": "<one sentence citing specific evidence>",
    "json_output": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "hotel_fields": 0.25,
    "review_pagination": 0.3,
    "review_field_completeness": 0.25,
    "json_output": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())