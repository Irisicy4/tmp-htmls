"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Create a content calendar in Google Sheets for a social media campaign promoting digital art tools, using trends from Later.com and Pinterest Trends.
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


TASK_INSTRUCTION = """Set up a content calendar in Google Sheets for a social media campaign promoting digital art tools. Create columns for platform (Instagram, Facebook, Twitter), post date, content type (image, video, carousel), and hashtags. Populate 5 rows with sample data based on trends from Later.com and Pinterest Trends."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a content calendar in Google Sheets for a social media campaign promoting digital art tools. The calendar must include columns for platform, post date, content type, and hashtags, and must be populated with 5 rows of sample data. The sample data must be based on trends researched from Later.com and Pinterest Trends.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a content calendar in Google Sheets for a social media campaign promoting digital art tools. Create columns for platform (Instagram, Facebook, Twitter), post date, content type (image, video, carousel), and hashtags. Populate 5 rows with sample data based on trends from Later.com and Pinterest Trends.

## Task-Specific Constraints
- Must create a Google Sheet with the specified columns.
- Must populate the sheet with 5 rows of sample data.
- Must research trends from both Later.com and Pinterest Trends.
- Must include at least one unique hashtag per row based on the trends.
- Output must be structured as a table in the Google Sheet.
- Must include a variety of content types (e.g., image, video, carousel).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Later.com and Pinterest Trends to gather data?
- Did the agent create a Google Sheet with the specified columns?
- Are there exactly 5 rows of sample data in the output?
- Are the hashtags in the rows relevant to the trends researched?
- Does the output include a variety of content types (e.g., image, video, carousel)?

### Step 2: Dimension Scoring

#### A. Content Calendar Accuracy (0.35)
Measures whether the Google Sheet contains the correct structure and data.

5 — Contains all specified columns and 5 rows of accurate, trend-based data.
4 — Contains all specified columns and 5 rows, but some data is not trend-based.
3 — Contains most specified columns and rows, but with incomplete or inaccurate data.
2 — Contains some specified columns or rows, but data is mostly missing or incorrect.
1 — Contains no relevant structure or data.

#### B. Research Coverage (0.30)
Measures whether the agent researched trends from both Later.com and Pinterest Trends.

5 — Clearly used both platforms and incorporated trends into all rows.
4 — Used both platforms but trends are only partially incorporated.
3 — Used one platform and partially incorporated trends.
2 — Attempted research but trends are not incorporated.
1 — Did not use any platforms or incorporate trends.

#### C. Content Variety (0.20)
Measures whether the content types in the rows are diverse.

5 — Includes all three content types (image, video, carousel).
4 — Includes two content types, with some variety.
3 — Includes only one content type, but it is consistent across rows.
2 — Content type is unclear or inconsistent.
1 — No content types are specified.

#### D. Formatting and Organization (0.15)
Measures whether the output is well-structured and easy to interpret.

5 — Output is fully structured as a clear, organized table.
4 — Output is mostly structured, with minor formatting issues.
3 — Output is partially structured but difficult to interpret.
2 — Output is poorly structured or disorganized.
1 — Output is not structured as a table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "content_calendar_accuracy": <1-5>,
  "research_coverage": <1-5>,
  "content_variety": <1-5>,
  "formatting_and_organization": <1-5>,
  "dimension_reasoning": {{
    "content_calendar_accuracy": "<one sentence citing specific evidence>",
    "research_coverage": "<one sentence citing specific evidence>",
    "content_variety": "<one sentence citing specific evidence>",
    "formatting_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "content_calendar_accuracy": 0.35,
    "research_coverage": 0.30,
    "content_variety": 0.20,
    "formatting_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())