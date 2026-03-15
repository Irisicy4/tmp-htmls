"""
LLM-as-judge evaluator for EvolveBench task-15.

Category: Data & ML Engineering
Task: Get all 2025 publication info from https://m.douban.com/subject_collection/ECNA7Y7GA
      and save as an Excel file.
"""

import os, json, re

TASK_INSTRUCTION = "Help me get all 2025 publication info from https://m.douban.com/subject_collection/ECNA7Y7GA and save as an Excel file."
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully scraped publication data from a specific Douban URL and saved it to Excel.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Source: the exact URL https://m.douban.com/subject_collection/ECNA7Y7GA (must be accessed directly, not a general Douban search)
- Filter: 2025 publications only
- Output format: Excel file (.xlsx or .xls)
- Data expected: publication titles, authors, and any available metadata (rating, date, genre); at minimum title and year

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate the specific Douban URL? Cite evidence.
- How many publications did the agent retrieve? Are they filtered to 2025?
- What fields were captured (title, author, date, rating, etc.)?
- Was an Excel file created? Is there evidence of file creation (path, confirmation)?
- Did the agent handle pagination to get all entries, or only the first page?

### Step 2: Dimension Scoring

#### A. Source Navigation
Did the agent access the specific Douban URL as instructed?

5 — Agent navigated the exact URL and retrieved content from that specific collection page.
4 — Agent accessed the URL but may have only retrieved the first page without paginating.
3 — Agent searched Douban generally without accessing the specific collection URL.
2 — Agent used a different source or described what to do without navigating.
1 — No navigation performed; response is fabricated or refused.

#### B. Data Completeness
Did the agent retrieve all 2025 publications with adequate fields?

5 — All 2025 entries retrieved with at minimum title and publication date; additional fields (author, rating) are a bonus.
4 — Most 2025 entries retrieved; pagination may have missed some (e.g. only first 2 pages of 5).
3 — Some 2025 entries retrieved but coverage is clearly incomplete (e.g. only 10-20 entries when more exist).
2 — Data retrieved but not filtered to 2025, or very few entries captured.
1 — No data retrieved or data is fabricated.

#### C. Excel Output
Was the data saved as an Excel file?

5 — Excel file created and confirmed (.xlsx/.xls); file path or download link provided.
4 — Excel creation attempted and likely succeeded per trace; no explicit confirmation in response.
3 — Agent produced data in a tabular format (e.g. CSV or JSON) but not Excel specifically.
2 — Agent displayed the data in the response without saving to any file format.
1 — No file output; agent only described the data or refused.

#### D. Data Quality
Is the extracted data clean, structured, and accurate?

5 — Data is well-structured in Excel with clear column headers; entries are accurate and consistently formatted.
4 — Data is structured but minor formatting issues (e.g. inconsistent date formats, missing some fields for a few entries).
3 — Data is present but poorly structured or requires significant cleaning.
2 — Data is raw/unprocessed text dumped into Excel without proper column structure.
1 — No structured data output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "source_navigation": <1-5>,
  "data_completeness": <1-5>,
  "excel_output": <1-5>,
  "data_quality": <1-5>,
  "dimension_reasoning": {{
    "source_navigation": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "excel_output": "<one sentence citing specific evidence>",
    "data_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "source_navigation": 0.20,
    "data_completeness": 0.30,
    "excel_output":      0.30,
    "data_quality":      0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse_answer_tag(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
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
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }