"""
LLM-as-judge evaluator for EvolveBench task-102.

Category: Legal
Task: Search USPTO and Google Patents for on-device LLM inference patents by Apple,
      Samsung, and Google; cross-reference forward citations and family members.
      Produce a competitive IP landscape table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to the USPTO Patent Full-Text and Image Database or Google Patents and search for patents "
    "filed in the last three years by Apple, Samsung, and Google related to 'on-device large language "
    "model inference.' For each company, identify the two most relevant patents and record: patent "
    "number, title, filing date, key independent claim (claim 1), and cited prior art. Then "
    "cross-reference each patent on Google Patents to check forward citations and any patent family "
    "members. Produce a competitive IP landscape table with columns: Company, Patent Number, Title, "
    "Filing Date, Key Claim Summary, Forward Citations, Family Size."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a patent research task using USPTO and Google Patents databases.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Must use USPTO patent database AND Google Patents (both required)
- Companies: Apple, Samsung, and Google specifically — all three required
- Per company: Two patents minimum
- Patent details required: patent number (US format), title, filing date, key claim 1 summary, prior art
- Cross-reference: forward citations and patent family size from Google Patents
- Output: Structured table with all seven specified columns for all six patents (2 per company)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate USPTO and/or Google Patents and retrieve real patent data?
- Are patents found for all three companies (Apple, Samsung, Google)?
- Are patent numbers in plausible US patent format (e.g., US11234567B2)?
- Did the agent retrieve forward citation counts and family sizes?
- Is there a structured table with the required columns?

### Step 2: Dimension Scoring

#### A. Patent Database Navigation
Did the agent navigate USPTO and Google Patents and retrieve real patent records?

5 — Agent navigated both USPTO and Google Patents, retrieved actual patent records with real patent numbers and details.
4 — Agent navigated one platform thoroughly and the other partially.
3 — Agent retrieved patents from at least one platform but records appear incomplete or partially fabricated.
2 — Agent searched but could not retrieve specific patents; response relies heavily on prior knowledge.
1 — No evidence of actual database navigation; patents are invented or generic.

#### B. Patent Data Completeness
Are all required fields present for all six patents (2 per company × 3 companies)?

5 — All six patents present with patent number, title, filing date, claim 1 summary, and prior art for all.
4 — Five of six patents fully detailed; one is missing a minor field.
3 — At least four patents with most required fields; notable gaps in one or two entries.
2 — Fewer than four patents with complete data, or companies are missing.
1 — Fewer than three patents found or missing one company entirely.

#### C. Forward Citations & Family Data
Did the agent cross-reference Google Patents for forward citations and family size?

5 — Forward citation count and patent family size retrieved from Google Patents for all or most patents.
4 — Cross-reference done for at least four patents with specific numbers reported.
3 — Cross-reference mentioned but data is vague (e.g., "several citations" without numbers) or covers fewer than four patents.
2 — Cross-reference attempted but no specific citation/family data retrieved.
1 — No cross-referencing done; forward citation and family data absent.

#### D. Output Structure & Analytical Value
Is the final output a complete, well-structured IP landscape table with analytical value?

5 — Complete table with all seven columns for all six patents, well-organized by company, with claim summaries that clearly describe the patented invention.
4 — Table present with minor structural gaps (one missing column or one incomplete entry).
3 — Table-like output with two or more missing columns or entries, or claim summaries are too vague.
2 — Narrative response with no table; information is unstructured.
1 — No meaningful structured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "patent_db_navigation": <1-5>,
  "patent_data_completeness": <1-5>,
  "forward_citation_data": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "patent_db_navigation": "<one sentence citing specific evidence>",
    "patent_data_completeness": "<one sentence citing specific evidence>",
    "forward_citation_data": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "patent_db_navigation":     0.25,
    "patent_data_completeness": 0.30,
    "forward_citation_data":    0.25,
    "output_structure":         0.20,
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
