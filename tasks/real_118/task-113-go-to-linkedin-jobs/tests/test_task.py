"""
LLM-as-judge evaluator for EvolveBench task-113.

Category: HR & Recruiting
Task: Search LinkedIn Jobs for Head of Product roles at SF startups, cross-reference
      Glassdoor ratings and Crunchbase funding data. Produce an opportunity report.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to LinkedIn Jobs and search for 'Head of Product' roles at Series B or later startups "
    "in San Francisco, CA posted in the last 30 days. Identify five distinct job postings and for "
    "each record: company name, job title, location (remote/hybrid/on-site), and any listed salary "
    "range. Then go to Glassdoor and look up the company rating, culture & values score, and CEO "
    "approval rating for each company. Finally, visit Crunchbase and find each company's funding "
    "stage, total funding raised, and most recent funding round date. Compile a Head of Product "
    "opportunity report table with all specified columns plus a 'priority tier' (High/Medium/Low) "
    "based on growth signals."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a multi-source recruiting research task combining LinkedIn Jobs, Glassdoor, and Crunchbase.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: LinkedIn Jobs, Glassdoor, AND Crunchbase — all three required
- Jobs: Five distinct "Head of Product" postings, SF area, Series B+, last 30 days
- LinkedIn fields: company name, job title, work mode (remote/hybrid/on-site), salary range (if listed)
- Glassdoor fields: overall company rating (out of 5), culture & values score, CEO approval %
- Crunchbase fields: funding stage, total funding raised, most recent round date
- Output: Table with all ten columns plus priority tier (High/Medium/Low)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate LinkedIn Jobs and find SF Head of Product postings from startups?
- Are five distinct company names identified with job details?
- Did the agent look up Glassdoor data for the companies?
- Did the agent look up Crunchbase funding data for the companies?
- Is there a complete table with priority tier assignments?

### Step 2: Dimension Scoring

#### A. LinkedIn Jobs Search
Did the agent navigate LinkedIn Jobs and find real SF startup Head of Product postings?

5 — Navigated LinkedIn Jobs, applied correct filters (title, location, date), and found five distinct postings from named companies with company names, work mode, and salary (or noted "not listed").
4 — Found four or five postings with most fields; one company is missing work mode or another field.
3 — Found at least three postings with company names; two postings are incomplete or filters partially applied.
2 — Found fewer than three specific postings; companies are generic or not clearly from recent LinkedIn search.
1 — No LinkedIn navigation; companies and job postings are fabricated.

#### B. Glassdoor Data Retrieval
Did the agent look up Glassdoor ratings for all five companies?

5 — Glassdoor overall rating, culture & values score, and CEO approval % retrieved for all five companies.
4 — Glassdoor data found for four of five companies; one is missing one score.
3 — Glassdoor data (at least overall rating) found for three or four companies.
2 — Glassdoor referenced for fewer than three companies; most data is estimated.
1 — Glassdoor not used; ratings absent or fabricated.

#### C. Crunchbase Funding Data
Did the agent retrieve funding data from Crunchbase for all five companies?

5 — Crunchbase consulted; funding stage, total funding, and most recent round date retrieved for all five companies.
4 — Crunchbase data found for four of five companies; one is missing the round date or total funding.
3 — Crunchbase data found for at least three companies; common gaps are total funding or round dates.
2 — Crunchbase referenced for fewer than three companies; funding data mostly estimated.
1 — Crunchbase not used; all funding data is from prior knowledge or fabricated.

#### D. Output Table & Priority Tier
Is the final report table complete with all columns and meaningful priority tier assignments?

5 — Complete ten-column table with priority tier (High/Medium/Low) for all five companies; tiers are logically assigned based on stated criteria (funding stage, Glassdoor score, growth signals).
4 — Table with nine of ten columns; priority tiers present for all companies but one tier is not justified.
3 — Table with eight or fewer columns; priority tiers present but criteria not explained.
2 — Table present with fewer than seven columns; no priority tier column.
1 — No table; narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "linkedin_search": <1-5>,
  "glassdoor_data": <1-5>,
  "crunchbase_data": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "linkedin_search": "<one sentence citing specific evidence>",
    "glassdoor_data": "<one sentence citing specific evidence>",
    "crunchbase_data": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "linkedin_search": 0.30,
    "glassdoor_data":  0.25,
    "crunchbase_data": 0.25,
    "output_table":    0.20,
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
