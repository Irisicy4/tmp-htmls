"""
LLM-as-judge evaluator for EvolveBench task-114.

Category: HR & Recruiting
Task: Look up top H-1B sponsors on USCIS Employer Data Hub, cross-reference careers pages
      and Glassdoor salaries, and produce a visa-sponsorship employer comparison table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to the USCIS H-1B Employer Data Hub and look up the top five technology companies by "
    "H-1B petition approvals in fiscal year 2023. Record each company's name, total approvals, "
    "total denials, and denial rate. Then visit the careers page and Glassdoor profile for each "
    "of those five companies and note: number of open software engineering roles (as of today), "
    "average Glassdoor salary for Software Engineer, and whether the company explicitly mentions "
    "visa sponsorship in their job postings. Produce a visa-sponsorship employer comparison table "
    "with all specified columns including an 'H-1B friendliness score' (1-5)."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed an H-1B employer research task combining USCIS data, company careers pages, and Glassdoor.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: USCIS H-1B Employer Data Hub, company careers pages, AND Glassdoor — all three required
- USCIS: Top 5 tech companies by H-1B approvals in FY2023; must use the actual data hub
- Per company: approvals count, denials count, denial rate (calculated as denials/petitions)
- Careers page: number of open SWE roles (current count)
- Glassdoor: average SWE salary + visa sponsorship mentioned in job postings (yes/no)
- Output: Seven-column table with H-1B friendliness score (1-5) based on approval volume and denial rate

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate the USCIS H-1B Employer Data Hub and retrieve FY2023 data?
- Are the top 5 tech companies plausible H-1B sponsors (e.g., Amazon, Infosys, TCS, Cognizant, Wipro, Google, Microsoft)?
- Did the agent visit company careers pages or Glassdoor for SWE data?
- Is there a complete table with all seven columns and a friendliness score?

### Step 2: Dimension Scoring

#### A. USCIS Data Hub Navigation
Did the agent navigate the USCIS H-1B Employer Data Hub and retrieve FY2023 petition data?

5 — Navigated USCIS Employer Data Hub, found top 5 tech companies by FY2023 approvals with specific approval and denial counts.
4 — Navigated USCIS Data Hub but retrieved data for four of five companies, or used FY2022 data instead of FY2023.
3 — Referenced USCIS data with plausible company names but counts appear approximated or from secondary sources.
2 — Company names are plausible H-1B sponsors but denial rates and approval counts appear fabricated or estimated.
1 — No USCIS navigation; companies and data are entirely from prior knowledge.

#### B. H-1B Data Accuracy
Are approval counts, denial counts, and denial rates specific and internally consistent?

5 — All five companies have specific approval counts, denial counts, and a calculated denial rate (denials/total petitions) that is internally consistent.
4 — Four of five companies have complete and consistent data; one is missing a denial count.
3 — All five companies named with approximate approval counts; denial rates estimated rather than calculated.
2 — Only two or three companies have specific counts; others are generic.
1 — No specific petition data; all figures are generic or fabricated.

#### C. Careers & Glassdoor Research
Did the agent visit careers pages and Glassdoor for current SWE data?

5 — Careers pages and/or Glassdoor visited for all five companies; open SWE role count, average SWE salary, and visa sponsorship mention (yes/no) reported for each.
4 — Careers/Glassdoor data found for four of five companies; one is missing one field.
3 — Data found for at least three companies; open SWE role count or visa sponsorship commonly missing.
2 — Glassdoor salaries cited for two or fewer companies; no careers page data.
1 — No careers page or Glassdoor navigation; all salary and sponsorship data is fabricated.

#### D. Output Table & Friendliness Score
Is the comparison table complete and are H-1B friendliness scores logical?

5 — Complete seven-column table for all five companies with H-1B friendliness scores (1-5) that are logically derived from approval volume and denial rate (higher approvals + lower denial rate = higher score).
4 — Table with six columns and friendliness scores present; scoring rationale implied but not stated.
3 — Table with five columns; friendliness scores present but not tied to specific USCIS metrics.
2 — Partial table with fewer than five columns; no friendliness score.
1 — No table; narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "uscis_navigation": <1-5>,
  "h1b_data_accuracy": <1-5>,
  "careers_glassdoor_research": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "uscis_navigation": "<one sentence citing specific evidence>",
    "h1b_data_accuracy": "<one sentence citing specific evidence>",
    "careers_glassdoor_research": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "uscis_navigation":          0.30,
    "h1b_data_accuracy":         0.25,
    "careers_glassdoor_research": 0.25,
    "output_table":              0.20,
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
