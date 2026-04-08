"""
LLM-as-judge evaluator for EvolveBench task-115.

Category: HR & Recruiting
Task: Look up BLS OOH data for ML Engineer, Data Scientist, and AI Research Scientist;
      cross-reference LinkedIn job counts and Levels.fyi compensation at Google/Meta/Amazon.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to the Bureau of Labor Statistics Occupational Outlook Handbook and look up the job "
    "outlook, median annual wage, and projected 10-year growth rate for three roles: Machine "
    "Learning Engineer, Data Scientist, and AI Research Scientist. Then go to LinkedIn Jobs and "
    "note the number of open positions in the United States for each role. Finally, visit "
    "Levels.fyi and find the median total compensation (base + bonus + equity) at the L4/E4 "
    "equivalent level at Google, Meta, and Amazon for each of the three roles. Produce a talent "
    "market intelligence report table with all specified columns and a 'market heat' rating "
    "(Hot / Warm / Cooling) for each role."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a talent market research task combining BLS OOH, LinkedIn Jobs, and Levels.fyi data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: BLS OOH (bls.gov/ooh), LinkedIn Jobs, AND Levels.fyi — all three required
- Roles: Machine Learning Engineer, Data Scientist, AND AI Research Scientist — all three
- BLS fields: median annual wage, projected 10-year growth %, job outlook description
- LinkedIn: US open position count (approximate is acceptable)
- Levels.fyi: median TC at L4/E4 equivalent at Google, Meta, AND Amazon for each role
- Output: Seven-column table with market heat rating (Hot/Warm/Cooling) per role

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate BLS OOH and find wage and growth data for the three roles?
- Did the agent search LinkedIn Jobs for US open position counts?
- Did the agent visit Levels.fyi and retrieve TC data for all three companies × three roles?
- Is there a complete seven-column table with market heat ratings?

### Step 2: Dimension Scoring

#### A. BLS OOH Data Retrieval
Did the agent navigate BLS OOH and retrieve median wage and growth data for all three roles?

5 — Navigated bls.gov/ooh, found median annual wage and projected 10-year growth rate for all three roles with specific figures.
4 — BLS data found for two of three roles with specific figures; third role approximated or matched to nearest BLS occupation.
3 — BLS data referenced for all three roles but one or two figures are approximate (e.g., "around $100k") or matched loosely.
2 — BLS OOH mentioned but data appears to be from prior knowledge rather than current navigation.
1 — No BLS navigation; wages and growth rates are fabricated or entirely from prior knowledge.

#### B. LinkedIn Open Position Count
Did the agent search LinkedIn Jobs and report US open position counts for each role?

5 — LinkedIn Jobs searched for all three roles in the United States; specific position counts (e.g., "12,453 results") reported for each.
4 — LinkedIn counts found for two of three roles; third role count is estimated.
3 — LinkedIn referenced for all three roles but counts are approximate or clearly dated (e.g., not from a current search).
2 — LinkedIn counts cited for one role or all counts are generic round numbers without search evidence.
1 — LinkedIn Jobs not searched; position counts absent or fabricated.

#### C. Levels.fyi Compensation Data
Did the agent retrieve TC data from Levels.fyi for all three companies and all three roles?

5 — Levels.fyi consulted; median TC (base + bonus + equity) at L4/E4 equivalent found at Google, Meta, AND Amazon for all three roles (nine data points).
4 — Levels.fyi data found for seven or eight of nine combinations; one or two company-role pairs are missing.
3 — Levels.fyi data found for at least six combinations; three missing.
2 — Levels.fyi data found for fewer than six combinations or data appears fabricated.
1 — Levels.fyi not used; all compensation figures are from prior knowledge.

#### D. Output Table & Market Heat Rating
Is the talent market intelligence table complete and are market heat ratings logical?

5 — Complete seven-column table (Role, BLS Median Wage, Growth Rate, LinkedIn Positions, Google TC, Meta TC, Amazon TC) plus market heat rating (Hot/Warm/Cooling) for all three roles, with ratings justified by the data.
4 — Table with six columns; market heat ratings present for all three roles.
3 — Table with five columns; or one role is missing from the table.
2 — Partial table with fewer than five columns; no market heat rating.
1 — No structured table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "bls_data_retrieval": <1-5>,
  "linkedin_position_count": <1-5>,
  "levelsfyi_compensation": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "bls_data_retrieval": "<one sentence citing specific evidence>",
    "linkedin_position_count": "<one sentence citing specific evidence>",
    "levelsfyi_compensation": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "bls_data_retrieval":      0.25,
    "linkedin_position_count": 0.20,
    "levelsfyi_compensation":  0.30,
    "output_table":            0.25,
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
