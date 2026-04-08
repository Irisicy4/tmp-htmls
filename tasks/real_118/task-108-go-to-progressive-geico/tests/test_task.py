"""
LLM-as-judge evaluator for EvolveBench task-108.

Category: Insurance & Actuarial
Task: Get auto insurance quotes from Progressive, Geico, and State Farm for a Chicago
      driver, look up NAIC complaint index scores, and produce a value analysis table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to Progressive, Geico, and State Farm and use each insurer's online quote tool to get "
    "an estimate for a 30-year-old male driver in Chicago, IL with a clean driving record, driving "
    "a 2022 Toyota Camry, seeking 100/300/100 liability coverage plus comprehensive and collision "
    "with a $500 deductible. Record the quoted annual premium from each insurer. Then visit the "
    "NAIC Consumer Insurance Search tool and look up each insurer's complaint index score for "
    "private passenger auto insurance in Illinois. Produce an auto insurance value analysis table "
    "with columns: Insurer, Annual Premium, Complaint Index, Coverage Details, and a 'value score'."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed an auto insurance quote comparison task using Progressive, Geico, State Farm, and the NAIC complaint database.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Insurers: Progressive, Geico, AND State Farm — all three required
- Quote parameters: 30-year-old male, Chicago IL, clean record, 2022 Toyota Camry, 100/300/100 liability, comp+collision, $500 deductible
- NAIC: complaint index scores for private passenger auto in Illinois (eapps.naic.org/cis/)
- Output columns: Insurer, Annual Premium, Complaint Index, Coverage Details, Value Score
- Value score: some explicit method comparing premium vs. complaint index (lower combined is better)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent attempt to get quotes from all three insurers (Progressive, Geico, State Farm)?
- Are annual premium amounts reported for each insurer?
- Did the agent access the NAIC consumer search tool for complaint index scores?
- Is there a comparison table with all five columns?

### Step 2: Dimension Scoring

#### A. Quote Retrieval Attempts
Did the agent navigate all three insurer quote tools and attempt to retrieve premiums?

5 — Agent navigated all three insurer sites and retrieved quoted annual premiums (or explained when quote tools redirected/required personal info that blocked completion).
4 — Agent successfully retrieved quotes from two of three insurers; third was attempted.
3 — Agent retrieved quotes from at least two insurers; one insurer was not attempted.
2 — Only one insurer's quote tool was used; others are listed with estimated or fabricated premiums.
1 — No insurer quote tool navigation; all premiums are from prior knowledge or fabricated.

#### B. Quote Parameter Accuracy
Are the correct coverage parameters applied (100/300/100, comp+collision, $500 deductible)?

5 — All parameters applied correctly for all quotes: correct coverage limits (100/300/100), comprehensive and collision, $500 deductible.
4 — Parameters correct for most quotes; one parameter slightly off (e.g., different deductible).
3 — Coverage type correct (full coverage) but specific limits or deductible not confirmed or different.
2 — Generic quote obtained without confirming coverage parameters.
1 — Coverage parameters not applied; quotes are generic or not tied to specified coverage.

#### C. NAIC Complaint Index Lookup
Did the agent access the NAIC consumer search and retrieve complaint index scores?

5 — Navigated NAIC eapps.naic.org/cis/, found complaint index scores for all three insurers for private passenger auto in Illinois, and reported specific numbers.
4 — Found complaint index scores for two of three insurers from NAIC.
3 — NAIC referenced and complaint index mentioned; scores found for at least one insurer or scores appear plausible but not clearly from NAIC.
2 — NAIC mentioned but no specific complaint index scores retrieved; values estimated.
1 — NAIC not used; complaint index absent or fabricated.

#### D. Output Table & Value Score
Is the comparison table complete and is the value score methodology clear?

5 — Complete five-column table (Insurer, Premium, Complaint Index, Coverage Details, Value Score) with an explained value score method (e.g., normalized score or premium/complaint ratio).
4 — Table present with four columns; value score present but methodology not explained.
3 — Table present with three columns; value score generic or missing.
2 — Narrative response with comparison but no structured table.
1 — No table and no value score.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "quote_retrieval": <1-5>,
  "quote_parameters": <1-5>,
  "naic_lookup": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "quote_retrieval": "<one sentence citing specific evidence>",
    "quote_parameters": "<one sentence citing specific evidence>",
    "naic_lookup": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "quote_retrieval":   0.30,
    "quote_parameters":  0.20,
    "naic_lookup":       0.30,
    "output_table":      0.20,
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
