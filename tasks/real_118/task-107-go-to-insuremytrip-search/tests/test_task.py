"""
LLM-as-judge evaluator for EvolveBench task-107.

Category: Insurance & Actuarial
Task: Compare travel insurance plans on InsureMyTrip for a Japan trip, then look up
      each insurer's AM Best financial strength rating. Produce a comparison table
      with a recommendation.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to InsureMyTrip and search for single-trip travel insurance for a 35-year-old traveler "
    "taking a 14-day trip to Japan, with a total trip cost of $4,000 and departure in 30 days. "
    "Compare the top three quoted plans and for each record: insurer name, plan name, premium, "
    "trip cancellation limit, medical evacuation limit, pre-existing condition waiver (yes/no), "
    "and COVID-19 coverage (yes/no). Then visit AM Best and look up the financial strength rating "
    "for each insurer. Compile a travel insurance comparison table with all specified columns "
    "and add a final recommendation row identifying the best value plan with justification."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a travel insurance comparison task using InsureMyTrip and AM Best.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: InsureMyTrip for quotes AND AM Best for insurer ratings (both required)
- Quote parameters: 35-year-old, 14-day Japan trip, $4,000 trip cost, departure in 30 days
- Number of plans: Three plans compared
- Required columns: Insurer, Plan Name, Premium, Trip Cancellation, Medical Evacuation, Pre-Existing Waiver, COVID Coverage, AM Best Rating
- Recommendation: Must identify best-value plan with one-sentence justification
- Output: Complete structured table plus recommendation row

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate InsureMyTrip and enter the correct trip parameters?
- Are three specific plan quotes retrieved with insurer names and premiums?
- Did the agent look up AM Best ratings for each insurer?
- Is there a complete comparison table with all eight columns?
- Is a recommendation present with justification?

### Step 2: Dimension Scoring

#### A. InsureMyTrip Quote Retrieval
Did the agent navigate InsureMyTrip with correct parameters and retrieve actual quotes?

5 — Navigated InsureMyTrip, entered all correct parameters (age 35, 14-day Japan trip, $4,000, 30-day departure), and retrieved three specific plan quotes with named insurers and premiums.
4 — Navigated InsureMyTrip and retrieved three quotes but one parameter was slightly off (e.g., approximate departure date).
3 — Navigated InsureMyTrip but retrieved fewer than three quotes, or parameters are clearly wrong.
2 — InsureMyTrip mentioned but quotes are generic or appear to be from prior knowledge rather than real navigation.
1 — No InsureMyTrip navigation; plans are fabricated or entirely from prior knowledge.

#### B. Plan Data Completeness
Are all required coverage fields present for all three plans?

5 — All three plans have premium, trip cancellation limit, medical evacuation limit, pre-existing waiver (yes/no), and COVID coverage (yes/no).
4 — Two of three plans have all fields; one is missing one field (typically COVID coverage or pre-existing waiver).
3 — All three plans have premium and trip cancellation; evacuation limit or boolean fields are missing for some.
2 — Fewer than three plans have most fields; critical fields like evacuation limit are missing.
1 — No specific plan data; generic coverage descriptions only.

#### C. AM Best Rating Lookup
Did the agent look up and report AM Best financial strength ratings for each insurer?

5 — AM Best rating (e.g., A+, A, A-) retrieved and reported for all three insurers.
4 — AM Best rating found for two of three insurers; one rating missing or unable to find.
3 — AM Best rating mentioned for insurers but ratings appear estimated or not from the AM Best site.
2 — AM Best mentioned as a source but no specific ratings retrieved.
1 — AM Best not used; insurer financial ratings absent.

#### D. Output Structure & Recommendation
Is the comparison table complete and is the recommendation specific and justified?

5 — Complete table with all eight columns for all three plans, plus a recommendation row naming the winning plan and a specific justification (e.g., "lowest premium with A+ rated insurer and full pre-existing waiver").
4 — Table has seven of eight columns; recommendation present but justification is generic.
3 — Table present with six or fewer columns; or recommendation is missing.
2 — Narrative response with plan info but no structured table.
1 — No table and no recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "quote_retrieval": <1-5>,
  "plan_data_completeness": <1-5>,
  "am_best_lookup": <1-5>,
  "output_and_recommendation": <1-5>,
  "dimension_reasoning": {{
    "quote_retrieval": "<one sentence citing specific evidence>",
    "plan_data_completeness": "<one sentence citing specific evidence>",
    "am_best_lookup": "<one sentence citing specific evidence>",
    "output_and_recommendation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "quote_retrieval":           0.30,
    "plan_data_completeness":    0.25,
    "am_best_lookup":            0.25,
    "output_and_recommendation": 0.20,
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
