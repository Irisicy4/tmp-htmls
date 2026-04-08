"""
LLM-as-judge evaluator for EvolveBench task.

Category: Insurance & Actuarial
Task: Extract homeowners insurance quotes from three platforms and summarize findings in a table.
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


TASK_INSTRUCTION = """Go to Lemonade (https://www.lemonade.com/homeowners), State Farm (https://www.statefarm.com/insurance/home), and Allstate (https://www.allstate.com/home-insurance.aspx). Use their online tools to get a homeowners insurance quote for a single-family home in Austin, TX valued at $350,000 with a $1,000 deductible, covering standard perils. For each insurer, extract the quoted annual premium, coverage limit, and any additional features like water damage, liability, or personal property protection. Summarize your findings in a table with columns for Insurer, Premium, Coverage Limit, and Additional Features."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to visit three homeowners insurance platforms (Lemonade, State Farm, and Allstate) and extract quotes for a specific property in Austin, TX. The agent must provide the annual premium, coverage limit, and additional features for each insurer, summarizing the findings in a structured table.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Lemonade (https://www.lemonade.com/homeowners), State Farm (https://www.statefarm.com/insurance/home), and Allstate (https://www.allstate.com/home-insurance.aspx). Use their online tools to get a homeowners insurance quote for a single-family home in Austin, TX valued at $350,000 with a $1,000 deductible, covering standard perils. For each insurer, extract the quoted annual premium, coverage limit, and any additional features like water damage, liability, or personal property protection. Summarize your findings in a table with columns for Insurer, Premium, Coverage Limit, and Additional Features.

## Task-Specific Constraints
- Must visit all three specified platforms: Lemonade, State Farm, and Allstate.
- Must extract and include the annual premium for each insurer.
- Must extract and include the coverage limit for each insurer.
- Must list additional features like water damage, liability, or personal property protection.
- Output must be organized as a structured table with columns for Insurer, Premium, Coverage Limit, and Additional Features.
- Must ensure the quotes are for a single-family home in Austin, TX valued at $350,000 with a $1,000 deductible.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are the annual premiums, coverage limits, and additional features present in the response?
- Is the output organized as a structured table with the required columns?
- Are the quotes specific to the property details provided (Austin, TX, $350,000 value, $1,000 deductible)?
- Are the extracted values accurate and sourced from the platforms?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly extracted and summarized the required data.

5 — All required data (premium, coverage limit, additional features) is present and accurate for all three insurers.
4 — Data is present and accurate for at least two insurers; minor errors for the third.
3 — Data is partially complete or contains notable inaccuracies for one or more insurers.
2 — Data is mostly missing or incorrect for multiple insurers.
1 — No usable data is present.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and extracted data from all required platforms.

5 — Successfully visited and extracted data from all three platforms.
4 — Successfully visited and extracted data from two platforms; minor issues with the third.
3 — Successfully visited and extracted data from one platform; failed for others.
2 — Attempted but failed to extract usable data from all platforms.
1 — Did not attempt to visit any platform.

#### C. Depth and Specificity (0.25)
Measures whether the agent provided detailed and specific information.

5 — Includes detailed premiums, coverage limits, and additional features for all insurers.
4 — Includes detailed information for at least two insurers; minor omissions for the third.
3 — Includes some details but lacks depth or specificity for one or more insurers.
2 — Includes minimal details; lacks specificity for most insurers.
1 — No meaningful details provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sourced.

5 — Output is organized as a structured table with accurate sourcing.
4 — Output is mostly well-organized; minor formatting or sourcing issues.
3 — Output is usable but lacks clarity or proper sourcing.
2 — Output is disorganized or poorly formatted.
1 — Output is unusable or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())