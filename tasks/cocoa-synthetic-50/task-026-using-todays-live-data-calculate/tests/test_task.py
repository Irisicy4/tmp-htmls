"""
LLM-as-judge evaluator for EvolveBench task.

Category: Finance & Economics
Task: Compare the investment viability of a 1-year Treasury bond versus a 1-year Certificate of Deposit (CD) using live data.
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


TASK_INSTRUCTION = """Using today's live data, calculate whether it's better to invest in a 1-year Treasury bond or a 1-year Certificate of Deposit (CD). Fetch the current 1-year Treasury yield from the U.S. Treasury website, and find the highest 1-year CD rate on Bankrate. Compare the two options considering the risk-free nature and liquidity restrictions of each investment."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to compare the investment viability of a 1-year Treasury bond versus a 1-year Certificate of Deposit (CD) using live data. The agent must fetch the current 1-year Treasury yield from the U.S. Treasury website, find the highest 1-year CD rate on Bankrate, and provide a comparison considering risk-free nature and liquidity restrictions.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using today's live data, calculate whether it's better to invest in a 1-year Treasury bond or a 1-year Certificate of Deposit (CD). Fetch the current 1-year Treasury yield from the U.S. Treasury website, and find the highest 1-year CD rate on Bankrate. Compare the two options considering the risk-free nature and liquidity restrictions of each investment.

## Task-Specific Constraints
- Must visit home.treasury.gov and bankrate.com to fetch live data.
- Must include the current 1-year Treasury yield and the highest available 1-year CD rate.
- Output must include a structured comparison (e.g., table or bullet points).
- Must address the risk-free nature of Treasury bonds and liquidity restrictions of CDs.
- Must provide a clear recommendation based on the comparison.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are the current 1-year Treasury yield and the highest 1-year CD rate present in the response?
- Is the output organized as a structured comparison (e.g., table or bullet points)?
- Does the response address the risk-free nature of Treasury bonds and liquidity restrictions of CDs?
- Is the recommendation clear and supported by the comparison?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's comparison is correct and complete.

5 — Includes accurate Treasury yield, highest CD rate, and a clear recommendation based on comparison.
4 — Includes most required data but has minor inaccuracies or lacks clarity in recommendation.
3 — Includes partial data and/or unclear recommendation but usable.
2 — Includes minimal data and/or incorrect recommendation.
1 — No usable data or recommendation.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and fetched necessary data.

5 — Successfully visited home.treasury.gov and bankrate.com, fetched correct data from both.
4 — Visited both platforms but data is incomplete or slightly incorrect.
3 — Visited at least one platform and fetched partial data.
2 — Visited one platform but failed to fetch usable data.
1 — Did not visit required platforms or fetch any data.

#### C. Depth of Comparison (0.25)
Measures the level of detail in the comparison and analysis.

5 — Provides detailed comparison including numerical data, risk-free nature, and liquidity restrictions.
4 — Provides good comparison but lacks minor details or depth.
3 — Provides basic comparison with minimal analysis.
2 — Provides incomplete or shallow comparison.
1 — No meaningful comparison provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and uses credible sources.

5 — Output is structured (e.g., table or bullet points) and cites credible sources.
4 — Output is mostly structured but lacks minor formatting or citation issues.
3 — Output is minimally structured but usable.
2 — Output is poorly structured or lacks credibility.
1 — Output is unstructured and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())