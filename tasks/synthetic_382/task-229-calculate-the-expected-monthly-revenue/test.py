"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Calculate the expected monthly revenue for a YouTube creator with 50k subscribers based on average CPM rates.
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


TASK_INSTRUCTION = """Calculate the expected monthly revenue for a YouTube creator with 50k subscribers based on average CPM rates. Visit Socialblade, TubeBuddy, and Statista for CPM benchmarks and combine this with estimated monthly view counts for channels in the 'Tech Reviews' niche."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the expected monthly revenue for a YouTube creator with 50k subscribers based on average CPM rates. The agent must gather CPM benchmarks from Socialblade, TubeBuddy, and Statista, and combine this with estimated monthly view counts for channels in the 'Tech Reviews' niche. A successful completion involves accurate calculations and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the expected monthly revenue for a YouTube creator with 50k subscribers based on average CPM rates. Visit Socialblade, TubeBuddy, and Statista for CPM benchmarks and combine this with estimated monthly view counts for channels in the 'Tech Reviews' niche.

## Task-Specific Constraints
- Must visit Socialblade, TubeBuddy, and Statista to gather CPM benchmarks.
- Must include CPM rates for the 'Tech Reviews' niche.
- Must estimate monthly view counts based on subscriber count and niche averages.
- Output must include detailed calculations with sources cited.
- Final output must be structured as a table or JSON format.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Socialblade, TubeBuddy, and Statista? Which platforms were actually visited?
- Are CPM benchmarks for the 'Tech Reviews' niche present in the response?
- Does the response include estimated monthly view counts based on subscriber count?
- Are calculations accurate and sources cited?
- Is the output structured as a table or JSON format?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the expected monthly revenue calculation is correct and complete.

5 — Includes accurate CPM rates, monthly view counts, and calculations with sources cited.
4 — Mostly accurate but missing minor details or sources.
3 — Partially correct; calculations are incomplete or missing sources.
2 — Incorrect calculations or missing key data.
1 — No attempt or completely wrong.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all specified platforms and used their data.

5 — Uses data from Socialblade, TubeBuddy, and Statista.
4 — Uses data from two platforms but misses one.
3 — Uses data from only one platform.
2 — Attempts platform navigation but gathers no usable data.
1 — No platform navigation attempted.

#### C. Depth of Analysis (0.25)
Measures the level of detail in the response, including niche-specific CPM benchmarks and view count estimates.

5 — Includes niche-specific CPM benchmarks and detailed view count estimates.
4 — Includes CPM benchmarks but lacks niche-specific details or detailed view counts.
3 — Includes some CPM data but lacks significant detail.
2 — Minimal detail; generic or vague data.
1 — No meaningful analysis provided.

#### D. Output Structure and Source Credibility (0.10)
Measures whether the output is well-organized and sources are credible.

5 — Output is structured as a table or JSON and cites credible sources.
4 — Output is structured but lacks source credibility.
3 — Output is poorly organized or missing citations.
2 — Output is disorganized or unclear.
1 — No structure or citations provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())