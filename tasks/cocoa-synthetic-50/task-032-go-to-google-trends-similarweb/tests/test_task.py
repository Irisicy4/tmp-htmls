"""
LLM-as-judge evaluator for EvolveBench task.

Category: Marketing & Analytics
Task: Extract data on top three trending search keywords related to 'electric cars' in the United States from Google Trends, Similarweb, and Ahrefs Free Keyword Generator.
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


TASK_INSTRUCTION = """Go to Google Trends, Similarweb, and Ahrefs Free Keyword Generator to extract data on the top three trending search keywords related to 'electric cars' in the United States. For each keyword, provide search popularity metrics, estimated traffic, and related keywords provided by these platforms."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves extracting data on the top three trending search keywords related to 'electric cars' in the United States from Google Trends, Similarweb, and Ahrefs Free Keyword Generator. The agent must provide search popularity metrics, estimated traffic, and related keywords for each keyword.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Google Trends, Similarweb, and Ahrefs Free Keyword Generator to extract data on the top three trending search keywords related to 'electric cars' in the United States. For each keyword, provide search popularity metrics, estimated traffic, and related keywords provided by these platforms.

## Task-Specific Constraints
- Must visit all three specified platforms: Google Trends, Similarweb, and Ahrefs Free Keyword Generator.
- Must identify exactly three trending keywords related to 'electric cars'.
- Must provide search popularity metrics, estimated traffic, and related keywords for each keyword.
- Output must be organized as a structured table or list.
- Metrics and related keywords must be sourced directly from the platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (Google Trends, Similarweb, Ahrefs)?
- Are exactly three trending keywords related to 'electric cars' identified?
- Are search popularity metrics, estimated traffic, and related keywords provided for each keyword?
- Is the output organized as a structured table or list?
- Are the metrics and related keywords sourced directly from the platforms?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the top three trending keywords and provided the required metrics.

5 — Identifies three keywords and provides all required metrics (popularity, traffic, related keywords) accurately.
4 — Identifies three keywords and provides most metrics accurately, with minor omissions.
3 — Identifies three keywords but provides incomplete or partially accurate metrics.
2 — Identifies fewer than three keywords or provides mostly incorrect metrics.
1 — Fails to identify keywords or provide any metrics.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all specified platforms and extracted data from each.

5 — Successfully uses all three platforms and extracts relevant data from each.
4 — Uses all three platforms but extracts incomplete data from one.
3 — Uses two platforms and extracts relevant data.
2 — Uses only one platform or extracts mostly irrelevant data.
1 — Fails to use any specified platform.

#### C. Depth of Metrics (0.20)
Measures the level of detail and specificity in the metrics provided.

5 — Provides detailed metrics for all keywords, including numerical values and comparisons.
4 — Provides detailed metrics for most keywords, with minor omissions.
3 — Provides basic metrics for all keywords, lacking depth or specificity.
2 — Provides incomplete or vague metrics for most keywords.
1 — Fails to provide any meaningful metrics.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and sourced directly from credible platforms.

5 — Output is structured as a clear table or list, with all data sourced from the specified platforms.
4 — Output is mostly well-organized, with minor formatting issues or unclear sourcing.
3 — Output is somewhat organized but lacks clarity or has sourcing issues.
2 — Output is poorly organized or mostly unsourced.
1 — Output is completely unstructured or unsourced.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited all three required platforms and identified three trending keywords related to 'electric cars'. Metrics were provided but lacked depth in some cases. The output was structured as a table, with minor sourcing issues.",
  "primary_deliverable_accuracy": 4,
  "coverage_of_platforms": 5,
  "depth_of_metrics": 3,
  "output_structure_and_credibility": 4,
  "dimension_reasoning": {
    "primary_deliverable_accuracy": "Three keywords were identified, but some metrics were incomplete.",
    "coverage_of_platforms": "All three platforms were visited and used to extract data.",
    "depth_of_metrics": "Metrics lacked numerical comparisons and detailed analysis.",
    "output_structure_and_credibility": "Output was structured as a table but had minor sourcing issues."
  },
  "overall_score": 4.05,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_metrics": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())