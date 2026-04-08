"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Evaluate and recommend the most cost-effective vector database for storing and querying 1 million vector embeddings based on benchmarks from documentation and GitHub pages.
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


TASK_INSTRUCTION = """Fetch performance benchmarks for three open-source vector databases (e.g., Milvus, Pinecone, Weaviate) using their documentation and GitHub pages. Calculate the most cost-effective option for storing and querying a dataset with 1 million vector embeddings, considering speed, scalability, and pricing. Recommend the best choice with supporting data."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to fetch performance benchmarks for three open-source vector databases (Milvus, Pinecone, Weaviate) using their documentation and GitHub pages. The agent must calculate the most cost-effective option for storing and querying a dataset with 1 million vector embeddings, considering speed, scalability, and pricing. A successful completion includes a clear recommendation supported by structured data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch performance benchmarks for three open-source vector databases (e.g., Milvus, Pinecone, Weaviate) using their documentation and GitHub pages. Calculate the most cost-effective option for storing and querying a dataset with 1 million vector embeddings, considering speed, scalability, and pricing. Recommend the best choice with supporting data.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (Milvus, Pinecone, Weaviate).
- Must include price data for all items compared.
- Output must be organized as a table or structured list.
- Must address speed, scalability, and pricing explicitly.
- Recommendation must be supported by specific data points from the sources.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are speed, scalability, and pricing data present in the response?
- Is the output organized as a table or structured list?
- Are the comparisons supported by specific data points from the platforms?
- Is the recommendation logically derived from the data provided?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the recommendation is correct and supported by the data.

5 — Recommendation is correct, fully supported by data, and addresses all constraints.
4 — Recommendation is correct but lacks minor supporting details.
3 — Recommendation is partially correct but misses key constraints.
2 — Recommendation is mostly incorrect or unsupported.
1 — Recommendation is absent or completely wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included relevant data.

5 — All three platforms visited and key data included for each.
4 — Two platforms visited with relevant data included.
3 — At least one platform visited with partial data included.
2 — Platforms visited but no relevant data included.
1 — No platforms visited or no data included.

#### C. Depth of Analysis (0.25)
Measures the specificity and detail of the comparisons.

5 — Includes detailed comparisons with specific numbers for speed, scalability, and pricing.
4 — Comparisons are present but lack minor details or specificity.
3 — Comparisons are present but lack significant details.
2 — Comparisons are mostly absent or vague.
1 — Comparisons are completely absent.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and uses credible sources.

5 — Output is well-organized, structured as required, and sources are credible.
4 — Output is mostly well-organized with minor structural issues.
3 — Output is partially organized but lacks clarity or credibility.
2 — Output is poorly organized or lacks credible sources.
1 — Output is completely unstructured or sources are absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())