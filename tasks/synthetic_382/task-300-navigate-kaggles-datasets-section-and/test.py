"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Extract NLP datasets from Kaggle with specific filters and provide structured output.
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


TASK_INSTRUCTION = """Navigate Kaggle’s datasets section and apply filters to find public datasets specifically tagged for NLP tasks with over 1,000 downloads and a size under 200 MB. Extract dataset names, descriptions, and download links for the first 10 results."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate Kaggle’s datasets section, apply filters to find NLP datasets with over 1,000 downloads and a size under 200 MB, and extract dataset names, descriptions, and download links for the first 10 results. The domain is Data & ML Engineering, and successful completion requires structured output with all specified details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate Kaggle’s datasets section and apply filters to find public datasets specifically tagged for NLP tasks with over 1,000 downloads and a size under 200 MB. Extract dataset names, descriptions, and download links for the first 10 results.

## Task-Specific Constraints
- Must navigate to Kaggle’s datasets section and apply the specified filters.
- Must extract dataset names, descriptions, and download links.
- Must provide results for the first 10 datasets matching the criteria.
- Output must be structured as a JSON object or table.
- Dataset size and download count must match the specified constraints.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Kaggle and apply the correct filters?
- Are dataset names, descriptions, and download links present in the response?
- Does the output include results for the first 10 datasets matching the criteria?
- Is the output structured as a JSON object or table?
- Are dataset size and download count constraints correctly applied?

### Step 2: Dimension Scoring

#### A. Dataset Filtering Accuracy (0.35)
Measures whether the agent applied the correct filters on Kaggle to identify datasets matching the criteria.

5 — All filters applied correctly and datasets match the specified constraints.
4 — Filters mostly correct, with minor errors affecting 1-2 datasets.
3 — Filters partially correct, with errors affecting 3-5 datasets.
2 — Filters mostly incorrect, with errors affecting over 5 datasets.
1 — Filters not applied or completely incorrect.

#### B. Data Extraction Completeness (0.30)
Measures whether the agent extracted names, descriptions, and download links for the required datasets.

5 — All 10 datasets include names, descriptions, and download links.
4 — 8-9 datasets include names, descriptions, and download links.
3 — 6-7 datasets include names, descriptions, and download links.
2 — 3-5 datasets include names, descriptions, and download links.
1 — Fewer than 3 datasets include names, descriptions, and download links.

#### C. Output Structure and Formatting (0.20)
Measures whether the output is structured correctly as a JSON object or table.

5 — Output is perfectly structured and easy to read.
4 — Output is mostly structured correctly, with minor formatting issues.
3 — Output is partially structured, with noticeable formatting issues.
2 — Output is poorly structured and difficult to interpret.
1 — Output is unstructured or completely incorrect.

#### D. Evidence Credibility (0.15)
Measures whether the extracted data appears accurate and credible based on the tool-call trace.

5 — All extracted data is accurate and matches the tool-call trace.
4 — Most extracted data is accurate, with minor discrepancies.
3 — Some extracted data is accurate, but noticeable discrepancies exist.
2 — Extracted data is mostly inaccurate or unverifiable.
1 — Extracted data is completely inaccurate or unverifiable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dataset_filtering_accuracy": <1-5>,
  "data_extraction_completeness": <1-5>,
  "output_structure_and_formatting": <1-5>,
  "evidence_credibility": <1-5>,
  "dimension_reasoning": {{
    "dataset_filtering_accuracy": "<one sentence citing specific evidence>",
    "data_extraction_completeness": "<one sentence citing specific evidence>",
    "output_structure_and_formatting": "<one sentence citing specific evidence>",
    "evidence_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "dataset_filtering_accuracy": 0.35,
    "data_extraction_completeness": 0.30,
    "output_structure_and_formatting": 0.20,
    "evidence_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())