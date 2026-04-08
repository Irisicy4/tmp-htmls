"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Extract metadata for the first 5 qualifying datasets on Kaggle filtered by 'computer vision' and 'open-source license'.
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


TASK_INSTRUCTION = """Go to Kaggle's dataset repository and apply filters for 'computer vision' and 'open-source license'. Extract metadata (title, description, rows, columns, last updated date) for the first 5 qualifying datasets."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to navigate Kaggle's dataset repository, apply specific filters ('computer vision' and 'open-source license'), and extract metadata for the first 5 qualifying datasets. This is a Data & ML Engineering task requiring structured output with specific details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Kaggle's dataset repository and apply filters for 'computer vision' and 'open-source license'. Extract metadata (title, description, rows, columns, last updated date) for the first 5 qualifying datasets.

## Task-Specific Constraints
- Must apply the 'computer vision' and 'open-source license' filters correctly.
- Must extract metadata for exactly 5 datasets.
- Metadata must include title, description, rows, columns, and last updated date.
- Output must be structured as a table or JSON object.
- Must provide accurate and complete metadata without omissions.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Kaggle and apply the required filters ('computer vision' and 'open-source license')?
- Did the agent extract metadata for exactly 5 datasets?
- Is the metadata complete, including title, description, rows, columns, and last updated date for each dataset?
- Is the output structured correctly (e.g., table or JSON)?
- Are there any inaccuracies or omissions in the extracted metadata?

### Step 2: Dimension Scoring

#### A. Metadata Accuracy (0.35)
Measures whether the extracted metadata is correct and complete.

5 — Metadata for all 5 datasets is accurate and complete.
4 — Metadata for 4 datasets is accurate and complete, with minor omissions.
3 — Metadata for at least 3 datasets is partially accurate but incomplete.
2 — Metadata for 1-2 datasets is mostly missing or incorrect.
1 — No metadata extracted or completely incorrect.

#### B. Filter Application (0.30)
Measures whether the agent correctly applied the required filters on Kaggle.

5 — Both filters ('computer vision' and 'open-source license') applied correctly.
4 — One filter applied correctly, with minor issues in the other.
3 — Partial application of filters, resulting in some qualifying datasets missed.
2 — Filters applied incorrectly or inconsistently.
1 — Filters not applied at all.

#### C. Output Structure (0.20)
Measures whether the output is well-organized and structured as required.

5 — Output is structured as a clear table or JSON object with all required fields.
4 — Output is mostly structured correctly, with minor formatting issues.
3 — Output is partially structured but lacks clarity or completeness.
2 — Output is poorly structured or difficult to interpret.
1 — Output is unstructured or completely missing.

#### D. Evidence Credibility (0.15)
Measures whether the agent's evidence and tool usage are credible and traceable.

5 — Tool usage and evidence are fully traceable and credible.
4 — Tool usage is mostly credible, with minor gaps in traceability.
3 — Tool usage is partially credible but lacks clarity in some areas.
2 — Tool usage is mostly unclear or untraceable.
1 — Tool usage is completely absent or non-credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent successfully navigated Kaggle and applied filters but extracted metadata for fewer than 5 datasets. Output structure was partially correct, but some metadata fields were incomplete.",
  "metadata_accuracy": 3,
  "filter_application": 4,
  "output_structure": 3,
  "evidence_credibility": 4,
  "dimension_reasoning": {
    "metadata_accuracy": "Metadata was extracted for 3 datasets, but some fields were incomplete.",
    "filter_application": "Both filters were applied correctly, but fewer datasets were extracted.",
    "output_structure": "Output was partially structured but lacked clarity in formatting.",
    "evidence_credibility": "Tool usage was traceable and credible, with minor gaps in execution trace."
  },
  "overall_score": 3.35,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "metadata_accuracy": 0.35,
    "filter_application": 0.30,
    "output_structure": 0.20,
    "evidence_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())