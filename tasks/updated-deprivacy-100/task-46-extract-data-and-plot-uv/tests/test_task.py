"""
LLM-as-judge evaluator for EvolveBench task-46-extract-data-and-plot-uv.

Category: Data Analysis and Visualization
Task: Extract data and plot the UV data for the past 5 years.
"""

import os, json, re

TASK_INSTRUCTION = """Extract data and plot the UV data for the past 5 years."""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to extract accurate UV data for the past 5 years, process it correctly, and produce a clear and informative plot. The evaluation will focus on the correctness of the data extraction, the appropriateness of the data processing steps, and the clarity and accuracy of the resulting visualization.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- The data must cover exactly the past 5 years from the current date.
- The UV data must be sourced from a reliable and verifiable source.
- The plot must clearly represent the trends or patterns in the UV data over the 5-year period.
- The axes, labels, and legends in the plot must be appropriately labeled and easy to understand.
- The agent must handle any missing or incomplete data in a reasonable and documented manner.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent extract UV data covering the past 5 years from a reliable source?
- Did the agent process the data correctly, including handling missing or incomplete data?
- Does the plot clearly and accurately represent the trends or patterns in the UV data?
- Are the axes, labels, and legends in the plot clear and appropriate?
- Did the agent document the data source and processing steps adequately?

### Step 2: Dimension Scoring

#### A. Data Extraction Accuracy
Measures the correctness and completeness of the extracted UV data.

5 — The data covers exactly the past 5 years and is sourced from a reliable, verifiable source.
4 — The data covers the past 5 years with minor gaps or ambiguities in the source.
3 — The data partially covers the past 5 years or the source is not fully reliable.
2 — The data is incomplete or the source is questionable.
1 — The data is missing or entirely unreliable.

#### B. Data Processing Quality
Evaluates how well the agent handled and processed the data.

5 — The data was processed correctly, with clear handling of missing or incomplete data.
4 — The data was processed correctly with minor issues in handling missing or incomplete data.
3 — The data processing had noticeable issues or lacked clarity in handling missing data.
2 — The data processing was flawed or poorly documented.
1 — The data processing was incorrect or not performed.

#### C. Visualization Clarity
Assesses the clarity and informativeness of the plot.

5 — The plot is clear, well-labeled, and effectively communicates trends or patterns.
4 — The plot is mostly clear and informative with minor labeling or clarity issues.
3 — The plot is somewhat clear but has noticeable issues in labeling or communication of trends.
2 — The plot is unclear or poorly labeled, making it hard to interpret.
1 — The plot is missing or entirely incomprehensible.

#### D. Documentation Completeness
Measures how well the agent documented the data source and processing steps.

5 — The data source and processing steps are thoroughly and clearly documented.
4 — The documentation is mostly complete with minor omissions or ambiguities.
3 — The documentation is partially complete or lacks clarity.
2 — The documentation is incomplete or unclear.
1 — The documentation is missing or entirely inadequate.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_extraction_accuracy": <1-5>,
  "data_processing_quality": <1-5>,
  "visualization_clarity": <1-5>,
  "documentation_completeness": <1-5>,
  "dimension_reasoning": {{
    "data_extraction_accuracy": "<one sentence citing specific evidence>",
    "data_processing_quality": "<one sentence citing specific evidence>",
    "visualization_clarity": "<one sentence citing specific evidence>",
    "documentation_completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_extraction_accuracy": 0.3,
    "data_processing_quality": 0.25,
    "visualization_clarity": 0.25,
    "documentation_completeness": 0.2,
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
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

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