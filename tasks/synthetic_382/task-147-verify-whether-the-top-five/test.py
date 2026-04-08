"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Verify whether the top five most-starred JavaScript frameworks on GitHub are still actively maintained.
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


TASK_INSTRUCTION = """Verify whether the top five most-starred JavaScript frameworks on GitHub are still actively maintained. Check the last commit date, number of open issues, and latest release date for each framework."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify the maintenance status of the top five most-starred JavaScript frameworks on GitHub. This includes checking the last commit date, number of open issues, and latest release date for each framework. The domain is software engineering, specifically open-source project analysis.

A successful completion requires the agent to provide accurate and structured data for all five frameworks, sourced directly from GitHub, and formatted in a clear, organized manner.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the top five most-starred JavaScript frameworks on GitHub are still actively maintained. Check the last commit date, number of open issues, and latest release date for each framework.

## Task-Specific Constraints
- Must identify the top five most-starred JavaScript frameworks on GitHub.
- Must provide last commit date, number of open issues, and latest release date for each framework.
- Must source data directly from GitHub repositories.
- Output must be organized as a structured table or list.
- Must include evidence of tool usage to navigate GitHub repositories.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to GitHub repositories for all five frameworks?
- Are the last commit date, number of open issues, and latest release date present for each framework?
- Is the output organized as a structured table or list?
- Are the identified frameworks the correct top five most-starred JavaScript frameworks on GitHub?
- Is the data accurate and sourced directly from GitHub?

### Step 2: Dimension Scoring

#### A. Framework Identification Accuracy (0.35)
Measures whether the agent correctly identified the top five most-starred JavaScript frameworks.

5 — Correctly identifies all five frameworks with accurate rankings.
4 — Identifies four frameworks correctly; one minor error in ranking.
3 — Identifies at least three frameworks correctly; significant ranking errors.
2 — Identifies fewer than three frameworks; major ranking errors.
1 — Fails to identify any frameworks correctly.

#### B. Data Completeness (0.30)
Measures whether the agent provides all required data (last commit date, number of open issues, latest release date) for each framework.

5 — Provides all required data for all five frameworks.
4 — Provides all required data for four frameworks; minor omissions for one.
3 — Provides partial data for at least three frameworks.
2 — Provides partial data for fewer than three frameworks.
1 — Fails to provide any meaningful data.

#### C. Evidence Specificity (0.20)
Measures whether the agent's response includes specific, sourced evidence from GitHub.

5 — Includes specific evidence for all frameworks, with direct GitHub links.
4 — Includes specific evidence for four frameworks; minor omissions for one.
3 — Includes partial evidence for at least three frameworks.
2 — Includes partial evidence for fewer than three frameworks.
1 — Fails to include any specific evidence.

#### D. Output Organization (0.15)
Measures whether the agent's response is well-structured and easy to interpret.

5 — Output is organized as a clear, structured table or list.
4 — Output is mostly organized; minor formatting issues.
3 — Output is partially organized; significant formatting issues.
2 — Output is poorly organized and difficult to interpret.
1 — Output is completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "framework_identification_accuracy": <1-5>,
  "data_completeness": <1-5>,
  "evidence_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "framework_identification_accuracy": "<one sentence citing specific evidence>",
    "data_completeness": "<one sentence citing specific evidence>",
    "evidence_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "framework_identification_accuracy": 0.35,
    "data_completeness": 0.30,
    "evidence_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())