"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Verify whether npm, Yarn, and pnpm are actively maintained by checking their last commit dates, open issue counts, and latest release dates from their GitHub repositories.
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


TASK_INSTRUCTION = """Verify whether the three most popular package managers for JavaScript (npm, Yarn, and pnpm) are actively maintained. Check their last commit dates, open issue counts, and latest release dates directly from their GitHub repositories."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires verifying the maintenance status of npm, Yarn, and pnpm by checking their last commit dates, open issue counts, and latest release dates from their GitHub repositories. This is a Software Engineering task, and successful completion involves accurate retrieval and reporting of this data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the three most popular package managers for JavaScript (npm, Yarn, and pnpm) are actively maintained. Check their last commit dates, open issue counts, and latest release dates directly from their GitHub repositories.

## Task-Specific Constraints
- Must visit the GitHub repositories for npm, Yarn, and pnpm.
- Must retrieve and report the last commit dates for all three repositories.
- Must retrieve and report the open issue counts for all three repositories.
- Must retrieve and report the latest release dates for all three repositories.
- Output must be organized as a structured list or table with clear labels for each data point.
- Must include specific URLs or sources for the retrieved data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the GitHub repositories for npm, Yarn, and pnpm? Which ones were actually visited?
- Are the last commit dates, open issue counts, and latest release dates present in the response?
- Is the output organized as a structured list or table with clear labels?
- Are the URLs or sources for the retrieved data included in the response?
- Are the reported data points accurate based on the provided sources?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the reported data (last commit dates, open issue counts, and latest release dates) is correct and complete.

5 — All data points are accurate and complete for npm, Yarn, and pnpm.
4 — Most data points are accurate, with minor errors or omissions.
3 — Some data points are accurate, but significant errors or omissions exist.
2 — Few data points are accurate, with major errors or omissions.
1 — No accurate data points are provided.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required GitHub repositories and retrieved data from them.

5 — Data is retrieved from all three repositories (npm, Yarn, pnpm).
4 — Data is retrieved from two repositories, with minor omissions.
3 — Data is retrieved from one repository, or partial data from two.
2 — Minimal data is retrieved, with major omissions.
1 — No data is retrieved from the required repositories.

#### C. Depth of Information (0.25)
Measures the level of detail in the reported data, including timestamps, counts, and release versions.

5 — Includes detailed timestamps, issue counts, and release versions for all platforms.
4 — Includes most details, with minor omissions.
3 — Includes some details, but lacks significant depth.
2 — Includes minimal details, with major omissions.
1 — No meaningful details are included.

#### D. Output Structure and Source Credibility (0.10)
Measures whether the output is well-organized and whether credible sources are cited.

5 — Output is well-organized, with clear labels and credible sources cited for all data.
4 — Output is mostly organized, with minor formatting issues or missing sources.
3 — Output is partially organized, with significant formatting issues or missing sources.
2 — Output is poorly organized, with minimal structure or credibility.
1 — Output is completely unstructured and lacks credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_information": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())