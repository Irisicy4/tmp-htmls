"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Verify whether the latest GLUE leaderboard results match published model performance claims.
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


TASK_INSTRUCTION = """Verify whether the latest leaderboard results for the GLUE benchmark (General Language Understanding Evaluation) accurately reflect model performance claims. Check the results against official GLUE benchmarks, research papers, and GitHub repositories. Produce a report stating whether the claims match published results."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to verify the accuracy of GLUE leaderboard results by cross-referencing them with official benchmark data, published research papers, and GitHub repositories. This task is in the domain of data and machine learning engineering, and success requires producing a report that confirms or refutes the accuracy of the claims based on credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the latest leaderboard results for the GLUE benchmark (General Language Understanding Evaluation) accurately reflect model performance claims. Check the results against official GLUE benchmarks, research papers, and GitHub repositories. Produce a report stating whether the claims match published results.

## Task-Specific Constraints
- Must visit gluebenchmark.com, arxiv.org, and github.com.
- Must cross-reference leaderboard results with at least two research papers.
- Must verify claims against official GLUE benchmark data.
- Must produce a structured report with clear conclusions.
- Must cite specific sources used for verification.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to gluebenchmark.com, arxiv.org, and github.com? Which platforms were actually visited?
- Did the agent cross-reference leaderboard results with at least two research papers?
- Did the agent verify claims against official GLUE benchmark data?
- Is the output organized as a structured report with clear conclusions?
- Are specific sources cited in the report?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the report correctly verifies leaderboard claims against credible sources.

5 — All claims are verified correctly, with no errors or omissions.
4 — Most claims are verified correctly, with minor errors or omissions.
3 — Some claims are verified correctly, but significant errors or omissions exist.
2 — Few claims are verified correctly, with major errors or omissions.
1 — No claims are verified correctly, or the report is absent.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited all required platforms and used sufficient sources.

5 — Visited all required platforms and used at least 3 credible sources.
4 — Visited most required platforms and used at least 2 credible sources.
3 — Visited some required platforms and used at least 1 credible source.
2 — Visited few required platforms and used no credible sources.
1 — Did not visit any required platforms or use any sources.

#### C. Depth of Analysis (0.25)
Measures the level of detail and specificity in verifying claims.

5 — Provides detailed analysis with specific numbers, comparisons, and citations.
4 — Provides good analysis with some specific details and citations.
3 — Provides basic analysis with limited details or citations.
2 — Provides minimal analysis with vague or unsupported claims.
1 — Provides no analysis or completely unsupported claims.

#### D. Report Structure and Source Credibility (0.10)
Measures the organization of the report and the credibility of cited sources.

5 — Report is well-organized and cites highly credible sources.
4 — Report is mostly organized and cites credible sources.
3 — Report is partially organized and cites some credible sources.
2 — Report is poorly organized and cites few credible sources.
1 — Report is disorganized and cites no credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "report_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "report_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_analysis": 0.25,
    "report_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())