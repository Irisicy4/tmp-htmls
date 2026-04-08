"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Investigate why a TikTok creator's views have dramatically decreased and provide a diagnosis with sources.
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


TASK_INSTRUCTION = """Investigate why a TikTok creator's views have dramatically decreased over the past week despite consistent posting. Look into recent algorithm changes, possible shadowban signals, or changes in content strategy. Provide a diagnosis with links to sources explaining the root cause and fixes, if applicable."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to investigate why a TikTok creator's views have decreased, analyze potential causes such as algorithm changes or shadowbanning, and provide a diagnosis with links to credible sources. Successful completion requires identifying the root cause and suggesting actionable fixes.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Investigate why a TikTok creator's views have dramatically decreased over the past week despite consistent posting. Look into recent algorithm changes, possible shadowban signals, or changes in content strategy. Provide a diagnosis with links to sources explaining the root cause and fixes, if applicable.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: tiktok.com, socialmediatoday.com, reddit.com.
- Must identify at least one plausible cause for the view decrease.
- Must provide at least two actionable fixes or recommendations.
- Must include links to credible sources supporting the diagnosis and recommendations.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify a plausible cause for the view decrease?
- Are actionable fixes or recommendations provided? How many?
- Are links to credible sources included and relevant to the diagnosis?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Diagnosis Accuracy (0.35)
Measures whether the agent correctly identified the root cause(s) of the view decrease.

5 — Identifies 2 or more plausible causes with detailed explanations.
4 — Identifies 1 plausible cause with detailed explanation.
3 — Identifies 1 plausible cause but lacks detail.
2 — Identifies a cause but it is implausible or poorly explained.
1 — No plausible cause identified.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited the required platforms and gathered relevant information.

5 — Successfully navigates all 3 platforms and extracts relevant data.
4 — Navigates 2 platforms and extracts relevant data.
3 — Navigates 1 platform and extracts some relevant data.
2 — Navigates 1 platform but data is irrelevant or insufficient.
1 — Does not navigate any platform.

#### C. Actionability of Recommendations (0.25)
Measures whether the recommendations provided are actionable and specific.

5 — Provides 3 or more actionable fixes with detailed steps.
4 — Provides 2 actionable fixes with detailed steps.
3 — Provides 1 actionable fix with some detail.
2 — Provides 1 fix but lacks detail or specificity.
1 — No actionable fixes provided.

#### D. Source Credibility and Output Structure (0.10)
Measures whether the sources are credible and the output is well-organized.

5 — Includes 3 or more credible sources; output is structured as a table or list.
4 — Includes 2 credible sources; output is structured as a table or list.
3 — Includes 1 credible source; output is structured but lacks clarity.
2 — Includes 1 source but it is not credible; output is disorganized.
1 — No sources included; output is disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "diagnosis_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "actionability_of_recommendations": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "diagnosis_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "actionability_of_recommendations": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "diagnosis_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "actionability_of_recommendations": 0.25,
    "source_credibility_and_output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())