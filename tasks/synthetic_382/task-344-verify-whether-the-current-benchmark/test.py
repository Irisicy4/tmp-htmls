"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Verify reproducibility claims and benchmark details for BERT, GPT-3, and RoBERTa on Papers with Code.
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


TASK_INSTRUCTION = """Verify whether the current benchmark results for BERT, GPT-3, and RoBERTa are still listed as reproducible on Papers with Code. Check for hardware configurations, reported metrics, and reproducibility claims for each model."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying reproducibility claims and benchmark details for BERT, GPT-3, and RoBERTa on Papers with Code. The agent must check hardware configurations, reported metrics, and reproducibility claims for each model. A successful completion includes accurate extraction of reproducibility information and structured presentation of findings.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the current benchmark results for BERT, GPT-3, and RoBERTa are still listed as reproducible on Papers with Code. Check for hardware configurations, reported metrics, and reproducibility claims for each model.

## Task-Specific Constraints
- Must verify reproducibility claims for all three models: BERT, GPT-3, and RoBERTa.
- Must include hardware configurations for each model if available.
- Must include reported metrics (e.g., accuracy, F1 score) for each model.
- Must explicitly state whether reproducibility claims are still valid for each model.
- Output must be organized as a structured table or list format.
- Must cite specific pages or sections visited on Papers with Code.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Papers with Code and verify claims for all three models (BERT, GPT-3, RoBERTa)?
- Are hardware configurations included for each model?
- Are reported metrics (e.g., accuracy, F1 score) present and correctly extracted?
- Does the response explicitly state whether reproducibility claims are valid for each model?
- Is the output organized as a structured table or list format?

### Step 2: Dimension Scoring

#### A. Reproducibility Verification Accuracy (0.35)
Measures whether the agent correctly verified reproducibility claims for all three models.

5 — Verifies reproducibility claims for all three models with correct conclusions.
4 — Verifies reproducibility claims for at least two models with correct conclusions.
3 — Verifies reproducibility claims for at least one model with correct conclusions.
2 — Attempts verification but provides incorrect or incomplete conclusions.
1 — Does not attempt verification or provides entirely incorrect conclusions.

#### B. Coverage of Required Details (0.30)
Measures whether the agent includes all required details (hardware configurations, metrics, reproducibility claims).

5 — Includes all required details for all three models.
4 — Includes most required details for at least two models.
3 — Includes some required details for at least one model.
2 — Includes few required details or provides incomplete information.
1 — Does not include any required details.

#### C. Depth of Information (0.20)
Measures the specificity and depth of the extracted information (e.g., detailed metrics, hardware specs).

5 — Provides highly specific and detailed information for all three models.
4 — Provides detailed information for at least two models.
3 — Provides some specific information for at least one model.
2 — Provides minimal or vague information.
1 — Provides no specific information.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and cites credible sources.

5 — Output is well-organized, structured, and cites credible sources for all claims.
4 — Output is mostly well-organized and cites credible sources for most claims.
3 — Output is partially organized and cites some credible sources.
2 — Output is poorly organized and lacks credible citations.
1 — Output is disorganized and provides no credible citations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "reproducibility_verification_accuracy": <1-5>,
  "coverage_of_required_details": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "reproducibility_verification_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_details": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "reproducibility_verification_accuracy": 0.35,
    "coverage_of_required_details": 0.30,
    "depth_of_information": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())