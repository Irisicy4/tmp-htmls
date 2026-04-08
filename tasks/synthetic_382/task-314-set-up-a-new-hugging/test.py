"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Set up a Hugging Face Space for training a sentiment analysis model on the Amazon review dataset, including dataset integration, training configuration, and reporting dashboard setup.
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


TASK_INSTRUCTION = """Set up a new Hugging Face Space for training a sentiment analysis model on the Amazon review dataset. Configure the environment by adding the dataset, specifying training parameters, and setting up a reporting dashboard for loss and accuracy."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves setting up a Hugging Face Space for training a sentiment analysis model on the Amazon review dataset. The agent must integrate the dataset, configure the training parameters, and set up a reporting dashboard for loss and accuracy metrics. Success requires proper use of the Hugging Face platform and correct implementation of all required deliverables.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a new Hugging Face Space for training a sentiment analysis model on the Amazon review dataset. Configure the environment by adding the dataset, specifying training parameters, and setting up a reporting dashboard for loss and accuracy.

## Task-Specific Constraints
- Must use the Hugging Face platform to create the Space and configure the environment.
- Must integrate the Amazon review dataset from archive.ics.uci.edu.
- Must specify training parameters such as learning rate, batch size, and epochs.
- Must set up a reporting dashboard that tracks loss and accuracy during training.
- Must provide evidence of successful execution, including screenshots or logs of the Space setup and training results.
- Must document all steps taken in a clear and structured format.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Hugging Face platform and create a Space?
- Did the agent integrate the Amazon review dataset correctly?
- Are the training parameters (learning rate, batch size, epochs) specified and appropriate?
- Is there a reporting dashboard for loss and accuracy, and does it function correctly?
- Are the steps documented clearly, with evidence of execution (e.g., screenshots or logs)?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the Hugging Face Space was correctly set up and the sentiment analysis model trained.

5 — Space created, dataset integrated, model trained successfully, and dashboard functional.
4 — Space created and model trained, but minor issues in dataset integration or dashboard functionality.
3 — Partial setup: Space created but training or dashboard incomplete.
2 — Attempted setup but mostly incorrect or incomplete.
1 — No meaningful attempt.

#### B. Coverage of Requirements (0.30)
Measures whether all task-specific constraints were addressed.

5 — All constraints satisfied (dataset integration, training parameters, dashboard setup, documentation).
4 — Most constraints satisfied, with minor omissions.
3 — Some constraints satisfied, but key elements missing.
2 — Few constraints satisfied; major omissions.
1 — No constraints satisfied.

#### C. Specificity and Detail (0.20)
Measures the depth of explanation and evidence provided.

5 — Detailed documentation with screenshots/logs for every step.
4 — Good documentation but missing minor details.
3 — Basic documentation with minimal evidence.
2 — Poor documentation with significant gaps.
1 — No documentation or evidence.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and credible.

5 — Clear, structured response with credible evidence.
4 — Mostly clear but minor organizational issues.
3 — Acceptable structure but lacks clarity or credibility.
2 — Poorly structured and unclear.
1 — No structure or credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "specificity_and_detail": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "specificity_and_detail": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "specificity_and_detail": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())