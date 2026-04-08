"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Set up a Kaggle notebook environment for training a deep learning model with TensorFlow 2, load the CIFAR-10 dataset, and configure GPU acceleration.
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


TASK_INSTRUCTION = """Set up a new Kaggle notebook environment for training a deep learning model with TensorFlow 2. Load the 'CIFAR-10' dataset, install necessary dependencies, and configure the runtime for GPU acceleration. Report the final environment configuration and a successful dataset load output."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves setting up a Kaggle notebook environment for deep learning with TensorFlow 2, loading the CIFAR-10 dataset, and configuring GPU acceleration. The deliverable includes reporting the environment configuration and confirming a successful dataset load.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a new Kaggle notebook environment for training a deep learning model with TensorFlow 2. Load the 'CIFAR-10' dataset, install necessary dependencies, and configure the runtime for GPU acceleration. Report the final environment configuration and a successful dataset load output.

## Task-Specific Constraints
- Must use Kaggle to set up the notebook environment.
- Must install TensorFlow 2 and confirm its installation.
- Must load the CIFAR-10 dataset successfully and display a summary of the dataset.
- Must configure GPU acceleration in the Kaggle runtime environment.
- Must report the final environment configuration in a structured format.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Kaggle and set up the notebook environment?
- Did the agent install TensorFlow 2 and confirm its installation?
- Was the CIFAR-10 dataset loaded successfully, and was a summary displayed?
- Was GPU acceleration configured in the Kaggle runtime environment?
- Was the final environment configuration reported in a structured format?

### Step 2: Dimension Scoring

#### A. Environment Setup Accuracy (0.35)
Measures whether the Kaggle notebook environment was set up correctly and TensorFlow 2 was installed.

5 — Kaggle notebook environment set up correctly, TensorFlow 2 installed and confirmed.
4 — Kaggle notebook set up correctly, TensorFlow 2 installed but confirmation missing.
3 — Partial setup of Kaggle notebook or TensorFlow installation incomplete.
2 — Attempted setup but mostly incorrect or incomplete.
1 — No attempt to set up the environment.

#### B. Dataset Loading Completeness (0.30)
Measures whether the CIFAR-10 dataset was loaded successfully and a summary was displayed.

5 — CIFAR-10 dataset loaded successfully, summary displayed with key statistics.
4 — CIFAR-10 dataset loaded successfully, summary displayed but missing details.
3 — CIFAR-10 dataset loaded but no summary displayed.
2 — Attempted to load dataset but failed.
1 — No attempt to load the dataset.

#### C. GPU Configuration Accuracy (0.20)
Measures whether GPU acceleration was configured successfully in the Kaggle runtime.

5 — GPU acceleration configured successfully and confirmed.
4 — GPU acceleration configured but confirmation missing.
3 — Partial configuration of GPU acceleration.
2 — Attempted configuration but mostly incorrect or incomplete.
1 — No attempt to configure GPU acceleration.

#### D. Output Structure and Reporting Quality (0.15)
Measures whether the final environment configuration and dataset load were reported in a structured format.

5 — Final environment configuration and dataset load reported in a clear, structured format.
4 — Final environment configuration and dataset load reported but with minor formatting issues.
3 — Final environment configuration and dataset load reported but lacks clarity or structure.
2 — Attempted reporting but mostly unclear or unstructured.
1 — No attempt to report the final environment configuration or dataset load.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "environment_setup_accuracy": <1-5>,
  "dataset_loading_completeness": <1-5>,
  "gpu_configuration_accuracy": <1-5>,
  "output_structure_and_reporting_quality": <1-5>,
  "dimension_reasoning": {{
    "environment_setup_accuracy": "<one sentence citing specific evidence>",
    "dataset_loading_completeness": "<one sentence citing specific evidence>",
    "gpu_configuration_accuracy": "<one sentence citing specific evidence>",
    "output_structure_and_reporting_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "environment_setup_accuracy": 0.35,
    "dataset_loading_completeness": 0.30,
    "gpu_configuration_accuracy": 0.20,
    "output_structure_and_reporting_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())