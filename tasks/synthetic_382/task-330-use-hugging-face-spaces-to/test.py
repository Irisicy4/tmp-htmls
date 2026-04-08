"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Set up a Hugging Face Space for text classification using RoBERTa, upload a dataset, and configure a UI for predictions.
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


TASK_INSTRUCTION = """Use Hugging Face Spaces to set up a basic demo for text classification. Configure the space with a public RoBERTa model, upload a small dataset, and set up a UI that allows users to input text and see predictions. Report the final demo URL and configuration details."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task is to create a text classification demo using Hugging Face Spaces. The agent must use a public RoBERTa model, upload a dataset, and configure a user interface for predictions. A successful completion includes a working demo URL and detailed configuration information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Hugging Face Spaces to set up a basic demo for text classification. Configure the space with a public RoBERTa model, upload a small dataset, and set up a UI that allows users to input text and see predictions. Report the final demo URL and configuration details.

## Task-Specific Constraints
- Must use Hugging Face Spaces as the platform for the demo.
- Must configure the demo with a public RoBERTa model.
- Must upload a small dataset for text classification.
- Must set up a UI that allows users to input text and view predictions.
- Must provide the final demo URL and configuration details in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Hugging Face Spaces to create the demo?
- Was a public RoBERTa model used in the configuration?
- Did the agent upload a dataset for text classification?
- Is the UI functional and does it allow user input and predictions?
- Was the final demo URL and configuration details provided?

### Step 2: Dimension Scoring

#### A. Demo Functionality (0.35)
Measures whether the demo is functional and meets the task requirements.

5 — Demo is fully functional, meets all requirements, and URL is provided.
4 — Demo is functional but minor elements are incomplete or missing.
3 — Demo is partially functional but lacks key features or details.
2 — Demo is mostly non-functional or incomplete.
1 — No demo provided or completely non-functional.

#### B. Platform Usage (0.30)
Measures whether the agent correctly used Hugging Face Spaces and the specified platforms.

5 — Hugging Face Spaces was used correctly, and all required platforms were visited.
4 — Hugging Face Spaces was used correctly, but some platforms were missed.
3 — Hugging Face Spaces was used, but usage was incomplete or incorrect.
2 — Hugging Face Spaces was partially used or misconfigured.
1 — Hugging Face Spaces was not used at all.

#### C. Dataset Integration (0.20)
Measures whether the agent uploaded and integrated a dataset for text classification.

5 — Dataset was uploaded and integrated correctly, with clear evidence.
4 — Dataset was uploaded but integration details are incomplete.
3 — Dataset was partially uploaded or integrated.
2 — Dataset upload or integration was mostly incorrect.
1 — No dataset was uploaded or integrated.

#### D. Response Clarity (0.15)
Measures the clarity and organization of the agent's final response.

5 — Response is clear, well-organized, and includes all required details.
4 — Response is mostly clear but minor details are missing or disorganized.
3 — Response is partially clear but lacks significant details.
2 — Response is unclear or disorganized.
1 — Response is completely unclear or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent created a demo using Hugging Face Spaces, configured it with a RoBERTa model, uploaded a dataset, and provided a functional UI. The final URL and configuration details were included.",
  "demo_functionality": 5,
  "platform_usage": 5,
  "dataset_integration": 5,
  "response_clarity": 5,
  "dimension_reasoning": {
    "demo_functionality": "The demo is fully functional and meets all requirements.",
    "platform_usage": "Hugging Face Spaces was used correctly, and all platforms were visited.",
    "dataset_integration": "The dataset was uploaded and integrated correctly.",
    "response_clarity": "The response is clear, well-organized, and includes all required details."
  },
  "overall_score": 5.0,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "demo_functionality": 0.35,
    "platform_usage": 0.30,
    "dataset_integration": 0.20,
    "response_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())