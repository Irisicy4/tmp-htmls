"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Investigate and resolve a fine-tuning issue with a T5 model in Hugging Face Transformers.
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


TASK_INSTRUCTION = """A user reports that fine-tuning a T5 model using Hugging Face Transformers fails when using version 4.30.0 due to a 'KeyError: attention_mask'. Investigate the root cause using Hugging Face documentation, GitHub issues, and community forums. Provide the affected version range and the official recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to investigate a fine-tuning issue with a T5 model in Hugging Face Transformers. The agent must identify the affected version range, determine the root cause, and provide the official recommended fix. The task involves navigating Hugging Face documentation, GitHub issues, and community forums to gather evidence.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports that fine-tuning a T5 model using Hugging Face Transformers fails when using version 4.30.0 due to a 'KeyError: attention_mask'. Investigate the root cause using Hugging Face documentation, GitHub issues, and community forums. Provide the affected version range and the official recommended fix.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (huggingface.co, github.com, discuss.huggingface.co).
- Must identify the affected version range of Hugging Face Transformers.
- Must provide the official recommended fix for the issue.
- Must explain the root cause of the 'KeyError: attention_mask' clearly.
- Output must be structured and include references to sources.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify the affected version range of Hugging Face Transformers?
- Did the agent provide the official recommended fix for the issue?
- Did the agent explain the root cause of the 'KeyError: attention_mask' clearly?
- Is the output structured and includes references to sources?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the affected version range and provided the official recommended fix.

5 — Identifies the exact version range and provides the official recommended fix with clear evidence.
4 — Identifies the version range and provides the fix, but with minor inaccuracies or missing evidence.
3 — Identifies the version range and fix, but lacks clarity or supporting evidence.
2 — Provides incomplete or incorrect version range and fix.
1 — Fails to provide the version range or fix.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent navigated all required platforms and gathered evidence from them.

5 — Uses all three platforms (huggingface.co, github.com, discuss.huggingface.co) with relevant evidence.
4 — Uses two platforms with relevant evidence.
3 — Uses at least one platform with some evidence.
2 — Navigates platforms but fails to gather relevant evidence.
1 — Does not navigate any required platforms.

#### C. Depth of Explanation (0.25)
Measures the clarity and depth of the explanation for the root cause of the issue.

5 — Provides a detailed and clear explanation of the root cause, citing specific technical details.
4 — Provides a clear explanation but lacks some depth or technical details.
3 — Provides a basic explanation with minimal technical details.
2 — Provides an unclear or incomplete explanation.
1 — Fails to explain the root cause.

#### D. Output Structure and Source Credibility (0.10)
Measures whether the output is well-organized and references credible sources.

5 — Output is well-structured and includes references to credible sources.
4 — Output is structured but lacks some references or credibility.
3 — Output is minimally structured with few references.
2 — Output is poorly structured and lacks credible references.
1 — Output is unstructured and lacks references.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_explanation": <1-5>,
  "output_structure_and_source_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_explanation": "<one sentence citing specific evidence>",
    "output_structure_and_source_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_explanation": 0.25,
    "output_structure_and_source_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())