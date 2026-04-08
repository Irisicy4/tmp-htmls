"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Set up a demo space on Hugging Face Spaces for a text classification model with input, prediction, and visualization dashboard.
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


TASK_INSTRUCTION = """Use the Hugging Face Spaces platform and set up a demo space for a text classification model. Include configuration for input text, prediction, and a visualization dashboard of class probabilities. Report the final configuration state with space URL and settings shown."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to set up a demo space on Hugging Face Spaces for a text classification model. The deliverable must include input text configuration, prediction functionality, and a visualization dashboard for class probabilities. A successful completion requires a functional space URL and clear reporting of the final configuration.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Hugging Face Spaces platform and set up a demo space for a text classification model. Include configuration for input text, prediction, and a visualization dashboard of class probabilities. Report the final configuration state with space URL and settings shown.

## Task-Specific Constraints
- Must use the Hugging Face Spaces platform to deploy the demo.
- Must include input text functionality for user interaction.
- Must implement prediction functionality for the text classification model.
- Must provide a visualization dashboard of class probabilities.
- Must report the final space URL and configuration settings in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use the Hugging Face Spaces platform to deploy the demo?
- Does the response include input text functionality for user interaction?
- Does the response include prediction functionality for the text classification model?
- Is there a visualization dashboard for class probabilities?
- Was the final space URL and configuration settings reported clearly?

### Step 2: Dimension Scoring

#### A. Deliverable Functionality (0.35)
Measures whether the demo space is functional and meets the primary requirements.

5 — Fully functional demo space with input, prediction, and visualization dashboard; URL and settings reported clearly.
4 — Functional demo space missing minor elements; URL and settings reported.
3 — Partially functional demo space; missing significant elements but usable.
2 — Demo space mostly non-functional or missing major elements.
1 — No functional demo space created.

#### B. Coverage of Requirements (0.30)
Measures whether all specified requirements were included.

5 — All specified requirements (input, prediction, visualization, URL, settings) included.
4 — One minor requirement missing or incomplete.
3 — Two requirements missing or incomplete.
2 — Three or more requirements missing or incomplete.
1 — No requirements fulfilled.

#### C. Implementation Specificity (0.20)
Measures the detail and clarity of the implementation.

5 — Detailed implementation with clear steps and specific settings reported.
4 — Mostly clear implementation with minor ambiguities.
3 — Some details missing but implementation understandable.
2 — Significant details missing or unclear implementation.
1 — No implementation details provided.

#### D. Output Structure and Reporting (0.15)
Measures the organization and clarity of the response.

5 — Response is well-organized, structured, and easy to follow.
4 — Response is mostly organized with minor issues.
3 — Response is somewhat disorganized but understandable.
2 — Response is poorly organized or difficult to follow.
1 — Response is completely disorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_functionality": <1-5>,
  "coverage_of_requirements": <1-5>,
  "implementation_specificity": <1-5>,
  "output_structure_and_reporting": <1-5>,
  "dimension_reasoning": {{
    "deliverable_functionality": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "implementation_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_reporting": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_functionality": 0.35,
    "coverage_of_requirements": 0.30,
    "implementation_specificity": 0.20,
    "output_structure_and_reporting": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())