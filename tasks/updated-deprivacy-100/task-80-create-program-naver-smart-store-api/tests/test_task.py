"""
LLM-as-judge evaluator for EvolveBench task-80.

Category: Software Engineering
Task: Please create a program to manage orders and customer inquiries using the Naver Smart Store API.
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


TASK_INSTRUCTION = """Please create a program to manage orders and customer inquiries using the Naver Smart Store API."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves building a functional Python (or similar) program that interfaces with Naver Smart Store API to manage orders and handle customer inquiries.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- API: Naver Smart Store API specifically
- Features: order management AND customer inquiry handling — both required
- Implementation: working code, not just pseudocode or description
- Language: Python preferred; should include API authentication setup

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent produce actual working code?
- Does the code integrate with Naver Smart Store API?
- Are both order management and customer inquiry features implemented?
- Is API authentication handled?
- Was the code saved as a file?

### Step 2: Dimension Scoring

#### A. Api Integration (0.3)
Does the code properly integrate with Naver Smart Store API?

5 — Correct API endpoints used; OAuth/token authentication implemented; API calls are syntactically correct.
4 — API integration present but authentication is incomplete.
3 — API references correct but implementation has significant gaps.
2 — Generic HTTP code without Naver-specific API knowledge.
1 — No API integration.

#### B. Feature Completeness (0.35)
Are both required features implemented?

5 — Order management (list, detail, status update) AND customer inquiry (list, respond) both implemented.
4 — One feature fully implemented; other partially done.
3 — One feature implemented; other missing.
2 — Only basic order listing with no inquiry handling.
1 — Neither feature implemented.

#### C. Code Quality (0.25)
Is the code well-structured and runnable?

5 — Clean, modular code with error handling, docstrings, and configuration management.
4 — Functional code with minor quality issues.
3 — Code runs but is poorly structured.
2 — Code has major bugs or won't run.
1 — No runnable code.

#### D. Delivery (0.1)
Was the code saved as a deliverable file?

5 — Python file saved with clear setup instructions.
4 — Code saved but minimal instructions.
3 — Code provided in response but not saved as file.
2 — Only partial code snippets.
1 — Nothing delivered.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "api_integration": <1-5>,
  "feature_completeness": <1-5>,
  "code_quality": <1-5>,
  "delivery": <1-5>,
  "dimension_reasoning": {{
    "api_integration": "<one sentence citing specific evidence>",
    "feature_completeness": "<one sentence citing specific evidence>",
    "code_quality": "<one sentence citing specific evidence>",
    "delivery": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "api_integration": 0.3,
    "feature_completeness": 0.35,
    "code_quality": 0.25,
    "delivery": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())