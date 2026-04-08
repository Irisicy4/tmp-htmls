"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Research and compare smart TVs between 50 to 55 inches under $500 across Amazon, Best Buy, and Walmart, and identify 3 to 5 models with their prices, pros, and cons.
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


TASK_INSTRUCTION = """I’m looking to buy a smart TV between 50 to 55 inches under $500. Research options on Amazon, Best Buy, and Walmart, compare their features (resolution, refresh rate, smart platform), and identify 3 to 5 models with their prices, pros, and cons."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching smart TVs between 50 to 55 inches under $500 on Amazon, Best Buy, and Walmart. The agent must compare features such as resolution, refresh rate, and smart platform, and provide a list of 3 to 5 models with their prices, pros, and cons. A successful completion requires accurate data, proper platform usage, and organized output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
I’m looking to buy a smart TV between 50 to 55 inches under $500. Research options on Amazon, Best Buy, and Walmart, compare their features (resolution, refresh rate, smart platform), and identify 3 to 5 models with their prices, pros, and cons.

## Task-Specific Constraints
- Must visit Amazon, Best Buy, and Walmart.
- Must include price data for all models compared.
- Must compare at least resolution, refresh rate, and smart platform for each model.
- Must identify 3 to 5 models with their pros and cons.
- Output must be organized as a structured list or table.
- Must avoid including models outside the specified size or price range.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Best Buy, and Walmart? Which platforms were actually visited?
- Does the response include 3 to 5 models within the specified size and price range?
- Are resolution, refresh rate, and smart platform compared for each model?
- Are prices, pros, and cons provided for each model?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent identified 3 to 5 models within the specified size and price range, with accurate details.

5 — Identifies 5 models within range, all details accurate.
4 — Identifies 4 models within range, most details accurate.
3 — Identifies 3 models within range, some details accurate.
2 — Identifies fewer than 3 models or many inaccuracies.
1 — No valid models identified.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Visited all 3 platforms and used data from each.
4 — Visited all 3 platforms but used data from only 2.
3 — Visited 2 platforms and used data from both.
2 — Visited only 1 platform or used data from only 1.
1 — Did not visit any required platforms.

#### C. Depth of Comparison (0.20)
Measures the depth and specificity of the comparisons made between models.

5 — Provides detailed comparisons for all required features (resolution, refresh rate, smart platform) for all models.
4 — Provides detailed comparisons for most required features for most models.
3 — Provides basic comparisons for required features for most models.
2 — Provides minimal or incomplete comparisons.
1 — Provides no meaningful comparisons.

#### D. Output Organization (0.15)
Measures whether the output is well-structured and easy to understand.

5 — Output is a clear, well-organized table or structured list.
4 — Output is mostly clear and organized, with minor issues.
3 — Output is somewhat organized but has noticeable issues.
2 — Output is poorly organized or hard to follow.
1 — Output is unstructured or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_comparison": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())