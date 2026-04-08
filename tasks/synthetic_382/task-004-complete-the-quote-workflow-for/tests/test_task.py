"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Complete the quote workflow for a custom-designed gaming PC on Origin PC's custom configurator with specific components and report the final configuration and price.
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


TASK_INSTRUCTION = """Complete the quote workflow for a custom-designed gaming PC on Origin PC's custom configurator. Choose AMD Ryzen 7 processor, NVIDIA GeForce RTX 3060 graphics card, 16GB RAM, and a 1TB SSD. Apply any available promotions or discounts visible on the product page. Report the final configuration details and the quoted price."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves using Origin PC's custom configurator to build a gaming PC with specific components (AMD Ryzen 7 processor, NVIDIA GeForce RTX 3060 graphics card, 16GB RAM, and a 1TB SSD). The agent must apply any visible promotions or discounts and report the final configuration and quoted price.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Complete the quote workflow for a custom-designed gaming PC on Origin PC's custom configurator. Choose AMD Ryzen 7 processor, NVIDIA GeForce RTX 3060 graphics card, 16GB RAM, and a 1TB SSD. Apply any available promotions or discounts visible on the product page. Report the final configuration details and the quoted price.

## Task-Specific Constraints
- Must use the Origin PC custom configurator to complete the task.
- Must select the specified components: AMD Ryzen 7 processor, NVIDIA GeForce RTX 3060 graphics card, 16GB RAM, and a 1TB SSD.
- Must apply any visible promotions or discounts available on the platform.
- Must report the final configuration details in a structured format (e.g., list or table).
- Must include the quoted price in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use the Origin PC custom configurator to complete the task?
- Did the agent select the correct components (AMD Ryzen 7, NVIDIA GeForce RTX 3060, 16GB RAM, 1TB SSD)?
- Did the agent apply any visible promotions or discounts, if available?
- Is the final configuration reported in a structured format (e.g., list or table)?
- Is the quoted price included and does it match the selected configuration?

### Step 2: Dimension Scoring

#### A. Configuration Accuracy (0.35)
Measures whether the agent selected the correct components.

5 — All specified components are correctly selected.
4 — One component is incorrect or missing.
3 — Two components are incorrect or missing.
2 — Most components are incorrect or missing.
1 — No attempt to select the correct components.

#### B. Promotion Application (0.30)
Measures whether the agent applied visible promotions or discounts.

5 — All applicable promotions/discounts are correctly applied.
4 — Most promotions/discounts are applied, but one is missed.
3 — Some promotions/discounts are applied, but several are missed.
2 — Few promotions/discounts are applied.
1 — No promotions/discounts are applied.

#### C. Reporting Completeness (0.20)
Measures whether the agent reported the final configuration and price.

5 — Both the configuration and price are fully reported in a structured format.
4 — Both are reported but with minor formatting or completeness issues.
3 — One is missing or incomplete.
2 — Both are incomplete or poorly formatted.
1 — No configuration or price is reported.

#### D. Output Structure and Clarity (0.15)
Measures the organization and clarity of the output.

5 — Output is well-organized, clear, and easy to follow.
4 — Output is mostly clear with minor organizational issues.
3 — Output is somewhat unclear or disorganized.
2 — Output is poorly organized and difficult to follow.
1 — Output is completely unclear or unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "configuration_accuracy": <1-5>,
  "promotion_application": <1-5>,
  "reporting_completeness": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "configuration_accuracy": "<one sentence citing specific evidence>",
    "promotion_application": "<one sentence citing specific evidence>",
    "reporting_completeness": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "configuration_accuracy": 0.35,
    "promotion_application": 0.30,
    "reporting_completeness": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())