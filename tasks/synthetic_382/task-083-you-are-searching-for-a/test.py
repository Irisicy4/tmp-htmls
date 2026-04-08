"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Compare the top 3 espresso machines under $300 from Amazon, Best Buy, and Walmart based on price, customer ratings, warranty, and features, and provide a comparison chart.
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


TASK_INSTRUCTION = """You are searching for a high-quality espresso machine under $300. Compare the top 3 espresso machines available on Amazon, Best Buy, and Walmart based on their price, customer ratings, warranty period, and additional features (e.g. milk frother, programmable settings). Provide a comparison chart summarizing your findings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves comparing the top 3 espresso machines under $300 from Amazon, Best Buy, and Walmart. The agent must provide a comparison chart summarizing price, customer ratings, warranty period, and additional features. A successful completion includes visiting all platforms, gathering accurate data, and presenting it in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
You are searching for a high-quality espresso machine under $300. Compare the top 3 espresso machines available on Amazon, Best Buy, and Walmart based on their price, customer ratings, warranty period, and additional features (e.g. milk frother, programmable settings). Provide a comparison chart summarizing your findings.

## Task-Specific Constraints
- Must visit Amazon, Best Buy, and Walmart.
- Must include price, customer ratings, warranty period, and features for all items compared.
- Output must be organized as a table or structured list.
- Must compare at least 3 espresso machines per platform.
- Must ensure all data is accurate and sourced from the platforms visited.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Best Buy, and Walmart? Which platforms were actually visited?
- Are price, customer ratings, warranty period, and features present for all items compared?
- Is the output organized as a table or structured list?
- Did the agent compare at least 3 espresso machines per platform?
- Are the data points accurate and sourced from the platforms visited?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the comparison chart is correct, complete, and matches the task requirements.

5 — Includes accurate data for all required attributes (price, ratings, warranty, features) for 3 machines per platform.
4 — Includes most required attributes but may miss minor details or have slight inaccuracies.
3 — Includes partial data; some attributes or machines missing.
2 — Includes minimal data; most attributes or machines missing.
1 — No usable data provided.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and gathered data from each.

5 — Data gathered from Amazon, Best Buy, and Walmart for 3 machines each.
4 — Data gathered from all platforms but fewer than 3 machines per platform.
3 — Data gathered from 2 platforms with partial machine coverage.
2 — Data gathered from only 1 platform.
1 — No data gathered from any platform.

#### C. Depth of Comparison (0.25)
Measures the level of detail in the comparison, including specific features and warranty details.

5 — Detailed comparison including specific features (e.g., milk frother, programmable settings) and warranty periods.
4 — Comparison includes most features and warranty details but lacks minor specifics.
3 — Comparison includes basic features but lacks depth or warranty details.
2 — Minimal comparison with few features mentioned.
1 — No meaningful comparison provided.

#### D. Output Structure and Organization (0.10)
Measures whether the output is well-organized and presented in the required format.

5 — Output is a clear, structured table or list with all required data points.
4 — Output is mostly structured but may have minor formatting issues.
3 — Output is partially structured but lacks clarity or completeness.
2 — Output is disorganized and hard to interpret.
1 — Output is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "The agent visited Amazon, Best Buy, and Walmart, gathered data for 3 machines per platform, and presented it in a structured table. Minor inaccuracies were found in warranty details.",
  "deliverable_accuracy": 4,
  "coverage_of_platforms": 5,
  "depth_of_comparison": 4,
  "output_structure_and_organization": 5,
  "dimension_reasoning": {{
    "deliverable_accuracy": "Most required attributes were included, but warranty details had minor inaccuracies.",
    "coverage_of_platforms": "Data was gathered from all three platforms for 3 machines each.",
    "depth_of_comparison": "Specific features and warranty details were included but lacked minor specifics.",
    "output_structure_and_organization": "The output was presented as a clear and structured table."
  }},
  "overall_score": 4.35,
  "passed": true
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())