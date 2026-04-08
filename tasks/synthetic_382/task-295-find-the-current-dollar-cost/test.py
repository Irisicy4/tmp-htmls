"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Determine the cost efficiency of three large language models for processing 10,000 inputs of 1,000 tokens each.
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


TASK_INSTRUCTION = """Find the current dollar cost per inference for three large language models: GPT-4, Claude 2, and Llama 2, using their API pricing pages and documentation. Calculate which model provides the best cost efficiency for processing 10,000 inputs of 1,000 tokens each."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves determining the cost efficiency of three large language models (GPT-4, Claude 2, and Llama 2) for processing 10,000 inputs of 1,000 tokens each. The agent must navigate to the pricing pages of openai.com, anthropic.com, and huggingface.co to gather accurate data. A successful completion includes structured output comparing costs and identifying the most cost-efficient model.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find the current dollar cost per inference for three large language models: GPT-4, Claude 2, and Llama 2, using their API pricing pages and documentation. Calculate which model provides the best cost efficiency for processing 10,000 inputs of 1,000 tokens each.

## Task-Specific Constraints
- Must visit openai.com, anthropic.com, and huggingface.co to gather pricing data.
- Must include cost-per-inference data for GPT-4, Claude 2, and Llama 2.
- Output must be organized as a structured table or list comparing costs.
- Must calculate the total cost for processing 10,000 inputs of 1,000 tokens for each model.
- Must identify the most cost-efficient model based on the calculations.
- Must cite sources for all pricing data used in the calculations.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to openai.com, anthropic.com, and huggingface.co? Which platforms were actually visited?
- Are cost-per-inference data for GPT-4, Claude 2, and Llama 2 present in the response?
- Is the output organized as a structured table or list comparing costs?
- Are the calculations for processing 10,000 inputs of 1,000 tokens accurate and based on cited data?
- Did the agent correctly identify the most cost-efficient model based on the calculations?

### Step 2: Dimension Scoring

#### A. Cost Calculation Accuracy (0.35)
Measures whether the agent correctly calculated the total cost for processing 10,000 inputs of 1,000 tokens for each model.

5 — All calculations are accurate and based on cited data.
4 — Minor errors in calculations or missing citations for one model.
3 — Partial calculations provided, but missing key data or citations.
2 — Major errors in calculations or missing data for multiple models.
1 — No calculations attempted or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and gathered pricing data from them.

5 — Pricing data gathered from all three platforms (openai.com, anthropic.com, huggingface.co).
4 — Data gathered from two platforms, with minor omissions.
3 — Data gathered from one platform, or incomplete data from multiple platforms.
2 — Minimal data gathered, major omissions.
1 — No data gathered from any platform.

#### C. Output Structure and Comparisons (0.20)
Measures whether the output is well-organized and includes structured comparisons.

5 — Output is organized as a clear table or list with detailed comparisons.
4 — Output is structured but lacks some detail or clarity.
3 — Output is partially structured but incomplete or unclear.
2 — Output is mostly unstructured or difficult to interpret.
1 — Output is completely unstructured or absent.

#### D. Source Credibility (0.15)
Measures whether the agent cited credible sources for all pricing data.

5 — All pricing data is sourced and citations are accurate.
4 — Most pricing data is sourced, with minor citation issues.
3 — Some pricing data is sourced, but citations are incomplete.
2 — Minimal sourcing provided, major credibility issues.
1 — No sourcing provided or sources are not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited openai.com and anthropic.com but did not navigate to huggingface.co. Pricing data for GPT-4 and Claude 2 is present, but Llama 2 data is missing. Calculations for GPT-4 and Claude 2 are accurate, but the output lacks structured comparisons.",
  "cost_calculation_accuracy": 3,
  "platform_coverage": 3,
  "output_structure_and_comparisons": 2,
  "source_credibility": 3,
  "dimension_reasoning": {
    "cost_calculation_accuracy": "Calculations for GPT-4 and Claude 2 are accurate, but Llama 2 data is missing.",
    "platform_coverage": "The agent visited openai.com and anthropic.com but did not navigate to huggingface.co.",
    "output_structure_and_comparisons": "The output lacks structured comparisons and is partially organized.",
    "source_credibility": "Most pricing data is sourced, but Llama 2 data is missing."
  },
  "overall_score": 2.85,
  "passed": false
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "cost_calculation_accuracy": 0.35,
    "platform_coverage": 0.30,
    "output_structure_and_comparisons": 0.20,
    "source_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())