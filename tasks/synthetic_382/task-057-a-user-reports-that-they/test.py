"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Investigate compatibility issue of an espresso machine purchased on Amazon and provide a resolution path.
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


TASK_INSTRUCTION = """A user reports that they purchased an espresso machine from Amazon, but the instructions say to use pods only, even though the product description states it is compatible with ground coffee. Investigate the issue using the Amazon product page, customer reviews, and FAQs, and provide a diagnosis and resolution path."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to investigate a compatibility issue with an espresso machine purchased on Amazon. The agent must analyze the product page, customer reviews, and FAQs to diagnose the discrepancy and provide a resolution path. A successful completion involves identifying the root cause of the issue and offering actionable advice.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports that they purchased an espresso machine from Amazon, but the instructions say to use pods only, even though the product description states it is compatible with ground coffee. Investigate the issue using the Amazon product page, customer reviews, and FAQs, and provide a diagnosis and resolution path.

## Task-Specific Constraints
- Must navigate to the Amazon product page and analyze the product description.
- Must review at least 3 customer reviews that discuss compatibility with ground coffee.
- Must check the FAQs section for relevant information.
- Must identify the root cause of the discrepancy (e.g., mislabeling, user error, etc.).
- Must provide a clear resolution path (e.g., contacting support, returning the product, etc.).
- Output must be structured as a diagnosis followed by a resolution path.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Amazon product page and analyze the product description?
- Did the agent review at least 3 customer reviews discussing compatibility with ground coffee?
- Did the agent check the FAQs section for relevant information?
- Did the agent identify the root cause of the discrepancy correctly?
- Is the output structured as a diagnosis followed by a resolution path?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the diagnosis and resolution path are correct and complete.

5 — Identifies the root cause accurately and provides a clear, actionable resolution path.
4 — Identifies the root cause but resolution path is slightly unclear or incomplete.
3 — Partially identifies the root cause or provides a vague resolution path.
2 — Misidentifies the root cause or provides an incorrect resolution path.
1 — Fails to identify the root cause and provides no resolution path.

#### B. Coverage of Sources (0.30)
Measures whether the agent reviewed all required sources (product page, reviews, FAQs).

5 — Reviews all required sources comprehensively (product page, 3+ reviews, FAQs).
4 — Reviews most sources but misses minor details.
3 — Reviews at least two sources but misses key information.
2 — Reviews only one source or misses significant details.
1 — Fails to review any sources.

#### C. Depth of Analysis (0.20)
Measures the specificity and depth of the agent's investigation.

5 — Provides detailed analysis with specific examples from reviews and FAQs.
4 — Provides analysis with some examples but lacks full depth.
3 — Provides general analysis with minimal examples.
2 — Provides shallow analysis with no examples.
1 — Provides no analysis.

#### D. Output Structure and Clarity (0.15)
Measures whether the response is well-organized and easy to understand.

5 — Output is structured clearly with a diagnosis followed by a resolution path.
4 — Output is mostly clear but slightly disorganized.
3 — Output is understandable but poorly structured.
2 — Output is confusing or lacks structure.
1 — Output is incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())