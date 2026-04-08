"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Use Sephora's foundation shade match tool to find and extract details of three recommended foundations.
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


TASK_INSTRUCTION = """Visit the Sephora website and use their foundation shade match tool. Input 'light skin' with neutral undertones and select three foundations recommended by the tool. Complete the process and extract the recommended product names, prices, and shade details from the final screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to use Sephora's foundation shade match tool to find three recommended foundations for 'light skin' with neutral undertones. The agent must extract the product names, prices, and shade details from the final screen. This task is in the shopping domain and requires structured output with specific details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit the Sephora website and use their foundation shade match tool. Input 'light skin' with neutral undertones and select three foundations recommended by the tool. Complete the process and extract the recommended product names, prices, and shade details from the final screen.

## Task-Specific Constraints
- Must use Sephora's foundation shade match tool.
- Must input 'light skin' with neutral undertones as specified.
- Must select exactly three foundations from the recommendations.
- Must extract product names, prices, and shade details for all three foundations.
- Output must be organized as a structured list or table.
- Must accurately reflect the data from the final screen.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Sephora's foundation shade match tool?
- Did the agent input 'light skin' with neutral undertones as specified?
- Did the agent select exactly three foundations from the recommendations?
- Are the product names, prices, and shade details present in the response?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the extracted product names, prices, and shade details are correct and complete.

5 — All three foundations are correctly identified with accurate names, prices, and shade details.
4 — All three foundations are identified but one or more details are slightly inaccurate.
3 — At least two foundations are identified with partial details.
2 — Only one foundation is identified or most details are missing.
1 — No foundations or details are correctly identified.

#### B. Coverage of Requirements (0.30)
Measures whether the agent followed all task constraints.

5 — All constraints are fully satisfied (tool used, correct input, three foundations selected).
4 — Most constraints are satisfied but one minor deviation is present.
3 — Partial satisfaction of constraints (e.g., incorrect input or fewer than three foundations selected).
2 — Major deviations from constraints (e.g., wrong tool or no foundations selected).
1 — None of the constraints are satisfied.

#### C. Specificity of Details (0.25)
Measures the depth and specificity of the extracted information.

5 — All details (names, prices, shades) are precise and complete.
4 — Most details are precise but one or two minor omissions exist.
3 — Some details are present but lack precision or completeness.
2 — Few details are present and most are vague or incorrect.
1 — No meaningful details are present.

#### D. Output Structure (0.10)
Measures the organization and readability of the output.

5 — Output is well-organized as a structured list or table with clear formatting.
4 — Output is organized but formatting is slightly unclear or inconsistent.
3 — Output is partially organized but lacks clarity or structure.
2 — Output is poorly organized and hard to interpret.
1 — Output is completely disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "specificity_of_details": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "specificity_of_details": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "specificity_of_details": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())