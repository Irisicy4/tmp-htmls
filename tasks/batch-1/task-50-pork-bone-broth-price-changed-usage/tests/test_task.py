"""
LLM-as-judge evaluator for EvolveBench task-50.

Category: Finance & Economics
Task: Persuasive cost analysis for broth product price change with new dilution ratio.
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

def _call(agent_response, execution_summary, system_prompt, user_prompt_template, task_instruction):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(
                    task_instruction=task_instruction,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes, dimensions, weights, pass_threshold):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dimensions)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in dimensions}
    overall = sum(aggregated[d] * weights[d] for d in dimensions)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= pass_threshold
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated


TASK_INSTRUCTION = """The price we receive from the manufacturer for the following product has changed. The usage amount has also changed, so I need a persuasive explanation of whether the actual unit cost for our store has gone up or down.

Product name: Pork bone broth (domestic)
Weight: 3kg
Original price (VAT excluded): 3,700 won/3kg, mixed with water at 1:1 ratio (2x dilution)
New price (VAT excluded): 5,600 won/3kg, mixed with water at 3:1 ratio (3x dilution)"""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent correctly computed the actual unit cost impact of a price change combined with a dilution ratio change, and presented a persuasive argument."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Math must be correct: original cost per unit = 3700/2 = 1850 won/kg-equivalent; new cost per unit = 5600/3 ≈ 1867 won/kg-equivalent — the actual unit cost increased slightly
- The agent must compute and present the per-unit cost, not just compare raw prices
- The argument must be persuasive and logically structured
- The conclusion must be mathematically accurate

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent compute the actual per-unit cost for both old and new pricing?
- Is the math correct (1850 vs 1867 won per unit)?
- Did the agent present a persuasive argument?
- Is the conclusion accurate (slight cost increase)?

### Step 2: Dimension Scoring

#### A. Mathematical Accuracy (0.4)
Is the per-unit cost calculation correct?

5 — Correctly computes: old = 3700÷2 = 1850 won/unit, new = 5600÷3 ≈ 1867 won/unit; concludes cost increased.
4 — Correct approach but minor rounding error or slightly different framing.
3 — Partially correct — identifies the dilution ratio matters but makes a computational error.
2 — Compares raw prices (3700 vs 5600) without accounting for dilution ratio.
1 — No calculation or clearly wrong answer.

#### B. Persuasive Logic (0.3)
Is the argument well-structured and convincing?

5 — Clear logical flow: states the problem, shows calculation, explains implication, draws conclusion.
4 — Mostly persuasive but one step is weak or unclear.
3 — Argument present but logic gaps make it less convincing.
2 — Descriptive rather than persuasive.
1 — No argument structure.

#### C. Practical Framing (0.2)
Is the output framed in a way useful for a store owner?

5 — Uses business language, explains impact per serving/use, gives actionable insight.
4 — Good framing but slightly too technical or academic.
3 — Correct info but not framed for a store owner audience.
2 — Very generic framing.
1 — No practical framing.

#### D. Output Clarity (0.1)
Is the response clear and easy to understand?

5 — Well-organized, concise, easy to read.
4 — Clear but slightly verbose.
3 — Understandable but requires effort.
2 — Confusing or poorly organized.
1 — Incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "mathematical_accuracy": <1-5>,
  "persuasive_logic": <1-5>,
  "practical_framing": <1-5>,
  "output_clarity": <1-5>,
  "dimension_reasoning": {{
    "mathematical_accuracy": "<one sentence citing specific evidence>",
    "persuasive_logic": "<one sentence citing specific evidence>",
    "practical_framing": "<one sentence citing specific evidence>",
    "output_clarity": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "mathematical_accuracy": 0.4,
    "persuasive_logic": 0.3,
    "practical_framing": 0.2,
    "output_clarity": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first,
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION),
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)],
                           DIMENSIONS, DIMENSION_WEIGHTS, PASS_THRESHOLD)
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