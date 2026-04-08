"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Set up a logo design contest for a fictional bakery named 'Golden Crust Bakery' on 99designs, including industry, style preferences, and example inspirations.
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


TASK_INSTRUCTION = """Navigate the 99designs contest creation workflow to set up a logo design contest for a fictional bakery named 'Golden Crust Bakery.' Select industry, style preferences, and upload example inspirations, and report the final contest preview details shown."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves setting up a logo design contest for a fictional bakery named 'Golden Crust Bakery' on 99designs. The agent must select the industry, specify style preferences, upload example inspirations, and provide the final contest preview details. Successful completion requires accurate setup and clear reporting of the contest preview.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate the 99designs contest creation workflow to set up a logo design contest for a fictional bakery named 'Golden Crust Bakery.' Select industry, style preferences, and upload example inspirations, and report the final contest preview details shown.

## Task-Specific Constraints
- Must navigate the 99designs platform and complete the contest setup workflow.
- Industry must be correctly selected as 'Food & Drink.'
- Style preferences must include at least three specific attributes (e.g., modern, playful, elegant).
- At least two example inspirations must be uploaded from external sources (e.g., pexels.com or coolors.co).
- Final contest preview details must be reported accurately, including budget and description.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to 99designs and complete the contest setup workflow?
- Was the industry correctly selected as 'Food & Drink'?
- Are at least three style preferences specified in the response?
- Are at least two example inspirations uploaded, and are they sourced from valid platforms?
- Is the final contest preview accurately reported, including budget and description?

### Step 2: Dimension Scoring

#### A. Contest Setup Accuracy (0.35)
Measures whether the contest setup workflow was completed correctly and the preview details are accurate.

5 — Contest setup completed fully with correct industry, style preferences, inspirations, and preview details.
4 — Contest setup mostly correct but with minor errors in one area (e.g., missing one style preference).
3 — Contest setup partially correct but missing multiple required elements.
2 — Contest setup attempted but mostly incorrect or incomplete.
1 — Contest setup not attempted or completely incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent used the required platforms and uploaded inspirations from valid sources.

5 — All required platforms visited and at least two inspirations uploaded from valid sources.
4 — Most required platforms visited with one inspiration missing or sourced incorrectly.
3 — Some required platforms visited but multiple inspirations missing or sourced incorrectly.
2 — Few required platforms visited with minimal effort to source inspirations.
1 — No required platforms visited or inspirations uploaded.

#### C. Style Specificity (0.20)
Measures the depth and specificity of style preferences provided.

5 — At least three specific style preferences provided with clear descriptions.
4 — Two specific style preferences provided with clear descriptions.
3 — One specific style preference provided or vague descriptions.
2 — Style preferences mentioned but unclear or generic.
1 — No style preferences provided.

#### D. Output Structure and Reporting (0.15)
Measures the organization and clarity of the final response.

5 — Response is well-organized, clearly structured, and includes all required details.
4 — Response is mostly well-organized but missing minor details.
3 — Response is partially organized but missing significant details.
2 — Response is poorly organized and unclear.
1 — Response is unstructured and missing most details.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "contest_setup_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "style_specificity": <1-5>,
  "output_structure_and_reporting": <1-5>,
  "dimension_reasoning": {{
    "contest_setup_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "style_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_reporting": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "contest_setup_accuracy": 0.35,
    "platform_coverage": 0.30,
    "style_specificity": 0.20,
    "output_structure_and_reporting": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())