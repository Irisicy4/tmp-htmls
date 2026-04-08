"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Create a social media post mockup promoting a food festival using Photopea or Canva.
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


TASK_INSTRUCTION = """Using Photopea or Canva, create a social media post mockup promoting a food festival. Include a vibrant color palette inspired by traditional dishes, use at least one licensed image of food, and add text for the event name, date, and location."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a social media post mockup promoting a food festival using Photopea or Canva. A successful completion must include a vibrant color palette inspired by traditional dishes, at least one licensed image of food, and text for the event name, date, and location.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Photopea or Canva, create a social media post mockup promoting a food festival. Include a vibrant color palette inspired by traditional dishes, use at least one licensed image of food, and add text for the event name, date, and location.

## Task-Specific Constraints
- Must use either Photopea or Canva to create the mockup.
- Must include a vibrant color palette inspired by traditional dishes.
- Must use at least one licensed image of food.
- Must include text for the event name, date, and location.
- Final output must resemble a professional social media post mockup.
- Must provide evidence of platform usage in the tool-call trace.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to either Photopea or Canva? Was the platform usage evident in the tool-call trace?
- Does the mockup include a vibrant color palette inspired by traditional dishes?
- Is there at least one licensed image of food present in the mockup?
- Does the mockup include text for the event name, date, and location?
- Does the final output resemble a professional social media post mockup?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the final mockup meets the task requirements.

5 — Mockup includes all required elements: vibrant color palette, licensed food image, event name, date, and location.
4 — Mockup includes most required elements but misses one minor detail.
3 — Mockup includes some required elements but lacks key details.
2 — Mockup is missing most required elements or is poorly executed.
1 — No mockup or completely irrelevant output.

#### B. Platform Usage (0.30)
Measures whether the agent used the required platforms correctly.

5 — Clear evidence of platform usage (Photopea or Canva) with relevant actions logged.
4 — Evidence of platform usage but with minor gaps or unclear actions.
3 — Partial evidence of platform usage, with key actions missing.
2 — Minimal evidence of platform usage or incorrect platform used.
1 — No evidence of platform usage.

#### C. Creative Quality (0.20)
Measures the creativity and visual appeal of the mockup.

5 — Mockup is highly creative, visually appealing, and aligns with the food festival theme.
4 — Mockup is creative and visually appealing but lacks minor thematic alignment.
3 — Mockup is somewhat creative but lacks polish or thematic relevance.
2 — Mockup is minimally creative or visually unappealing.
1 — Mockup is not creative or visually relevant.

#### D. Evidence Organization (0.15)
Measures the organization and clarity of the agent's response and tool-call trace.

5 — Response and tool-call trace are well-organized, detailed, and easy to follow.
4 — Response and tool-call trace are mostly organized with minor gaps.
3 — Response and tool-call trace are partially organized but lack clarity.
2 — Response and tool-call trace are poorly organized or unclear.
1 — Response and tool-call trace are disorganized or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_usage": <1-5>,
  "creative_quality": <1-5>,
  "evidence_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "creative_quality": "<one sentence citing specific evidence>",
    "evidence_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_usage": 0.30,
    "creative_quality": 0.20,
    "evidence_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())