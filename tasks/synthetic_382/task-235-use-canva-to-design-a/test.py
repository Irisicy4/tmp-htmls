"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Design a YouTube thumbnail on Canva with specific visual and textual elements.
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


TASK_INSTRUCTION = """Use Canva to design a thumbnail for a YouTube video titled 'Top 10 Productivity Hacks for Creators'. Include the title in bold text, a vibrant background, and an icon related to productivity. Report the final design configuration."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves designing a YouTube thumbnail using Canva. The thumbnail must include the video title in bold text, a vibrant background, and an icon related to productivity. Successful completion requires the agent to report the final design configuration, including details of the text, background, and icon used.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Canva to design a thumbnail for a YouTube video titled 'Top 10 Productivity Hacks for Creators'. Include the title in bold text, a vibrant background, and an icon related to productivity. Report the final design configuration.

## Task-Specific Constraints
- Must use Canva to create the thumbnail.
- The title 'Top 10 Productivity Hacks for Creators' must be included in bold text.
- The background must be vibrant and visually engaging.
- An icon related to productivity must be included.
- The agent must report the final design configuration, detailing the text, background, and icon used.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Canva to create the thumbnail?
- Is the title 'Top 10 Productivity Hacks for Creators' included in bold text?
- Does the thumbnail have a vibrant background?
- Is there an icon related to productivity included?
- Does the agent provide a clear and complete report of the final design configuration?

### Step 2: Dimension Scoring

#### A. Thumbnail Design Accuracy (0.35)
Measures whether the thumbnail meets the specified design requirements.

5 — All specified elements (bold title, vibrant background, productivity icon) are present and correctly implemented.
4 — One element is slightly incomplete or incorrectly implemented.
3 — At least two elements are present but partially incomplete or incorrect.
2 — Only one element is present or mostly incorrect.
1 — None of the specified elements are present or correct.

#### B. Platform Usage Accuracy (0.30)
Measures whether the agent correctly used Canva to create the thumbnail.

5 — Canva was used, and the tool-call trace confirms appropriate usage steps.
4 — Canva was used, but the tool-call trace shows minor issues or missing steps.
3 — Canva was used, but the tool-call trace is incomplete or unclear.
2 — Canva usage is questionable or poorly documented.
1 — Canva was not used or entirely absent from the tool-call trace.

#### C. Report Completeness (0.20)
Measures whether the agent's report of the final design configuration is clear and complete.

5 — The report includes detailed descriptions of the text, background, and icon used.
4 — The report is mostly complete but lacks minor details.
3 — The report is partially complete but missing significant details.
2 — The report is vague or incomplete.
1 — No report or entirely unclear.

#### D. Creativity and Visual Appeal (0.15)
Measures the creativity and visual appeal of the thumbnail design.

5 — The thumbnail is highly creative and visually appealing, with excellent color and layout choices.
4 — The thumbnail is creative and visually appealing, but slightly less polished.
3 — The thumbnail is moderately creative and visually appealing.
2 — The thumbnail is minimally creative or visually appealing.
1 — The thumbnail lacks creativity and visual appeal.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "thumbnail_design_accuracy": <1-5>,
  "platform_usage_accuracy": <1-5>,
  "report_completeness": <1-5>,
  "creativity_and_visual_appeal": <1-5>,
  "dimension_reasoning": {{
    "thumbnail_design_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_accuracy": "<one sentence citing specific evidence>",
    "report_completeness": "<one sentence citing specific evidence>",
    "creativity_and_visual_appeal": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "thumbnail_design_accuracy": 0.35,
    "platform_usage_accuracy": 0.30,
    "report_completeness": 0.20,
    "creativity_and_visual_appeal": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())