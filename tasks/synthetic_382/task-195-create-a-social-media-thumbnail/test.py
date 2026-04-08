"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Create a social media thumbnail mockup for a YouTube video using Canva, incorporating specific design elements.
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


TASK_INSTRUCTION = """Create a social media thumbnail mockup for an upcoming YouTube video titled 'Top 10 AI Tools for Creators' using Canva's free templates. Include a bright color palette, bold text, and relevant imagery (e.g., AI tools logos or abstract technology visuals). Save the thumbnail design and export it as a PNG."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a social media thumbnail mockup for a YouTube video titled 'Top 10 AI Tools for Creators' using Canva. The design must include a bright color palette, bold text, and relevant imagery such as AI tools logos or abstract technology visuals. The deliverable is a PNG file of the thumbnail design.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a social media thumbnail mockup for an upcoming YouTube video titled 'Top 10 AI Tools for Creators' using Canva's free templates. Include a bright color palette, bold text, and relevant imagery (e.g., AI tools logos or abstract technology visuals). Save the thumbnail design and export it as a PNG.

## Task-Specific Constraints
- Must use Canva's free templates to create the thumbnail.
- Must include a bright color palette in the design.
- Must include bold text with the title 'Top 10 AI Tools for Creators'.
- Must incorporate relevant imagery such as AI tools logos or abstract technology visuals.
- Must export the thumbnail design as a PNG file.
- Must visit at least one additional platform (e.g., Unsplash or Google) to source imagery.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Canva and use its free templates?
- Did the agent include a bright color palette in the design?
- Did the agent include bold text with the correct title?
- Did the agent incorporate relevant imagery (e.g., AI tools logos or abstract visuals)?
- Did the agent export the thumbnail design as a PNG file?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the thumbnail design meets the task requirements.

5 — Thumbnail includes all specified elements: bright color palette, bold text with the correct title, relevant imagery, and is exported as PNG.
4 — Thumbnail includes most elements but is missing one minor detail (e.g., imagery slightly off-topic).
3 — Thumbnail includes some elements but is missing key details (e.g., incorrect title or missing imagery).
2 — Thumbnail is mostly incomplete or incorrect.
1 — Thumbnail is absent or completely wrong.

#### B. Platform Usage Coverage (0.30)
Measures whether the agent used all required platforms effectively.

5 — Agent used Canva and at least one additional platform (e.g., Unsplash or Google) to source imagery.
4 — Agent used Canva and attempted to use another platform but was incomplete.
3 — Agent used Canva but did not use any additional platforms.
2 — Agent attempted Canva but failed to complete the task.
1 — Agent did not use Canva or any other platform.

#### C. Design Specificity (0.20)
Measures the depth and specificity of the design elements.

5 — Design includes highly specific and relevant imagery (e.g., recognizable AI tools logos or detailed abstract visuals).
4 — Design includes somewhat specific imagery but lacks detail.
3 — Design includes generic imagery with minimal relevance.
2 — Design includes poor or irrelevant imagery.
1 — Design lacks any imagery.

#### D. Output Structure and Quality (0.15)
Measures the organization and quality of the output.

5 — PNG file is properly exported, and the design is visually well-organized.
4 — PNG file is exported but design has minor organizational flaws.
3 — PNG file is exported but design is poorly organized.
2 — PNG file is missing or design is disorganized.
1 — No output file or design.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_usage_coverage": <1-5>,
  "design_specificity": <1-5>,
  "output_structure_and_quality": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_coverage": "<one sentence citing specific evidence>",
    "design_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_usage_coverage": 0.30,
    "design_specificity": 0.20,
    "output_structure_and_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())