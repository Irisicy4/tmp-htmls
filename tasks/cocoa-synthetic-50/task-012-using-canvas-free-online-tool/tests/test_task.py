"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Create a YouTube video thumbnail mockup using Canva with specific elements.
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


TASK_INSTRUCTION = """Using Canva's free online tool, create a mockup of a YouTube video thumbnail for a video titled 'Top 5 Video Editing Tips in 2023.' Include a catchy headline, a relevant background image, and branded elements (e.g., logo or banner). Use free templates and images available on Canva."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a mockup of a YouTube video thumbnail using Canva. The thumbnail must include a catchy headline, a relevant background image, and branded elements such as a logo or banner. The agent must use free templates and images available on Canva.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Canva's free online tool, create a mockup of a YouTube video thumbnail for a video titled 'Top 5 Video Editing Tips in 2023.' Include a catchy headline, a relevant background image, and branded elements (e.g., logo or banner). Use free templates and images available on Canva.

## Task-Specific Constraints
- Must use Canva as the primary platform.
- Must include a headline relevant to the video title.
- Must include a background image related to video editing.
- Must include branded elements such as a logo or banner.
- Must use free templates and images available on Canva.
- Final response must describe the thumbnail design clearly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Canva as the primary platform?
- Does the response include a headline relevant to the video title?
- Does the response describe a background image related to video editing?
- Are branded elements such as a logo or banner included in the description?
- Did the agent use free templates and images available on Canva?

### Step 2: Dimension Scoring

#### A. Thumbnail Design Accuracy (0.35)
Measures whether the thumbnail design matches the task requirements.

5 — Includes all required elements: headline, background image, branded elements, and uses free Canva templates.
4 — Includes most required elements but misses one minor detail.
3 — Includes some required elements but is incomplete or unclear.
2 — Includes few required elements and is mostly incorrect.
1 — Includes none of the required elements.

#### B. Platform Usage (0.30)
Measures whether the agent correctly used Canva and adhered to constraints.

5 — Clearly used Canva as the primary platform and adhered to all constraints.
4 — Used Canva but missed one constraint (e.g., free templates).
3 — Used Canva but missed multiple constraints.
2 — Used Canva incorrectly or minimally.
1 — Did not use Canva.

#### C. Content Relevance (0.20)
Measures whether the thumbnail content is relevant to the video topic.

5 — Content is highly relevant to 'Top 5 Video Editing Tips in 2023.'
4 — Content is mostly relevant but slightly generic.
3 — Content is somewhat relevant but lacks specificity.
2 — Content is minimally relevant or off-topic.
1 — Content is completely irrelevant.

#### D. Response Clarity (0.15)
Measures whether the agent's response clearly describes the thumbnail design.

5 — Response is detailed and clearly describes all design elements.
4 — Response is mostly clear but misses minor details.
3 — Response is somewhat clear but lacks detail.
2 — Response is unclear or poorly structured.
1 — Response is completely unclear.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent used Canva and included a headline, background image, and branded elements. However, the response lacked clarity on whether free templates were used.",
  "thumbnail_design_accuracy": 4,
  "platform_usage": 4,
  "content_relevance": 4,
  "response_clarity": 3,
  "dimension_reasoning": {
    "thumbnail_design_accuracy": "The response includes most required elements but misses clarity on free template usage.",
    "platform_usage": "Canva was used as the primary platform, but adherence to all constraints is unclear.",
    "content_relevance": "The content is mostly relevant to the video topic.",
    "response_clarity": "The response is somewhat clear but lacks detail."
  },
  "overall_score": 3.85,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "thumbnail_design_accuracy": 0.35,
    "platform_usage": 0.30,
    "content_relevance": 0.20,
    "response_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())