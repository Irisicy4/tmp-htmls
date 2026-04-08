"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Configure a mock promotional image for a clothing store sale using Photopea.
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


TASK_INSTRUCTION = """Go to Photopea.com and configure a mock promotional image for a clothing store sale: upload a free T-shirt template, add a '50% OFF' badge, apply text overlays for the store name, and save the configured image with layers intact in PSD format."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a promotional image for a clothing store sale using Photopea. The agent must upload a free T-shirt template, add a '50% OFF' badge, apply text overlays for the store name, and save the image in PSD format with layers intact.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Photopea.com and configure a mock promotional image for a clothing store sale: upload a free T-shirt template, add a '50% OFF' badge, apply text overlays for the store name, and save the configured image with layers intact in PSD format.

## Task-Specific Constraints
- Must use Photopea.com as the platform for image editing.
- Must upload a free T-shirt template.
- Must add a visible '50% OFF' badge to the image.
- Must include text overlays for the store name.
- Must save the image in PSD format with layers intact.
- Must ensure the final image is visually coherent and professional-looking.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Photopea.com and use it for image editing?
- Did the agent upload a free T-shirt template?
- Is there a visible '50% OFF' badge in the image?
- Are text overlays for the store name present and readable?
- Was the image saved in PSD format with layers intact?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the promotional image meets all specified requirements.

5 — Image includes T-shirt template, '50% OFF' badge, store name text overlays, and is saved in PSD format with layers intact.
4 — Image includes most elements but may miss one minor requirement.
3 — Image is partially complete, missing multiple elements or saved incorrectly.
2 — Image is mostly incomplete or saved in the wrong format.
1 — No promotional image created or completely incorrect.

#### B. Coverage of Requirements (0.30)
Measures whether all specified requirements were addressed.

5 — All requirements (platform usage, badge, text overlays, PSD format) are fully addressed.
4 — Most requirements are addressed, with minor omissions.
3 — Some requirements are addressed, but key elements are missing.
2 — Few requirements are addressed, with major omissions.
1 — No requirements are addressed.

#### C. Visual Coherence (0.20)
Measures the professionalism and visual coherence of the final image.

5 — Image is visually professional, coherent, and well-designed.
4 — Image is visually acceptable but lacks polish or minor coherence issues.
3 — Image is somewhat coherent but has noticeable design flaws.
2 — Image is poorly designed or visually incoherent.
1 — Image is completely unprofessional or incoherent.

#### D. Execution Trace Quality (0.15)
Measures whether the agent's tool-call trace demonstrates effective use of Photopea.

5 — Tool-call trace shows effective and logical use of Photopea for all required steps.
4 — Tool-call trace shows mostly effective use of Photopea with minor inefficiencies.
3 — Tool-call trace shows partial use of Photopea but misses key steps.
2 — Tool-call trace shows poor use of Photopea with major inefficiencies.
1 — Tool-call trace shows no meaningful use of Photopea.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent successfully navigated to Photopea, uploaded a T-shirt template, added a '50% OFF' badge, included text overlays for the store name, and saved the image in PSD format. Minor design flaws were noted.",
  "primary_deliverable_accuracy": 4,
  "coverage_of_requirements": 5,
  "visual_coherence": 4,
  "execution_trace_quality": 4,
  "dimension_reasoning": {
    "primary_deliverable_accuracy": "The image includes most required elements but has minor omissions.",
    "coverage_of_requirements": "All specified requirements were addressed.",
    "visual_coherence": "The image is visually acceptable but lacks polish.",
    "execution_trace_quality": "The tool-call trace shows effective use of Photopea with minor inefficiencies."
  },
  "overall_score": 4.25,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "visual_coherence": 0.20,
    "execution_trace_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())