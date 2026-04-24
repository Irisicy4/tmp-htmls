"""
LLM-as-judge evaluator for EvolveBench task-11-apply-the-following-effects-in.

Category: GIF Creation
Task: Apply the following effects in order to the original image to create a GIF.
  1. The image fades in and is revealed slowly.
  2. The image rotates one rotation clockwise.
  3. The heartbeat effect imp
"""

import os, json, re

TASK_INSTRUCTION = """Apply the following effects in order to the original image to create a GIF.
  1. The image fades in and is revealed slowly.
  2. The image rotates one rotation clockwise.
  3. The heartbeat effect implemented earlier appears once.
  4. The image fades out and slowly disappears.
  
  Let each movement last for 1 second."""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """The judge is evaluating the agent's ability to correctly apply specified visual effects to an image in sequence, ensuring the resulting GIF adheres to the task instructions. The evaluation focuses on the accuracy of the effects, their timing, and the overall coherence of the animation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- The GIF must include all four specified effects in the correct order: fade-in, clockwise rotation, heartbeat effect, fade-out.
- Each effect must last exactly 1 second.
- The heartbeat effect must match the implementation described in earlier instructions.
- The transitions between effects must be smooth and visually coherent.
- The final GIF must loop seamlessly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Does the GIF include all four effects in the correct order?
- Does each effect last exactly 1 second?
- Is the heartbeat effect implemented as described in earlier instructions?
- Are the transitions between effects smooth and visually coherent?
- Does the GIF loop seamlessly without visible glitches?

### Step 2: Dimension Scoring

#### A. Effect Sequence Accuracy
Measures whether the effects are applied in the correct order.

5 — All four effects are applied in the correct order without any omissions or rearrangements.
4 — All effects are applied in the correct order, but minor deviations are present (e.g., slight timing misalignment).
3 — Most effects are applied in the correct order, but one effect is missing or out of sequence.
2 — Several effects are missing or applied out of sequence.
1 — The sequence is completely incorrect or missing entirely.

#### B. Timing Precision
Evaluates whether each effect lasts exactly 1 second as specified.

5 — All effects last exactly 1 second with no deviations.
4 — All effects are close to 1 second, with minor deviations of less than 0.2 seconds.
3 — Most effects are close to 1 second, but one or more have noticeable timing deviations.
2 — Several effects have significant timing deviations exceeding 0.5 seconds.
1 — Timing is completely inconsistent or ignored.

#### C. Heartbeat Effect Quality
Assesses the fidelity of the heartbeat effect implementation.

5 — The heartbeat effect matches the earlier implementation perfectly in appearance and timing.
4 — The heartbeat effect is mostly accurate, with minor deviations in appearance or timing.
3 — The heartbeat effect is somewhat accurate but has noticeable deviations in appearance or timing.
2 — The heartbeat effect is poorly implemented and does not resemble the earlier description.
1 — The heartbeat effect is missing or completely incorrect.

#### D. Transition Smoothness
Evaluates the smoothness and coherence of transitions between effects.

5 — Transitions between all effects are smooth and visually coherent.
4 — Transitions are mostly smooth, with minor visual inconsistencies.
3 — Transitions are somewhat smooth, but noticeable glitches or abrupt changes are present.
2 — Transitions are poorly executed, with significant visual disruptions.
1 — Transitions are completely incoherent or missing.

#### E. Loop Quality
Assesses whether the GIF loops seamlessly without visible glitches.

5 — The GIF loops seamlessly with no visible glitches.
4 — The GIF loops mostly seamlessly, with minor visual inconsistencies at the loop point.
3 — The GIF loops somewhat seamlessly, but noticeable glitches are present at the loop point.
2 — The GIF does not loop seamlessly, with significant visual disruptions at the loop point.
1 — The GIF does not loop at all or has major glitches throughout.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "effect_sequence_accuracy": <1-5>,
  "timing_precision": <1-5>,
  "heartbeat_effect_quality": <1-5>,
  "transition_smoothness": <1-5>,
  "loop_quality": <1-5>,
  "dimension_reasoning": {{
    "effect_sequence_accuracy": "<one sentence citing specific evidence>",
    "timing_precision": "<one sentence citing specific evidence>",
    "heartbeat_effect_quality": "<one sentence citing specific evidence>",
    "transition_smoothness": "<one sentence citing specific evidence>",
    "loop_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "effect_sequence_accuracy": 0.3,
    "timing_precision":    0.25,
    "heartbeat_effect_quality": 0.2,
    "transition_smoothness": 0.15,
    "loop_quality":        0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse_answer_tag(text):
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
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
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }