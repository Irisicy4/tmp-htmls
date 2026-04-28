import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Apply the following effects in order to the original image to create a GIF.\n"
    "  1. The image fades in and is revealed slowly.\n"
    "  2. The image rotates one rotation clockwise.\n"
    "  3. The heartbeat effect implemented earlier appears once.\n"
    "  4. The image fades out and slowly disappears.\n"
    "\n"
    "  Let each movement last for 1 second.")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Output format: animated GIF file
- Effects must appear in exact order: (1) fade in, (2) clockwise rotation, (3) heartbeat, (4) fade out
- Each effect must last approximately 1 second
- The heartbeat effect referenced must be the one implemented in the prior context of this task

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question with direct observations:

- Did the agent produce a GIF file? Where is it saved?
- Which of the 4 effects are present: fade in, clockwise rotation, heartbeat, fade out?
- Is the order of effects correct (1 → 2 → 3 → 4)?
- Is there any indication of timing (approximately 1 second per effect)?
- What tool or method did the agent use to create the GIF?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. GIF Output (0.30)
Was a GIF file produced?

5 — A GIF file was explicitly created and saved to disk; file path or confirmation provided.
4 — GIF creation was completed but file path is unclear or confirmation is indirect.
3 — Agent attempted to create a GIF but success is ambiguous (e.g. process ran without error but no file confirmed).
2 — Agent produced image frames or code for GIF creation but did not assemble into a final GIF.
1 — No GIF produced; task abandoned or error not resolved.

#### B. Effect Completeness (0.35)
Were all 4 required effects implemented?

5 — All 4 effects present: fade in, clockwise rotation, heartbeat, fade out.
4 — 3 of 4 effects implemented; one is missing or described but not clearly implemented.
3 — 2 of 4 effects implemented.
2 — Only 1 effect implemented.
1 — No effects implemented.

#### C. Effect Sequence (0.20)
Were the effects applied in the correct order (fade in → rotate → heartbeat → fade out)?

5 — Effects appear in the exact specified order based on trace or agent description.
4 — Order is mostly correct; one effect is slightly out of place.
3 — Effects are present but order cannot be confirmed from the trace or response.
2 — Effects are in the wrong order.
1 — No sequence apparent.

#### D. Timing Compliance (0.15)
Did each effect last approximately 1 second?

5 — Agent explicitly set duration of approximately 1 second per effect; confirmed in trace or code.
4 — Timing set for most effects; one effect has unspecified or different duration.
3 — Agent mentioned timing but 1-second constraint was not consistently applied.
2 — Timing not addressed; no duration specified for effects.
1 — No timing consideration at all.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "dimension_reasoning": {{
    "gif_output": "<one sentence citing specific evidence>",
    "effect_completeness": "<one sentence citing specific evidence>",
    "effect_sequence": "<one sentence citing specific evidence>",
    "timing_compliance": "<one sentence citing specific evidence>"
  }},
  "gif_output": <1-5>,
  "effect_completeness": <1-5>,
  "effect_sequence": <1-5>,
  "timing_compliance": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "gif_output":          0.30,
    "effect_completeness": 0.35,
    "effect_sequence":     0.20,
    "timing_compliance":   0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
