"""LLM-as-judge evaluator for task-11-apply-the-following-effects-in.

Category: GIF Creation
Task: Apply the following effects in order to the original image to create a GIF.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Apply the following effects in order to the original image to create a GIF.\n  1. The image fades in and is revealed slowly.\n  2. The image rotates one rotation clockwise.\n  3. The heartbeat effect implemented earlier appears once.\n  4. The image fades out and slowly disappears.\n  \n  Let each movement last for 1 second.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully applied specified visual effects to an image in sequence to create a GIF as instructed. The evaluation emphasizes the accuracy of the effects, their timing (1 second per movement), and the smoothness of the transitions.'

CONSTRAINTS = """- The GIF must include all four specified effects in the correct order: fade-in, clockwise rotation, heartbeat effect, fade-out.
- Each effect must last exactly 1 second.
- The heartbeat effect must match the implementation described in earlier instructions.
- The transitions between effects must be smooth and visually coherent.
- The final GIF must loop seamlessly."""

EVIDENCE_QUESTIONS = """- Does the GIF include all four effects in the correct order?
- Does each effect last exactly 1 second?
- Is the heartbeat effect implemented as described in earlier instructions?
- Are the transitions between effects smooth and visually coherent?
- Does the GIF loop seamlessly without visible glitches?"""

DIMENSION_RUBRICS = """#### A. Effect Sequence Accuracy
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
1 — The GIF does not loop at all or has major glitches throughout."""

DIMENSION_WEIGHTS = {
    'effect_sequence_accuracy': 0.3,
    'timing_precision': 0.25,
    'heartbeat_effect_quality': 0.2,
    'transition_smoothness': 0.15,
    'loop_quality': 0.1,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
