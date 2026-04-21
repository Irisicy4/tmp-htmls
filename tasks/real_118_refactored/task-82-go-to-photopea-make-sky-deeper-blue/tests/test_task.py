import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Change this image to have a deeper blue sky background. Use Photopea at https://www.photopea.com to open an image from Wikipedia (e.g. a landscape photo) and deepen the sky to a richer blue.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Tool: Photopea at photopea.com
- Action: deepen/intensify the blue sky — not replace it entirely
- Method: selection tool + hue/saturation or color balance adjustment
- Output: edited image saved as file

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent open Photopea?
- Was an image loaded?
- What method was used to deepen the sky blue (selection, adjustment layers)?
- Was the edit executed and result saved?
- Does the output have a visibly deeper blue sky?

### Step 2: Dimension Scoring

#### A. Tool Usage (0.25)
Did the agent use Photopea correctly?

5 — Agent opened Photopea, loaded an image, and used editing tools.
4 — Photopea opened but with some setup difficulty.
3 — Agent described Photopea workflow without evidence of execution.
2 — Agent used a different image editor.
1 — No image editor used.

#### B. Sky Selection (0.25)
Was the sky area properly selected for editing?

5 — Sky selected using Magic Wand, Select by Color, or Quick Select; selection refined.
4 — Sky approximately selected but imprecisely.
3 — Whole image edited instead of just sky.
2 — No selection — edits applied globally.
1 — No selection attempted.

#### C. Color Adjustment (0.35)
Was the sky color deepened effectively?

5 — Sky is visibly deeper blue using Hue/Saturation, Curves, or Color Balance adjustment.
4 — Sky deepened but effect is subtle or inconsistent.
3 — Color adjustment applied but to wrong area or with wrong method.
2 — Adjustment attempted but no visible change.
1 — No color adjustment.

#### D. File Output (0.15)
Was the edited image saved?

5 — Edited image exported and saved as PNG/JPG file.
4 — Save initiated but format or location unclear.
3 — Image edited but not saved.
2 — Agent described saving without evidence.
1 — No output file.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tool_usage": <1-5>,
  "sky_selection": <1-5>,
  "color_adjustment": <1-5>,
  "file_output": <1-5>,
  "dimension_reasoning": {{
    "tool_usage": "<one sentence citing specific evidence>",
    "sky_selection": "<one sentence citing specific evidence>",
    "color_adjustment": "<one sentence citing specific evidence>",
    "file_output": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "tool_usage": 0.25,
    "sky_selection": 0.25,
    "color_adjustment": 0.35,
    "file_output": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
