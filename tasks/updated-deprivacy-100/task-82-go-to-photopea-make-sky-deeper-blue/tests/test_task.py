"""LLM-as-judge evaluator for task-82-go-to-photopea-make-sky-deeper-blue.

Category: Image Editing
Task: Go to https://www.photopea.com. Open this image URL: https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/1280px-24701-nature-natural-beauty.jpg. Use Photopea's se
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "Go to https://www.photopea.com. Open this image URL: https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/1280px-24701-nature-natural-beauty.jpg. Use Photopea's selection and Hue/Saturation adjustment tools to make the sky in the image a deeper, more vivid blue. Export the result as a PNG file."

SYSTEM_PROMPT_EXTRA = "Your job is to assess whether an AI agent successfully enhanced the sky's color in an image using Photopea's selection and Hue/Saturation adjustment tools. The task involves accessing the specified image URL, making precise edits to achieve a deeper, more vivid blue sky, and exporting the final result as a PNG file. Accuracy and adherence to the instructions are key evaluation criteria."

CONSTRAINTS = """- Access the image from the provided URL.
- Use Photopea's selection tool to isolate the sky area.
- Adjust the Hue/Saturation settings to make the sky a deeper, more vivid blue.
- Export the edited image as a PNG file."""

EVIDENCE_QUESTIONS = """- Did the agent successfully open the image from the provided URL in Photopea?
- Did the agent correctly isolate the sky using the selection tool?
- Did the agent adjust the Hue/Saturation settings to achieve a deeper, more vivid blue for the sky?
- Was the edited image exported as a PNG file?
- Does the final image meet the visual expectations described in the task?"""

DIMENSION_RUBRICS = """#### A. Image Access And Import
Evaluates whether the agent successfully accessed and imported the image from the provided URL into Photopea.

5 — The image was successfully accessed and imported without errors.
4 — The image was accessed and imported, but with minor delays or issues.
3 — The image was accessed and imported, but with noticeable errors or incomplete steps.
2 — The image was accessed but not properly imported into Photopea.
1 — The image was not accessed or imported at all.

#### B. Sky Selection Accuracy
Evaluates the accuracy of the agent's selection of the sky area in the image.

5 — The sky was precisely and completely selected without affecting other parts of the image.
4 — The sky was mostly selected with minor inaccuracies or spillover.
3 — The sky selection was partially accurate but included noticeable errors.
2 — The sky selection was poorly executed, with significant inaccuracies.
1 — The sky was not selected or the selection was completely incorrect.

#### C. Color Adjustment Quality
Evaluates the quality of the color adjustment applied to make the sky a deeper, more vivid blue.

5 — The sky's color was adjusted to a vivid and natural-looking blue, meeting the task's expectations.
4 — The sky's color was adjusted well, but with minor deviations from the expected vividness or natural appearance.
3 — The sky's color adjustment was noticeable but lacked vividness or appeared unnatural.
2 — The sky's color adjustment was poorly executed, with significant issues in vividness or natural appearance.
1 — The sky's color was not adjusted or the adjustment was completely incorrect.

#### D. Export Format And Quality
Evaluates whether the edited image was exported as a PNG file and retained high quality.

5 — The image was exported as a PNG file with high quality and no errors.
4 — The image was exported as a PNG file with minor quality issues.
3 — The image was exported as a PNG file but with noticeable quality issues.
2 — The image was exported in the wrong format or with significant quality issues.
1 — The image was not exported or the export was completely incorrect."""

DIMENSION_WEIGHTS = {
    'image_access_and_import': 0.2,
    'sky_selection_accuracy': 0.3,
    'color_adjustment_quality': 0.3,
    'export_format_and_quality': 0.2,
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
