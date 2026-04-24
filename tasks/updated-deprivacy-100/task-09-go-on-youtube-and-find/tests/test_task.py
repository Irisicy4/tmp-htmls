"""LLM-as-judge evaluator for task-09-go-on-youtube-and-find.

Category: Web Research
Task: go on youtube and find videos of silent hill 3 that explains the story
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'go on youtube and find videos of silent hill 3 that explains the story'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully located and identified YouTube videos that explain the story of Silent Hill 3.'

CONSTRAINTS = """- The videos must be from YouTube.
- The videos must explicitly explain the story of Silent Hill 3.
- The agent must provide accurate and accessible links to the videos."""

EVIDENCE_QUESTIONS = """- Did the agent provide valid YouTube links?
- Do the videos explicitly explain the story of Silent Hill 3?
- Are the videos relevant and free from unrelated content?"""

DIMENSION_RUBRICS = """#### A. Relevance Of Videos
Measures how closely the videos align with the task of explaining the story of Silent Hill 3.

5 — All videos are highly relevant and focus entirely on explaining the story of Silent Hill 3.
4 — Most videos are relevant and primarily focus on explaining the story of Silent Hill 3.
3 — Some videos are relevant, but others include unrelated or tangential content.
2 — Few videos are relevant, with most being unrelated or off-topic.
1 — None of the videos are relevant to the story of Silent Hill 3.

#### B. Accuracy Of Links
Assesses whether the provided YouTube links are valid and lead to the intended videos.

5 — All links are valid and lead directly to the intended videos.
4 — Most links are valid, with only minor issues (e.g., one broken link).
3 — Some links are valid, but others are broken or incorrect.
2 — Few links are valid, with most being broken or leading to unrelated content.
1 — None of the links are valid or lead to the intended videos.

#### C. Depth Of Explanation
Evaluates how thoroughly the videos explain the story of Silent Hill 3.

5 — The videos provide a comprehensive and detailed explanation of the story.
4 — The videos provide a good explanation, with minor gaps in detail.
3 — The videos provide a basic explanation, but significant details are missing.
2 — The videos provide a minimal explanation, with most details missing.
1 — The videos do not explain the story at all.

#### D. Presentation Quality
Assesses the clarity and professionalism of the videos' presentation.

5 — The videos are clear, well-structured, and professionally presented.
4 — The videos are mostly clear and well-structured, with minor issues in presentation.
3 — The videos are somewhat clear but lack structure or professionalism.
2 — The videos are unclear or poorly presented, making them hard to follow.
1 — The videos are completely unclear and unprofessional in presentation.

#### E. Variety Of Sources
Evaluates whether the agent provided a diverse range of sources for the videos.

5 — The videos come from a wide variety of credible and diverse sources.
4 — The videos come from a good variety of sources, with minor redundancy.
3 — The videos come from a limited range of sources, with some redundancy.
2 — The videos come from very few sources, with significant redundancy.
1 — The videos all come from the same source or lack diversity entirely."""

DIMENSION_WEIGHTS = {
    'relevance_of_videos': 0.35,
    'accuracy_of_links': 0.25,
    'depth_of_explanation': 0.2,
    'presentation_quality': 0.1,
    'variety_of_sources': 0.1,
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
