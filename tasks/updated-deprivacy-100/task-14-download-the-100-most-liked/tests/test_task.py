"""LLM-as-judge evaluator for task-14-download-the-100-most-liked.

Category: Web Research
Task: Download the 100 most liked videos on tiktok about diving mask and diving snorkel
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Download the 100 most liked videos on tiktok about diving mask and diving snorkel'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully downloaded and organized the 100 most liked TikTok videos related to diving masks and diving snorkels. The evaluation emphasizes the relevance of the videos to the topic, the completeness of the dataset, and adherence to ethical and legal guidelines.'

CONSTRAINTS = """- Only videos explicitly related to diving masks and diving snorkels should be included.
- The dataset must consist of exactly 100 videos.
- The videos must be ranked by the number of likes, starting from the most liked.
- The agent must ensure compliance with TikTok's terms of service and copyright laws.
- The videos must be downloaded in a format that preserves their original quality."""

EVIDENCE_QUESTIONS = """- Did the agent collect exactly 100 videos?
- Are all the videos directly related to diving masks and diving snorkels?
- Are the videos ranked correctly by the number of likes?
- Did the agent provide evidence of compliance with TikTok's terms of service and copyright laws?
- Is the quality of the downloaded videos consistent with their original format?"""

DIMENSION_RUBRICS = """#### A. Relevance To Topic
Measures how well the videos align with the topic of diving masks and diving snorkels.

5 — All 100 videos are directly and clearly related to diving masks and diving snorkels.
4 — At least 90% of the videos are related to diving masks and diving snorkels, with minor deviations.
3 — At least 75% of the videos are related to diving masks and diving snorkels, but some are off-topic.
2 — Fewer than 75% of the videos are related to diving masks and diving snorkels.
1 — The majority of the videos are unrelated to diving masks and diving snorkels.

#### B. Ranking Accuracy
Assesses whether the videos are correctly ranked by the number of likes.

5 — All 100 videos are perfectly ranked by the number of likes in descending order.
4 — The ranking is mostly accurate, with minor errors affecting fewer than 5 videos.
3 — The ranking has noticeable errors affecting 5-15 videos.
2 — The ranking has significant errors affecting more than 15 videos.
1 — The ranking is largely incorrect or absent.

#### C. Dataset Completeness
Evaluates whether the dataset contains exactly 100 videos.

5 — The dataset contains exactly 100 videos with no omissions or duplicates.
4 — The dataset contains 95-99 videos, with no duplicates.
3 — The dataset contains 90-94 videos, or includes a few duplicates.
2 — The dataset contains 80-89 videos, or includes several duplicates.
1 — The dataset contains fewer than 80 videos, or is riddled with duplicates.

#### D. Legal And Ethical Compliance
Checks whether the agent adhered to TikTok's terms of service and copyright laws.

5 — The agent provided clear evidence of full compliance with TikTok's terms of service and copyright laws.
4 — The agent provided some evidence of compliance, with minor gaps in documentation.
3 — The agent provided limited evidence of compliance, with noticeable gaps in documentation.
2 — The agent provided little evidence of compliance, with major gaps in documentation.
1 — The agent provided no evidence of compliance or clearly violated terms of service or copyright laws.

#### E. Video Quality Preservation
Assesses whether the downloaded videos maintain their original quality.

5 — All videos are downloaded in their original quality with no noticeable degradation.
4 — Most videos are downloaded in their original quality, with minor degradation in a few.
3 — Some videos show noticeable degradation in quality.
2 — Many videos show significant degradation in quality.
1 — The majority of the videos are of poor quality compared to their originals."""

DIMENSION_WEIGHTS = {
    'relevance_to_topic': 0.25,
    'ranking_accuracy': 0.25,
    'dataset_completeness': 0.2,
    'legal_and_ethical_compliance': 0.15,
    'video_quality_preservation': 0.15,
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
