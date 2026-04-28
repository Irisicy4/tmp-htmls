"""LLM-as-judge evaluator for task-28-this-is-set-of-junior.

Category: Educational Web Development
Task: This is a set of junior high school English review materials. I want to create an interactive learning webpage based on these materials so students can quickly and enjoyably review and master each kno
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'This is a set of junior high school English review materials. I want to create an interactive learning webpage based on these materials so students can quickly and enjoyably review and master each knowledge point. Please help me design and deploy this webpage. Treat each page or every 2 pages of the PDF as one topic and organize by topic. For each topic, add interactive exercises with instant feedback and a wrong-answer book with statistics. Also generate a knowledge graph for each topic and integrate text-to-speech. Deploy the code to GitHub and host the webpage using GitHub Pages.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully designed, developed, and deployed an interactive learning webpage based on junior high school English review materials.'

CONSTRAINTS = """- The webpage must organize content by topic, with each page or every 2 pages of the PDF treated as one topic.
- Interactive exercises must provide instant feedback and include a wrong-answer book with statistics.
- A knowledge graph must be generated for each topic.
- Text-to-speech functionality must be integrated for all content.
- The code must be deployed to GitHub and hosted using GitHub Pages."""

EVIDENCE_QUESTIONS = """- Does the webpage organize content by topic based on the PDF structure?
- Are interactive exercises present for each topic, and do they provide instant feedback?
- Is there a wrong-answer book with statistics for each topic?
- Does each topic include a functional knowledge graph?
- Is text-to-speech functionality integrated and operational?
- Has the code been deployed to GitHub and hosted successfully using GitHub Pages?"""

DIMENSION_RUBRICS = """#### A. Topic Organization
Evaluates whether the webpage organizes content by topic as specified.

5 — Content is perfectly organized by topic, with clear alignment to the PDF structure.
4 — Content is mostly organized by topic, with minor deviations from the PDF structure.
3 — Content organization is partially aligned with the PDF structure, with noticeable gaps.
2 — Content organization is poorly aligned with the PDF structure, making topics unclear.
1 — Content organization is completely misaligned with the PDF structure.

#### B. Interactive Exercises
Assesses the presence and functionality of interactive exercises with instant feedback.

5 — All topics include fully functional interactive exercises with instant feedback.
4 — Most topics include functional interactive exercises with instant feedback.
3 — Some topics include interactive exercises, but functionality is inconsistent.
2 — Few topics include interactive exercises, and functionality is poor.
1 — Interactive exercises are missing or completely non-functional.

#### C. Knowledge Graphs
Evaluates the presence and quality of knowledge graphs for each topic.

5 — Knowledge graphs are present for all topics and are highly informative.
4 — Knowledge graphs are present for most topics and are moderately informative.
3 — Knowledge graphs are present for some topics but lack detail or clarity.
2 — Knowledge graphs are present for few topics and are poorly constructed.
1 — Knowledge graphs are missing or completely unusable.

#### D. Text To Speech Integration
Assesses the integration and functionality of text-to-speech features.

5 — Text-to-speech is fully integrated and works flawlessly across all content.
4 — Text-to-speech is mostly integrated and works well with minor issues.
3 — Text-to-speech is partially integrated and works inconsistently.
2 — Text-to-speech is poorly integrated and rarely functional.
1 — Text-to-speech is missing or completely non-functional.

#### E. Deployment And Hosting
Evaluates whether the code was successfully deployed to GitHub and hosted using GitHub Pages.

5 — Code is fully deployed to GitHub and the webpage is hosted flawlessly on GitHub Pages.
4 — Code is deployed to GitHub and the webpage is hosted with minor issues.
3 — Code is partially deployed to GitHub and hosting is inconsistent.
2 — Code deployment and hosting are poorly executed, with major issues.
1 — Code is not deployed or the webpage is not hosted on GitHub Pages."""

DIMENSION_WEIGHTS = {
    'topic_organization': 0.25,
    'interactive_exercises': 0.25,
    'knowledge_graphs': 0.2,
    'text_to_speech_integration': 0.15,
    'deployment_and_hosting': 0.15,
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
