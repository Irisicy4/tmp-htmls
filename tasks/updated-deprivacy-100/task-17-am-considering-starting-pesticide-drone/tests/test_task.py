"""LLM-as-judge evaluator for task-17-am-considering-starting-pesticide-drone.

Category: Business Feasibility Analysis
Task: I am considering starting a pesticide drone business.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'I am considering starting a pesticide drone business.\n  Please research and tell me in detail:\n  - Future positioning and how to prepare for starting the business\n  - How much money is needed\n  - Essential products to purchase (e.g. vehicles, generators, drones), and which drone brand is good (e.g. DJI)\n  - Whether it can be run as a standalone business\n  - How much it would cost to operate (e.g. price per unit area)\n  - Which month is the peak season, and how much can be earned in a year\n  - How long it takes to recoup the investment\n  I am 35 years old (born 1991); please consider what is feasible at my age.\n  I like using computers and have some photography skills – please see if this helps with the work.\n  \n  Compile everything and create a Google Docs document.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully researched and compiled a detailed feasibility analysis for starting a pesticide drone business, addressing all specified aspects.'

CONSTRAINTS = """- The analysis must address all points in the instruction, including costs, equipment, feasibility, and seasonal considerations.
- The document must be compiled into a Google Docs file and shared as a link.
- The analysis must consider the user's age, skills, and interests as stated in the instruction."""

EVIDENCE_QUESTIONS = """- Does the analysis include detailed information on future positioning and preparation for starting the business?
- Are the costs (startup and operational) clearly broken down and supported by evidence or references?
- Is there a comprehensive list of essential products, including specific drone brands and their suitability?
- Does the analysis evaluate whether the business can be run as a standalone operation?
- Are seasonal trends and potential earnings (including peak months) addressed with realistic estimates?
- Is there a clear calculation of the time required to recoup the investment?
- Does the analysis consider the user's age, computer skills, and photography experience in the context of the business?"""

DIMENSION_RUBRICS = """#### A. Completeness
The extent to which all aspects of the instruction are addressed in detail.

5 — All points in the instruction are addressed comprehensively with detailed and relevant information.
4 — Most points are addressed in detail, but one or two aspects are less thorough.
3 — Some points are addressed, but several key aspects are missing or lack detail.
2 — Few points are addressed, and the analysis is incomplete or superficial.
1 — The analysis fails to address most points in the instruction.

#### B. Accuracy And Relevance
The accuracy and relevance of the information provided in the analysis.

5 — All information is accurate, up-to-date, and directly relevant to the task.
4 — Most information is accurate and relevant, with minor inaccuracies or tangential details.
3 — Some information is accurate, but there are notable inaccuracies or irrelevant details.
2 — Much of the information is inaccurate or irrelevant to the task.
1 — The analysis is largely inaccurate or irrelevant.

#### C. Usability
The extent to which the analysis is actionable and useful for the user.

5 — The analysis is highly actionable, with clear steps and practical recommendations tailored to the user's context.
4 — The analysis is mostly actionable, with some practical recommendations but minor gaps in clarity or tailoring.
3 — The analysis is somewhat actionable, but lacks clear steps or sufficient tailoring to the user's context.
2 — The analysis is minimally actionable, with vague or impractical recommendations.
1 — The analysis is not actionable or useful for the user.

#### D. Presentation And Formatting
The quality of the document's presentation and formatting.

5 — The document is well-organized, visually appealing, and easy to read, with proper formatting and structure.
4 — The document is mostly well-organized and readable, with minor formatting issues.
3 — The document is somewhat organized, but has noticeable formatting or readability issues.
2 — The document is poorly organized and difficult to read, with significant formatting issues.
1 — The document is disorganized and unreadable, with little to no attention to formatting.

#### E. Consideration Of User Context
The extent to which the analysis considers the user's age, skills, and interests.

5 — The analysis fully integrates the user's age, skills, and interests into the recommendations and feasibility assessment.
4 — The analysis mostly considers the user's context, with minor gaps in integration.
3 — The analysis somewhat considers the user's context, but lacks depth or thoroughness.
2 — The analysis minimally considers the user's context, with little integration into the recommendations.
1 — The analysis does not consider the user's age, skills, or interests."""

DIMENSION_WEIGHTS = {
    'completeness': 0.3,
    'accuracy_and_relevance': 0.25,
    'usability': 0.2,
    'presentation_and_formatting': 0.15,
    'consideration_of_user_context': 0.1,
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
