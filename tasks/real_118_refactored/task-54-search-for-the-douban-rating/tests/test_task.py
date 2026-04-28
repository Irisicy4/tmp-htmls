import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Search for the Douban rating of the book 《智人之上》.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Douban (douban.com) specifically
- Book: 《智人之上》— must find the correct book entry
- Data: must retrieve the actual rating score and number of ratings
- Accuracy: rating must be from Douban, not Goodreads or other platforms

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Douban?
- Did the agent find the correct book entry for 《智人之上》?
- What rating was reported (score out of 10, number of raters)?
- Is this clearly from Douban?

### Step 2: Dimension Scoring

#### A. Platform Navigation (0.35)
Did the agent navigate to Douban and find the book?

5 — Agent navigated to douban.com, searched for 《智人之上》, and reached the correct book page.
4 — Agent reached Douban but had difficulty finding the exact entry.
3 — Agent found rating data from Douban but via indirect method (e.g. Google snippet).
2 — Agent found rating from a non-Douban source.
1 — No navigation to Douban.

#### B. Rating Accuracy (0.35)
Is the reported rating accurate and from Douban?

5 — Rating score (e.g. 8.x/10) and number of raters clearly reported from Douban.
4 — Rating score reported but rater count missing.
3 — Rating approximately correct but imprecise.
2 — Rating reported but from wrong source or clearly incorrect.
1 — No rating provided.

#### C. Book Identification (0.22)
Did the agent identify the correct book?

5 — Confirmed correct book with author (Yuval Noah Harari) and edition details.
4 — Correct book identified with author but no edition detail.
3 — Book identified by title only without confirming author.
2 — Wrong edition or potential confusion with similar title.
1 — Wrong book identified.

#### D. Response Clarity (0.08)
Is the response clear and concise?

5 — Rating presented clearly with book title, score, and rater count.
4 — Clear but missing one detail.
3 — Information present but buried.
2 — Confusing or verbose.
1 — No useful output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "platform_navigation": "<one sentence citing specific evidence>",
    "rating_accuracy": "<one sentence citing specific evidence>",
    "book_identification": "<one sentence citing specific evidence>",
    "response_clarity": "<one sentence citing specific evidence>",
  }},
  "platform_navigation": <1-5>,
  "rating_accuracy": <1-5>,
  "book_identification": <1-5>,
  "response_clarity": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "platform_navigation": 0.35,
    "rating_accuracy": 0.35,
    "book_identification": 0.22,
    "response_clarity": 0.08,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
