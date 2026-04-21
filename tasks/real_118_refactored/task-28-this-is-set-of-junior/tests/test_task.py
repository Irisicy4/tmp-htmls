import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""This is a set of junior high school English review materials. I want to create an interactive learning webpage based on these materials so students can quickly and enjoyably review and master each knowledge point. Please help me design and deploy this webpage. Treat each page or every 2 pages of the PDF as one topic and organize by topic. For each topic, add interactive exercises with instant feedback and a wrong-answer book with statistics. Also generate a knowledge graph for each topic and integrate text-to-speech. Deploy the code to GitHub and host the webpage using GitHub Pages.""")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Source material: a PDF of junior high school English review materials
- Organisation: each page or every 2 pages = one topic; content must be topic-organised
- Required features per topic: (1) interactive exercises with instant feedback, (2) wrong-answer book with statistics, (3) knowledge graph, (4) text-to-speech
- Deployment: code pushed to GitHub AND webpage hosted live on GitHub Pages (a working URL required)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question:

- Did the agent read or parse the PDF source material?
- Was the content organised by topic (per page or per 2 pages)?
- Were interactive exercises with instant feedback implemented?
- Was a wrong-answer book with statistics included?
- Was a knowledge graph generated per topic?
- Was text-to-speech integrated?
- Was the code pushed to a GitHub repository?
- Was the webpage deployed to GitHub Pages with a live URL?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. Task Completion (0.40)
Did the agent build and deploy the interactive learning webpage with all required features?

5 — Webpage built with all four required features, deployed to GitHub Pages, working URL provided.
4 — Webpage built and deployed but one required feature missing.
3 — Webpage built with 2–3 features missing, or deployment incomplete.
2 — Partial webpage built but not deployed, or major features absent.
1 — No webpage built or task abandoned.

#### B. Information Quality (0.30)
Is the webpage content accurately derived from the PDF source material and correctly structured?

5 — Content clearly derived from the PDF, correctly organised by topic, faithful to source material.
4 — Mostly accurate; minor misalignment with source material.
3 — Partially derived from source; some topics missing or content generalised.
2 — Content appears fabricated or unrelated to source material.
1 — No source material used.

#### C. Document Saved (0.20)
Was the code committed to GitHub and the page live on GitHub Pages?

5 — Code committed to a GitHub repository AND a live GitHub Pages URL provided and accessible.
4 — Code on GitHub but Pages deployment incomplete or URL not confirmed.
3 — Code saved in a non-GitHub location (e.g. local file only).
2 — Code produced but not saved anywhere accessible.
1 — No output saved.

#### D. Completeness (0.10)
Were all four required interactive features implemented?

5 — All four features present: interactive exercises with feedback, wrong-answer book with statistics, knowledge graph, text-to-speech.
4 — Three of four features present.
3 — Two of four features present.
2 — One feature present.
1 — No required features implemented.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "task_completion": <1-5>,
  "information_quality": <1-5>,
  "document_saved": <1-5>,
  "completeness": <1-5>,
  "dimension_reasoning": {{
    "task_completion": "<one sentence citing specific evidence>",
    "information_quality": "<one sentence citing specific evidence>",
    "document_saved": "<one sentence citing specific evidence>",
    "completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "task_completion":    0.40,
    "information_quality": 0.30,
    "document_saved":     0.20,
    "completeness":       0.10,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
