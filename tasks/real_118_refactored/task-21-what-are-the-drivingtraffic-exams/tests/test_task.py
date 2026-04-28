import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("What are the driving/traffic exams like in China, the US, and France? What are the differences between them?")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- All three countries must be covered: China, US, and France
- Both description AND comparison are required — not just listing each country's system separately
- Research expected: agent should search for current, accurate exam structures rather than relying solely on prior knowledge
- Key aspects to cover: written/theory test, practical/driving test, required hours, pass rates, notable differences

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Are all three countries covered (China, US, France)?
- What specific exam components are described for each country?
- Does the agent explicitly compare and contrast the systems?
- Did the agent search for this information or rely on prior knowledge?
- Are any facts clearly wrong or outdated?

### Step 2: Dimension Scoring

#### A. Country Coverage
Are all three countries covered with sufficient detail?

5 — All three countries covered with: exam structure, components (theory/practical), and key requirements.
4 — All three covered but one country has noticeably less detail.
3 — Two countries covered well; one is only briefly mentioned.
2 — Only one or two countries covered.
1 — No country-specific detail provided.

#### B. Comparative Analysis
Does the agent explicitly compare the three systems rather than just listing them separately?

5 — Structured comparison of key differences (e.g. number of required hours, difficulty, pass rates, unique features); highlights what makes each system distinctive.
4 — Comparison present but focuses on 1–2 dimensions only (e.g. only compares written tests).
3 — Implicit comparison (e.g. "China requires X while the US does not") but no dedicated comparison section.
2 — Three systems described separately with no explicit comparison.
1 — No comparison; single-country description or generic overview.

#### C. Information Accuracy
Is the information accurate and reasonably current?

5 — Key facts are correct (e.g. China's 科目一–四 structure, US state-by-state system, French code de la route + conduite); no clearly wrong claims.
4 — Mostly accurate; 1–2 minor inaccuracies or outdated details.
3 — Generally correct direction but several vague or unverifiable claims.
2 — Significant inaccuracies or information appears fabricated.
1 — Clearly wrong or completely generic.

#### D. Response Organisation
Is the output well-structured and easy to read?

5 — Clear structure: per-country sections + dedicated comparison; headers or table used effectively.
4 — Well-organised but comparison is embedded in narrative rather than structured separately.
3 — Content is present but formatting makes it hard to extract comparative information quickly.
2 — Wall of text with no clear organisation.
1 — Unstructured or non-responsive.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "dimension_reasoning": {{
    "country_coverage": "<one sentence>",
    "comparative_analysis": "<one sentence>",
    "information_accuracy": "<one sentence>",
    "response_organisation": "<one sentence>"
  }},
  "country_coverage": <1-5>,
  "comparative_analysis": <1-5>,
  "information_accuracy": <1-5>,
  "response_organisation": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {"country_coverage": 0.25, "comparative_analysis": 0.35, "information_accuracy": 0.25, "response_organisation": 0.15}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
