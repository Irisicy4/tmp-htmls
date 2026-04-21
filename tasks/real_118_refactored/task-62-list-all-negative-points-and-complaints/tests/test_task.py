import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""List all negative points and complaints about Claro Brasil, CNPJ 40432544000147.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target: Claro Brasil, CNPJ 40432544000147
- Scope: all publicly available negative points — Google reviews, Reclame Aqui, social media, news
- Must cover multiple sources, not just one platform
- Output: structured list of complaints by category or source

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for the company by name and CNPJ?
- What platforms were searched (Google, Reclame Aqui, Procon, social media)?
- What specific complaints were found?
- Are complaints organized by category or source?

### Step 2: Dimension Scoring

#### A. Search Breadth (0.3)
Did the agent search multiple platforms?

5 — Searched 4+ sources: Google reviews, Reclame Aqui, Procon, social media, forums.
4 — Searched 2-3 sources.
3 — Searched only one source (e.g. Google reviews only).
2 — Searched web generally without targeting complaint platforms.
1 — No research performed.

#### B. Complaint Specificity (0.35)
Are specific complaints reported with detail?

5 — Specific complaints with examples: dates, complaint text snippets, recurring themes.
4 — Specific complaints listed but without much detail.
3 — Categories of complaints listed (e.g. 'delivery issues') without specifics.
2 — Vague characterization ('bad reviews exist').
1 — No complaints found or reported.

#### C. Source Attribution (0.2)
Are sources cited for each complaint category?

5 — Each complaint or theme attributed to a specific source with context.
4 — Most complaints attributed to sources.
3 — Some attribution but inconsistent.
2 — No source attribution.
1 — No sourcing.

#### D. Output Organization (0.15)
Is the output well-organized and easy to use?

5 — Complaints organized by category or source with clear headings.
4 — Organized but one level of structure.
3 — List format without categorization.
2 — Wall of text.
1 — No organization.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "search_breadth": <1-5>,
  "complaint_specificity": <1-5>,
  "source_attribution": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "search_breadth": "<one sentence citing specific evidence>",
    "complaint_specificity": "<one sentence citing specific evidence>",
    "source_attribution": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "search_breadth": 0.3,
    "complaint_specificity": 0.35,
    "source_attribution": 0.2,
    "output_organization": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
