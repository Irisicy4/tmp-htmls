import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Tell me everything that is publicly known about Anya Taylor-Joy.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target: Anya Taylor-Joy
- Scope: publicly available information only (social media, news, professional profiles)
- Must search multiple sources
- Output: structured summary of findings by category (professional, social, public records, etc.)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search for the individual on multiple platforms?
- What public information was found?
- Were sources cited?
- Is output organized by category of information?

### Step 2: Dimension Scoring

#### A. Search Breadth (0.3)
Did the agent search multiple platforms?

5 — Searched Google, LinkedIn, social media, news, and public records platforms.
4 — Searched 3-4 platforms.
3 — Searched 2 platforms.
2 — Only searched Google.
1 — No search performed.

#### B. Information Completeness (0.3)
Was relevant public information found and reported?

5 — Comprehensive findings covering professional background, social presence, and any public news/records.
4 — Good coverage of 2 categories.
3 — Basic info found (e.g. LinkedIn profile summary only).
2 — Very minimal findings.
1 — No information found.

#### C. Source Attribution (0.25)
Are sources cited for findings?

5 — Every piece of information attributed to a specific URL or platform.
4 — Most information sourced.
3 — Some sources cited.
2 — No source attribution.
1 — No sourcing.

#### D. Output Organization (0.15)
Is output organized by information category?

5 — Clear sections: Professional, Social Media, Public Records, News/Media, etc.
4 — Some organization but not fully categorical.
3 — Organized list without clear categories.
2 — Unstructured paragraph.
1 — No organization.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "search_breadth": <1-5>,
  "information_completeness": <1-5>,
  "source_attribution": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "search_breadth": "<one sentence citing specific evidence>",
    "information_completeness": "<one sentence citing specific evidence>",
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
    "information_completeness": 0.3,
    "source_attribution": 0.25,
    "output_organization": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
