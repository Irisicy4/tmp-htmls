import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to Progressive, Geico, and State Farm and use each insurer's online quote tool to get "
    "an estimate for a 30-year-old male driver in Chicago, IL with a clean driving record, driving "
    "a 2022 Toyota Camry, seeking 100/300/100 liability coverage plus comprehensive and collision "
    "with a $500 deductible. Record the quoted annual premium from each insurer. Then visit the "
    "NAIC Consumer Insurance Search tool and look up each insurer's complaint index score for "
    "private passenger auto insurance in Illinois. Produce an auto insurance value analysis table "
    "with columns: Insurer, Annual Premium, Complaint Index, Coverage Details, and a 'value score'.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Insurers: Progressive, Geico, AND State Farm — all three required
- Quote parameters: 30-year-old male, Chicago IL, clean record, 2022 Toyota Camry, 100/300/100 liability, comp+collision, $500 deductible
- NAIC: complaint index scores for private passenger auto in Illinois (eapps.naic.org/cis/)
- Output columns: Insurer, Annual Premium, Complaint Index, Coverage Details, Value Score
- Value score: some explicit method comparing premium vs. complaint index (lower combined is better)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent attempt to get quotes from all three insurers (Progressive, Geico, State Farm)?
- Are annual premium amounts reported for each insurer?
- Did the agent access the NAIC consumer search tool for complaint index scores?
- Is there a comparison table with all five columns?

### Step 2: Dimension Scoring

#### A. Quote Retrieval Attempts
Did the agent navigate all three insurer quote tools and attempt to retrieve premiums?

5 — Agent navigated all three insurer sites and retrieved quoted annual premiums (or explained when quote tools redirected/required personal info that blocked completion).
4 — Agent successfully retrieved quotes from two of three insurers; third was attempted.
3 — Agent retrieved quotes from at least two insurers; one insurer was not attempted.
2 — Only one insurer's quote tool was used; others are listed with estimated or fabricated premiums.
1 — No insurer quote tool navigation; all premiums are from prior knowledge or fabricated.

#### B. Quote Parameter Accuracy
Are the correct coverage parameters applied (100/300/100, comp+collision, $500 deductible)?

5 — All parameters applied correctly for all quotes: correct coverage limits (100/300/100), comprehensive and collision, $500 deductible.
4 — Parameters correct for most quotes; one parameter slightly off (e.g., different deductible).
3 — Coverage type correct (full coverage) but specific limits or deductible not confirmed or different.
2 — Generic quote obtained without confirming coverage parameters.
1 — Coverage parameters not applied; quotes are generic or not tied to specified coverage.

#### C. NAIC Complaint Index Lookup
Did the agent access the NAIC consumer search and retrieve complaint index scores?

5 — Navigated NAIC eapps.naic.org/cis/, found complaint index scores for all three insurers for private passenger auto in Illinois, and reported specific numbers.
4 — Found complaint index scores for two of three insurers from NAIC.
3 — NAIC referenced and complaint index mentioned; scores found for at least one insurer or scores appear plausible but not clearly from NAIC.
2 — NAIC mentioned but no specific complaint index scores retrieved; values estimated.
1 — NAIC not used; complaint index absent or fabricated.

#### D. Output Table & Value Score
Is the comparison table complete and is the value score methodology clear?

5 — Complete five-column table (Insurer, Premium, Complaint Index, Coverage Details, Value Score) with an explained value score method (e.g., normalized score or premium/complaint ratio).
4 — Table present with four columns; value score present but methodology not explained.
3 — Table present with three columns; value score generic or missing.
2 — Narrative response with comparison but no structured table.
1 — No table and no value score.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "quote_retrieval": <1-5>,
  "quote_parameters": <1-5>,
  "naic_lookup": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "quote_retrieval": "<one sentence citing specific evidence>",
    "quote_parameters": "<one sentence citing specific evidence>",
    "naic_lookup": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "quote_retrieval":   0.30,
    "quote_parameters":  0.20,
    "naic_lookup":       0.30,
    "output_table":      0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
