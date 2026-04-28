import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to the USITC Harmonized Tariff Schedule online database and look up the HTS classification "
    "codes and current duty rates for three products: (1) lithium-ion batteries for electric vehicles, "
    "(2) solar photovoltaic panels, (3) cotton T-shirts. For each product, record the 10-digit HTS "
    "code, general duty rate, and any applicable Section 301 (China) tariff surcharge. Then go to "
    "the CBP CROSS Ruling database and search for binding customs rulings related to each product; "
    "find one relevant ruling per product and note the ruling number, date, and key classification "
    "decision. Compile a tariff exposure report table with columns: Product, HTS Code, Base Duty "
    "Rate, Section 301 Surcharge, Total Effective Rate, CBP Ruling Number, Ruling Date, Key Finding.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: USITC HTS (hts.usitc.gov) AND CBP CROSS (rulings.cbp.gov) — both required
- Products: lithium-ion EV batteries, solar PV panels, cotton T-shirts — all three
- HTS data: 10-digit code, general (MFN) duty rate, Section 301 surcharge (if applicable)
- CBP CROSS: one ruling per product (ruling number in format NYxxx or HQxxx or similar)
- Output: Eight-column table with all three products

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate the USITC HTS database and look up codes for all three products?
- Are 10-digit HTS codes provided and plausible for each product?
- Did the agent navigate the CBP CROSS database and find rulings?
- Are ruling numbers in a plausible CBP format (e.g., N123456, HQ123456)?
- Is there a complete eight-column table?

### Step 2: Dimension Scoring

#### A. USITC HTS Navigation & Code Accuracy
Did the agent navigate the USITC HTS database and retrieve plausible codes for all three products?

5 — Navigated hts.usitc.gov (or official HTS tool), retrieved 10-digit codes for all three products that are plausible (e.g., EV batteries ~8507.60.00, solar panels ~8541.40.00, T-shirts ~6109.10.00).
4 — Navigated USITC and retrieved codes for two of three products with 10-digit precision; one product has an 8-digit or less specific code.
3 — Navigated USITC but codes for one or two products appear wrong or are 6-digit only.
2 — HTS codes cited but not clearly from USITC navigation; codes appear to be from prior knowledge.
1 — No HTS database navigation; codes absent or clearly incorrect.

#### B. Tariff Rate Data
Are general duty rate and Section 301 surcharge correctly identified for all three products?

5 — General duty rate and Section 301 surcharge (or explicit confirmation that none applies) for all three products, with specific percentages.
4 — General duty rate and Section 301 info for two of three products; one is incomplete.
3 — General duty rates present for most products but Section 301 surcharges absent or vague.
2 — Duty rates cited for all products but appear to be rough estimates rather than database lookups.
1 — No specific tariff rates; only general descriptions.

#### C. CBP CROSS Ruling Retrieval
Did the agent navigate CBP CROSS and find plausible rulings for all three products?

5 — Navigated rulings.cbp.gov, found one ruling per product with a specific ruling number, date (YYYY-MM-DD format), and key classification decision for all three.
4 — Rulings found for two of three products with specific numbers and dates.
3 — Rulings found for at least two products; ruling numbers present but dates or key findings missing.
2 — CBP CROSS mentioned but rulings found for only one product or ruling numbers appear fabricated.
1 — CBP CROSS not used; ruling information absent or entirely fabricated.

#### D. Output Table Completeness
Is the tariff exposure report table complete with all eight columns for all three products?

5 — Complete eight-column table (Product, HTS Code, Base Duty, Section 301, Total Rate, Ruling Number, Ruling Date, Key Finding) for all three products.
4 — Table with seven of eight columns for all three products.
3 — Table with six or fewer columns, or one product missing from the table.
2 — Partial table (fewer than five columns) or one product missing.
1 — No table structure; narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "hts_navigation": "<one sentence citing specific evidence>",
    "tariff_rate_data": "<one sentence citing specific evidence>",
    "cross_ruling_retrieval": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "hts_navigation": <1-5>,
  "tariff_rate_data": <1-5>,
  "cross_ruling_retrieval": <1-5>,
  "output_table": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "hts_navigation": 0.35,
    "tariff_rate_data": 0.26,
    "cross_ruling_retrieval": 0.22,
    "output_table": 0.17,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
