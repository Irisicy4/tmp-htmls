import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to InsureMyTrip and search for single-trip travel insurance for a 35-year-old traveler "
    "taking a 14-day trip to Japan, with a total trip cost of $4,000 and departure in 30 days. "
    "Compare the top three quoted plans and for each record: insurer name, plan name, premium, "
    "trip cancellation limit, medical evacuation limit, pre-existing condition waiver (yes/no), "
    "and COVID-19 coverage (yes/no). Then visit AM Best and look up the financial strength rating "
    "for each insurer. Compile a travel insurance comparison table with all specified columns "
    "and add a final recommendation row identifying the best value plan with justification.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: InsureMyTrip for quotes AND AM Best for insurer ratings (both required)
- Quote parameters: 35-year-old, 14-day Japan trip, $4,000 trip cost, departure in 30 days
- Number of plans: Three plans compared
- Required columns: Insurer, Plan Name, Premium, Trip Cancellation, Medical Evacuation, Pre-Existing Waiver, COVID Coverage, AM Best Rating
- Recommendation: Must identify best-value plan with one-sentence justification
- Output: Complete structured table plus recommendation row

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate InsureMyTrip and enter the correct trip parameters?
- Are three specific plan quotes retrieved with insurer names and premiums?
- Did the agent look up AM Best ratings for each insurer?
- Is there a complete comparison table with all eight columns?
- Is a recommendation present with justification?

### Step 2: Dimension Scoring

#### A. InsureMyTrip Quote Retrieval
Did the agent navigate InsureMyTrip with correct parameters and retrieve actual quotes?

5 — Navigated InsureMyTrip, entered all correct parameters (age 35, 14-day Japan trip, $4,000, 30-day departure), and retrieved three specific plan quotes with named insurers and premiums.
4 — Navigated InsureMyTrip and retrieved three quotes but one parameter was slightly off (e.g., approximate departure date).
3 — Navigated InsureMyTrip but retrieved fewer than three quotes, or parameters are clearly wrong.
2 — InsureMyTrip mentioned but quotes are generic or appear to be from prior knowledge rather than real navigation.
1 — No InsureMyTrip navigation; plans are fabricated or entirely from prior knowledge.

#### B. Plan Data Completeness
Are all required coverage fields present for all three plans?

5 — All three plans have premium, trip cancellation limit, medical evacuation limit, pre-existing waiver (yes/no), and COVID coverage (yes/no).
4 — Two of three plans have all fields; one is missing one field (typically COVID coverage or pre-existing waiver).
3 — All three plans have premium and trip cancellation; evacuation limit or boolean fields are missing for some.
2 — Fewer than three plans have most fields; critical fields like evacuation limit are missing.
1 — No specific plan data; generic coverage descriptions only.

#### C. AM Best Rating Lookup
Did the agent look up and report AM Best financial strength ratings for each insurer?

5 — AM Best rating (e.g., A+, A, A-) retrieved and reported for all three insurers.
4 — AM Best rating found for two of three insurers; one rating missing or unable to find.
3 — AM Best rating mentioned for insurers but ratings appear estimated or not from the AM Best site.
2 — AM Best mentioned as a source but no specific ratings retrieved.
1 — AM Best not used; insurer financial ratings absent.

#### D. Output Structure & Recommendation
Is the comparison table complete and is the recommendation specific and justified?

5 — Complete table with all eight columns for all three plans, plus a recommendation row naming the winning plan and a specific justification (e.g., "lowest premium with A+ rated insurer and full pre-existing waiver").
4 — Table has seven of eight columns; recommendation present but justification is generic.
3 — Table present with six or fewer columns; or recommendation is missing.
2 — Narrative response with plan info but no structured table.
1 — No table and no recommendation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "quote_retrieval": <1-5>,
  "plan_data_completeness": <1-5>,
  "am_best_lookup": <1-5>,
  "output_and_recommendation": <1-5>,
  "dimension_reasoning": {{
    "quote_retrieval": "<one sentence citing specific evidence>",
    "plan_data_completeness": "<one sentence citing specific evidence>",
    "am_best_lookup": "<one sentence citing specific evidence>",
    "output_and_recommendation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "quote_retrieval":           0.30,
    "plan_data_completeness":    0.25,
    "am_best_lookup":            0.25,
    "output_and_recommendation": 0.20,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
