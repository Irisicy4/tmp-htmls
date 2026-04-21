import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""I am now in central London. Please tell me 10 places with a good view of Big Ben, their addresses, and for each place provide directions from the city center, nearby attractions, and nearby restaurants.""")

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Exactly 10 locations required (fewer is a failure; more is acceptable)
- Each location must include all four elements: address, directions from city centre, nearby attractions, nearby restaurants
- Locations must genuinely offer a view of Big Ben (e.g. Westminster Bridge, Jubilee Gardens, Southbank, St James's Park, Parliament Square, Lambeth Bridge, Victoria Tower Gardens, Hungerford Bridge)
- Context: user is currently in central London and needs actionable, specific guidance

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
Carefully read the agent response and trace, then answer each question:

- How many distinct locations were listed?
- Did each location include a street address?
- Were directions from central London provided for each location?
- Were nearby attractions listed for each location?
- Were nearby restaurants listed for each location?
- Are the locations plausible Big Ben viewpoints (not just anywhere near Westminster)?

### Step 2: Dimension Scoring
Score each dimension 1–5 using the rubrics below.

#### A. Task Completion (0.36)
Did the agent find and present 10 locations with Big Ben views, each with all four required details?

5 — 10 or more locations listed, all plausible Big Ben viewpoints, all four sub-details present for each.
4 — 8–9 locations, or 10 locations with one sub-detail missing for a few.
3 — 5–7 locations, or 10 locations with multiple sub-details consistently missing.
2 — Fewer than 5 locations, or most sub-details absent.
1 — Fewer than 3 locations or task not attempted.

#### B. Information Quality (0.27)
Is the location and logistical information accurate?

5 — Correct London addresses, plausible directions, real nearby attractions and restaurants, genuine viewpoints.
4 — Mostly accurate; minor errors in one or two locations.
3 — Some accurate content mixed with vague or unverifiable details.
2 — Information appears hallucinated or geographically inaccurate.
1 — No real London information provided.

#### C. Response Quality (0.18)
Is the response well-structured and easy for a visitor to use?

5 — Clearly structured per location with all four elements visually distinct; immediately actionable.
4 — Good structure with minor formatting inconsistencies.
3 — Information present but organisation makes it hard to use.
2 — Disorganised or requires significant effort to extract details.
1 — Not usable.

#### D. Completeness (0.18)
Were all four required elements provided for all 10 locations?

5 — All four elements (address, directions, attractions, restaurants) present for all 10 locations.
4 — All four elements present for most locations; 1–2 locations missing one element.
3 — One element consistently missing across multiple locations.
2 — Two or more elements missing across most locations.
1 — Fewer than 2 of the 4 required elements present throughout.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising your Step 1 findings>",
  "task_completion": <1-5>,
  "information_quality": <1-5>,
  "response_quality": <1-5>,
  "completeness": <1-5>,
  "dimension_reasoning": {{
    "task_completion": "<one sentence citing specific evidence>",
    "information_quality": "<one sentence citing specific evidence>",
    "response_quality": "<one sentence citing specific evidence>",
    "completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "task_completion":    0.36,
    "information_quality": 0.27,
    "response_quality":   0.18,
    "completeness":       0.18,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
