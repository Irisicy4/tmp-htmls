import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""I want to drive from Sydney to Melbourne tomorrow. Please plan a route for me with light traffic. I need to drive at night at around 100 km/h and arrive within one day.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Origin: Sydney, NSW
- Destination: Melbourne, VIC
- Speed: ~100 km/h (highway driving)
- Constraint: arrive within one day
- Preference: light traffic route
- Must include: estimated distance, duration, major roads/highways, rest stop recommendations

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent plan a specific route between the two cities?
- What highways and roads were identified?
- What is the estimated total distance and driving time?
- Does the route feasibility check out (can it be done in one day at 100 km/h)?
- Are rest stops or timing recommendations included?

### Step 2: Dimension Scoring

#### A. Route Planning (0.35)
Did the agent plan a specific, complete route?

5 — Specific route with named highways (Hume Highway, M31, etc.), major towns passed, and key junctions.
4 — Route planned but some segments vague.
3 — General direction planned without specific highways.
2 — Only origin/destination acknowledged without route detail.
1 — No route planned.

#### B. Feasibility Analysis (0.3)
Is the route feasible within the constraints?

5 — Distance and time calculated (approximately 870 km, ~9-10 hours), confirmed doable in one day.
4 — Feasibility confirmed but calculation less precise.
3 — Feasibility stated without calculation.
2 — Route planned but feasibility not addressed.
1 — Route infeasible or not analyzed.

#### C. Traffic Optimization (0.2)
Is the route optimized for light traffic?

5 — Alternative routes compared, nighttime driving advantages noted, peak congestion areas identified.
4 — Light traffic preference addressed but only partially.
3 — Traffic mentioned but no specific optimization.
2 — Generic route without traffic consideration.
1 — Traffic not addressed.

#### D. Practical Details (0.15)
Are practical driving details included?

5 — Rest stops, fuel stations, toll estimates, and timing recommendations included.
4 — Some practical details (e.g. rest stop timing) included.
3 — Distance and time only, no practical tips.
2 — Very limited practical info.
1 — No practical details.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "route_planning": "<one sentence citing specific evidence>",
    "feasibility_analysis": "<one sentence citing specific evidence>",
    "traffic_optimization": "<one sentence citing specific evidence>",
    "practical_details": "<one sentence citing specific evidence>",
  }},
  "route_planning": <1-5>,
  "feasibility_analysis": <1-5>,
  "traffic_optimization": <1-5>,
  "practical_details": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "route_planning": 0.35,
    "feasibility_analysis": 0.3,
    "traffic_optimization": 0.2,
    "practical_details": 0.15,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
