"""
LLM-as-judge evaluator for EvolveBench task-10.

Category: Design (custom rubric — open-ended creative task)
Task: "You are a graduate student in architecture. Now you want to build a lakeside
       park on the Wushan campus of South China University of Technology. It is located
       on the central axis of the campus. It covers an area of about 4,000 square meters
       and has a height difference of half a meter. Find relevant cases that can be learned
       and generate a design sketch. It is required to reflect the psychosis of the place
       combined with Lingnan characteristics."

Note: This is an open-ended design task with two distinct sub-tasks:
  1. Research: find real reference cases for the design
  2. Generation: produce an actual design sketch (image output)

The task also contains a likely translation artifact — "psychosis of the place" is almost
certainly intended as "genius loci" (spirit/sense of place), a standard architectural concept.
The judge should interpret it charitably as such.

Because this task requires image generation, the judge must check whether an actual
sketch/image was produced — not just a textual description of a design.
"""

import os, json, re

TASK_INSTRUCTION = (
    "You are a graduate student in architecture. Now you want to build a lakeside park "
    "on the Wushan campus of South China University of Technology. It is located on the "
    "central axis of the campus. It covers an area of about 4,000 square meters and has "
    "a height difference of half a meter. Find relevant cases that can be learned and "
    "generate a design sketch. It is required to reflect the psychosis of the place "
    "combined with Lingnan characteristics."
)

PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed an open-ended architectural design task.

Important context:
- "Psychosis of the place" in this task is almost certainly a translation of "genius loci" (spirit/sense of place) — a standard architectural concept referring to the distinctive atmosphere and character of a location. Evaluate it accordingly.
- Lingnan (岭南) refers to the traditional architectural and cultural style of the Guangdong/southern China region, characterised by features like ventilated corridors, water elements, courtyards, and subtropical plant use.
- This task has two distinct sub-tasks: (1) research reference cases, and (2) generate a design sketch. Both must be attempted for a passing score.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Site: Wushan campus of South China University of Technology (SCUT), central axis, lakeside
- Area: ~4,000 square meters with 0.5m height difference
- Style requirement: Lingnan architectural characteristics
- Concept requirement: reflect genius loci (spirit/sense of place) of the site
- Two required outputs: (1) reference case research, (2) a design sketch (visual output)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research reference cases? What cases were found and are they relevant (campus parks, Lingnan architecture, lakeside landscapes)?
- Did the agent produce a design sketch? Is there an image, diagram, or visual output in the response or trace?
- Did the agent address the site constraints (4000sqm, 0.5m height difference, central axis location)?
- Did the agent incorporate Lingnan characteristics? What specific elements were mentioned (e.g. corridors, water features, courtyards)?
- Did the agent address genius loci / sense of place for the SCUT Wushan campus?

### Step 2: Dimension Scoring

#### A. Reference Case Research
Did the agent find and present relevant design reference cases?

5 — 3+ relevant reference cases found (campus parks, Lingnan landscapes, or lakeside parks) with specific project names, locations, and design lessons drawn from each.
4 — 2–3 reference cases found with names and relevance explained; one case is weakly relevant.
3 — 1–2 reference cases found with names; connection to the design task is stated but thin.
2 — Reference cases mentioned vaguely (e.g. "traditional Lingnan gardens") without specific projects.
1 — No reference case research performed; agent jumped directly to design without research.

#### B. Design Sketch Output
Did the agent actually generate a visual design sketch?

5 — A design sketch or diagram was produced (image file, rendered plan, or annotated diagram); it depicts the lakeside park layout.
4 — A sketch was produced but is rough or minimally annotated; clearly an attempt at visual output.
3 — Agent described a design in detail (zones, elements, layout) but produced no visual output.
2 — Agent mentioned it would generate a sketch but produced only text or failed to generate an image.
1 — No design output of any kind; agent refused or only asked clarifying questions.

#### C. Site & Program Responsiveness
Does the design respond to the specific site constraints?

5 — Design explicitly addresses: 4000sqm scale, 0.5m height change (e.g. terracing, ramps), lakeside edge treatment, and central axis relationship.
4 — Design addresses 3 of the 4 site constraints; one is missing or vague.
3 — Design addresses 2 site constraints; scale and height difference are either ignored or only briefly mentioned.
2 — Design is generic; site constraints are acknowledged but not reflected in the design decisions.
1 — Site constraints completely ignored; design could apply to any location.

#### D. Lingnan & Genius Loci Integration
Does the design meaningfully incorporate Lingnan characteristics and sense of place?

5 — Design integrates 3+ specific Lingnan elements (e.g. ventilated corridors, water courtyards, subtropical plantings, grey brick, ornamental screens) AND articulates how the design reflects the campus's genius loci.
4 — 2 Lingnan elements present with explanation; genius loci addressed but superficially.
3 — Lingnan style mentioned and 1 element described; sense of place not explicitly addressed.
2 — Lingnan aesthetics referenced generally (e.g. "traditional southern Chinese style") without specific design elements.
1 — No Lingnan characteristics or genius loci consideration; design is generic or Western in character.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "reference_case_research": <1-5>,
  "design_sketch_output": <1-5>,
  "site_program_responsiveness": <1-5>,
  "lingnan_genius_loci": <1-5>,
  "dimension_reasoning": {{
    "reference_case_research": "<one sentence citing specific evidence>",
    "design_sketch_output": "<one sentence citing specific evidence>",
    "site_program_responsiveness": "<one sentence citing specific evidence>",
    "lingnan_genius_loci": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

# Design Sketch Output is highest weight — producing any visual output is the
# hardest and most distinctive part of this task. Site responsiveness is second
# because the constraints are what make this a specific task, not a generic one.
DIMENSION_WEIGHTS = {
    "reference_case_research":    0.20,
    "design_sketch_output":       0.35,
    "site_program_responsiveness":0.25,
    "lingnan_genius_loci":        0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse_answer_tag(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call_judge_once(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        user_content = USER_PROMPT_TEMPLATE.format(
            task_instruction=TASK_INSTRUCTION,
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            max_tokens=1024,
        )
        return _parse_answer_tag(completion.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def _majority_vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first_call = _call_judge_once(agent_response, execution_summary)
    if first_call and "error" not in first_call:
        overall = first_call.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _majority_vote([first_call, _call_judge_once(agent_response, execution_summary), _call_judge_once(agent_response, execution_summary)])
        else:
            scores = first_call; scores["_votes_used"] = 1
    else:
        scores = first_call or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    feedback_lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: feedback_lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): feedback_lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        feedback_lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): feedback_lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        feedback_lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(feedback_lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }