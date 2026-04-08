"""
LLM-as-judge evaluator for EvolveBench task-10.

Category: Design (custom rubric — open-ended creative task)
Task: "You are a graduate student in architecture. You want to build a lakeside park on
       the main campus of Tsinghua University. It is located on the central axis of the
       campus. It covers an area of about 4,000 square meters and has a height difference
       of half a meter. Find relevant cases that can be learned and generate a design sketch.
       It is required to reflect the genius loci of the place combined with northern Chinese
       garden characteristics."

Note: This is an open-ended design task with two distinct sub-tasks:
  1. Research: find real reference cases for the design
  2. Generation: produce an actual design sketch (image output)

Genius loci (spirit/sense of place) is a standard architectural concept referring to the
distinctive atmosphere and character of a location. Northern Chinese garden characteristics
include elements such as axial symmetry, imperial/literati garden motifs, stone rockeries,
pavilions, planted pines and willows, and integration with Beijing's cultural landscape.

Because this task requires image generation, the judge must check whether an actual
sketch/image was produced — not just a textual description of a design.
"""

import os, json, re

TASK_INSTRUCTION = (
    "You are a graduate student in architecture. You want to build a lakeside park on the "
    "main campus of Tsinghua University. It is located on the central axis of the campus. "
    "It covers an area of about 4,000 square meters and has a height difference of half a "
    "meter. Find relevant cases that can be learned and generate a design sketch. It is "
    "required to reflect the genius loci of the place combined with northern Chinese garden "
    "characteristics."
)

PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed an open-ended architectural design task.

Important context:
- The site is on the main campus of Tsinghua University (清华大学) in Beijing, located on the central axis — a prestigious and historically significant location.
- Genius loci (场所精神) refers to the distinctive atmosphere and spirit of a place — a standard architectural concept. The design should convey what makes Tsinghua's campus uniquely meaningful.
- Northern Chinese garden characteristics (北方园林特色) include axial symmetry, imperial and literati garden motifs, stone rockeries, pavilions, planted pines and willows, moon gates, and integration with Beijing's cultural and historical landscape.
- This task has two distinct sub-tasks: (1) research reference cases, and (2) generate a design sketch. Both must be attempted for a passing score.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Location: main campus of Tsinghua University (北京, Beijing), central axis, lakeside
- Area: approx. 4,000 square meters, trapezoidal site
- Topography: half-meter height difference across the site
- Design requirement: reflect genius loci of the place combined with northern Chinese garden characteristics
- Deliverables: (1) relevant precedent case studies AND (2) a design sketch/concept (visual output)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research reference cases? What cases were found and are they relevant (campus parks, northern Chinese gardens, lakeside landscapes, Beijing heritage gardens)?
- Did the agent produce a design sketch? Is there an image, diagram, or visual output in the response or trace?
- Did the agent address the site constraints (4000sqm, 0.5m height difference, central axis on Tsinghua main campus)?
- Did the agent incorporate northern Chinese garden characteristics? What specific elements were mentioned (e.g. axial symmetry, pavilions, stone rockeries, pine/willow plantings, moon gates)?
- Did the agent address genius loci / sense of place for Tsinghua's main campus?

### Step 2: Dimension Scoring

#### A. Reference Case Research
Did the agent find and present relevant design reference cases?

5 — 3+ relevant reference cases found (campus parks, northern Chinese gardens, Beijing heritage gardens, or lakeside parks) with specific project names, locations, and design lessons drawn from each.
4 — 2–3 reference cases found with names and relevance explained; one case is weakly relevant.
3 — 1–2 reference cases found with names; connection to the design task is stated but thin.
2 — Reference cases mentioned vaguely (e.g. "traditional northern Chinese gardens") without specific projects.
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

5 — Design explicitly addresses: 4000sqm scale, 0.5m height change (e.g. terracing, ramps, level changes), lakeside edge treatment, and central axis relationship on Tsinghua main campus.
4 — Design addresses 3 of the 4 site constraints; one is missing or vague.
3 — Design addresses 2 site constraints; scale and height difference are either ignored or only briefly mentioned.
2 — Design is generic; site constraints are acknowledged but not reflected in the design decisions.
1 — Site constraints completely ignored; design could apply to any location.

#### D. Northern Chinese Garden & Genius Loci Integration
Does the design meaningfully incorporate northern Chinese garden characteristics and the genius loci of Tsinghua's campus?

5 — Design integrates 3+ specific northern Chinese garden elements (e.g. axial symmetry, pavilions, stone rockeries, pine/willow plantings, moon gates, imperial-style water features) AND articulates how the design reflects Tsinghua campus's genius loci (academic heritage, Beijing cultural landscape).
4 — 2 northern Chinese garden elements present with explanation; genius loci addressed but superficially.
3 — Northern Chinese garden style mentioned and 1 element described; sense of place for Tsinghua not explicitly addressed.
2 — Garden aesthetics referenced generally (e.g. "traditional Chinese garden") without specific northern elements or campus context.
1 — No northern Chinese garden characteristics or genius loci consideration; design is generic or lacks cultural grounding.

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