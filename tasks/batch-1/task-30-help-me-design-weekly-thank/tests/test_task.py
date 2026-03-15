"""
LLM-as-judge evaluator for EvolveBench task-30.

Category: Design
Task: Design a weekly thank-you card template for employees.
      Must include: completed tasks section, philosophical quote, encouraging message.
      User will fill in content — agent designs the template only.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Help me design a weekly thank-you card template to send to employees. "
    "The card should include: this week's completed tasks + a philosophical quote "
    "+ an encouraging message. (I will fill in the content; mainly help me design the template.)"
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. This is a design task — the agent must produce a visual or well-structured card template. The user explicitly said they will fill in content, so the agent should focus on structure, layout, and visual design rather than writing the actual quote or tasks."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Output: a template (not a filled-in card) — placeholder text for all content sections
- Required sections: (1) completed tasks area, (2) philosophical quote area, (3) encouraging message area
- Format: visual design preferred — HTML/CSS card, image design, or printable template
- User fills content: agent should use placeholders like "[Employee Name]", "[Tasks this week]", "[Quote]"
- The card should look professionally designed, not just a text document

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- What format did the agent use for the template (HTML, image, text, Word doc)?
- Are all 3 required sections present (tasks, quote, encouraging message)?
- Are placeholder labels used rather than filled-in content?
- Is there visual design consideration (layout, colours, typography)?
- Was a file saved/exported?

### Step 2: Dimension Scoring

#### A. Template Structure
Are all 3 required sections present with appropriate placeholders?

5 — All 3 sections present with clear labels and placeholder text; additional elements like employee name, date, sender name also included as placeholders.
4 — All 3 sections present; 1–2 are labelled but missing placeholder text.
3 — 2 of 3 sections present; one section is missing entirely.
2 — Only 1 section present; card is mostly a generic template without required sections.
1 — No structured sections; just a text response.

#### B. Visual Design Quality
Does the template have professional visual design?

5 — Visual card design with: clear layout, typography hierarchy, colour scheme, borders/decorative elements; card looks like a real thank-you card.
4 — Good layout with some visual design; typography or colour is basic but intentional.
3 — Structured format (e.g. HTML table or numbered sections) but minimal visual design.
2 — Plain text with section headers; no visual design consideration.
1 — No design; just a description of what the template should contain.

#### C. Format Appropriateness
Is the output format practical for the use case (sending to employees)?

5 — Format is practical: HTML card (easily emailed), printable PDF template, or image design; reusable without modification.
4 — Format is usable but requires minor adjustment to be practical for sending (e.g. plain HTML without email-friendly styling).
3 — Format works as a reference but not directly usable (e.g. markdown that would need conversion).
2 — Format is inappropriate for the use case (e.g. a code snippet with no visual output).
1 — No usable output format.

#### D. File Saved
Was the template saved as a file or concrete output?

5 — File created and saved (HTML, image, PDF, or doc); file path or confirmation provided.
4 — File creation attempted; trace confirms write but no explicit path in response.
3 — Complete template provided in response that could be directly saved; no file created.
2 — Partial template in response; incomplete for direct use.
1 — No concrete template output; only a description.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "template_structure": <1-5>,
  "visual_design_quality": <1-5>,
  "format_appropriateness": <1-5>,
  "file_saved": <1-5>,
  "dimension_reasoning": {{"template_structure": "<one sentence>", "visual_design_quality": "<one sentence>", "format_appropriateness": "<one sentence>", "file_saved": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"template_structure": 0.30, "visual_design_quality": 0.30, "format_appropriateness": 0.20, "file_saved": 0.20}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    t = result.get("task_result") or ""
    if isinstance(t, str) and t.strip(): return t
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and len(c) > 20: return c
    return ""

def _parse(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except: return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task_instruction=TASK_INSTRUCTION, agent_response=agent_response, execution_summary=execution_summary or "Not available.")}],
            max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid)//2] for d in DIMENSIONS}
    overall = sum(agg[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    agg["overall_score"] = round(overall, 2); agg["passed"] = overall >= PASS_THRESHOLD
    median = sorted(valid, key=lambda v: abs(v.get("overall_score",0)-overall))[0]
    agg["evidence_summary"] = median.get("evidence_summary",""); agg["dimension_reasoning"] = median.get("dimension_reasoning",{}); agg["_votes_used"] = len(valid)
    return agg

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.", "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)]) if abs(float(overall) - PASS_THRESHOLD) <= 0.5 else (first.__setitem__("_votes_used", 1) or first)
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0); passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"] + [f"  {d}: {scores[d]}/5" for d in DIMENSIONS if d in scores]
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    if scores.get("dimension_reasoning"):
        lines.append("\nDimension reasoning:")
        for d, r in scores["dimension_reasoning"].items(): lines.append(f"  {d}: {r}")
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline: {scores['_votes_used']} calls, majority vote)")
    return {"passed": bool(passed), "feedback": "\n".join(lines), "details": {"task_completed": result.get("status") == "success", "overall_score": overall, "dimension_scores": {d: scores.get(d) for d in DIMENSIONS}, "evidence_summary": scores.get("evidence_summary"), "dimension_reasoning": scores.get("dimension_reasoning"), "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)}}