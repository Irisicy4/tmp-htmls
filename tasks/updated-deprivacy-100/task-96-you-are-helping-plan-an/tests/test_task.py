"""
LLM-as-judge evaluator for EvolveBench task-96.

Category: Daily Activities
Task: You are helping plan an upcoming school art festival. Research the Tokyo University
      of the Arts high school art festival and identify one featured or alumni artist from
      their recent exhibitions. Create a Google Document summarizing the festival and the
      artist. In a separate tab, draft a proposal for inviting that artist to host a
      workshop, including a proposed schedule and content outline.
"""

import os, json, re
PASS_THRESHOLD = 3.0

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse(text):
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_INSTRUCTION,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
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
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)])
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }


TASK_INSTRUCTION = """You are helping plan an upcoming school art festival. Research the Tokyo University of the Arts high school art festival and identify one featured or alumni artist from their recent exhibitions. Create a Google Document summarizing the festival and the artist. In a separate tab, draft a proposal for inviting that artist to host a workshop, including a proposed schedule and content outline."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based research and document creation task.

The task involves:
1. Researching the Tokyo University of the Arts (東京藝術大学, TUA) affiliated high school art festival
2. Identifying ONE specific featured or alumni artist discovered through research of recent exhibitions (no artist is pre-specified — the agent must find one)
3. Creating a Google Document with a festival summary and artist profile
4. Adding a separate tab with a workshop proposal (schedule + content outline) for the identified artist

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Institution: Tokyo University of the Arts (東京藝術大学, TUA) affiliated high school
- Task 1: Research the TUA high school art festival (dates, theme, featured works/exhibitions)
- Task 2: Identify ONE specific artist — featured in or notable alumnus from recent TUA exhibitions (must be discovered through research, not assumed)
- Task 3: Create a Google Document with (a) festival summary and (b) identified artist profile
- Task 4: Add a separate tab in the same Google Document with a workshop proposal for the identified artist (must include a proposed schedule and content outline)
- The artist should be real and verifiable from recent TUA-related exhibition research

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent research the TUA affiliated high school art festival? Were specific details found (dates, theme, works)?
- Did the agent identify a specific named artist from recent TUA exhibitions? Is there evidence the artist was discovered through research?
- Was a Google Doc created? Is there a URL or confirmed creation in the trace?
- Does the Doc contain both a festival summary and an artist profile?
- Does the separate tab contain a concrete workshop proposal with schedule and content outline?

### Step 2: Dimension Scoring

#### A. Research Quality (0.25)
Did the agent find real information about the TUA high school art festival and identify a real artist?

5 — Specific findings on the TUA festival (dates, theme, or featured works) AND a named artist identified with evidence linking them to recent TUA exhibitions.
4 — Festival researched and artist named, but one element is thinly supported.
3 — One thoroughly researched; the other vague or inferred without clear sourcing.
2 — Only the festival or only an artist researched; no clear link to TUA exhibitions.
1 — No research; response based on prior knowledge or hallucination.

#### B. Google Doc Creation (0.2)
Was a Google Document actually created?

5 — Google Doc created with confirmed URL or sharing link visible in trace or response.
4 — Google Doc created but link not confirmed or sharing status unclear.
3 — Doc described or partially created; creation not clearly confirmed in trace.
2 — Content written in agent response instead of a Google Doc.
1 — No Google Doc created.

#### C. Summary Completeness (0.25)
Does the Google Doc contain both a festival summary and artist profile?

5 — Both present with specific facts: festival dates/theme/works and artist name/background/notable works.
4 — Both present but one section is thin or lacks specific details.
3 — One section complete; the other minimal or absent.
2 — Only one section present (festival summary or artist profile, not both).
1 — No summaries in the document.

#### D. Proposal Concreteness (0.3)
Does the separate tab contain a concrete workshop proposal with schedule and content outline?

5 — Specific workshop content tailored to the identified artist's practice, duration, proposed schedule (e.g. timed agenda), and content outline for student participants all defined.
4 — Good proposal with schedule and content but one element (e.g. timing detail) is vague.
3 — Proposal present but generic, lacking artist-specific content or a clear schedule.
2 — Vague proposal or only a brief mention without schedule or content outline.
1 — No proposal in the separate tab.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "research_quality": <1-5>,
  "google_doc_creation": <1-5>,
  "summary_completeness": <1-5>,
  "proposal_concreteness": <1-5>,
  "dimension_reasoning": {{
    "research_quality": "<one sentence citing specific evidence>",
    "google_doc_creation": "<one sentence citing specific evidence>",
    "summary_completeness": "<one sentence citing specific evidence>",
    "proposal_concreteness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "research_quality": 0.25,
    "google_doc_creation": 0.2,
    "summary_completeness": 0.25,
    "proposal_concreteness": 0.3,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())