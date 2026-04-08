"""
LLM-as-judge evaluator for EvolveBench task-103.

Category: Legal
Task: Retrieve GDPR Articles 28 and 32 from EUR-Lex, ICO guidance on processor contracts,
      and CNIL processor checklist; compile a GDPR compliance checklist for a B2B SaaS
      data processor with source citations.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Navigate to EUR-Lex and retrieve the full text of GDPR Article 28 (processor obligations) "
    "and Article 32 (security of processing). Then go to the UK ICO website and find their guidance "
    "on data processor contracts and security requirements. Finally, visit the CNIL guidance page "
    "and locate their published checklist for data processors. Using all three sources, compile a "
    "GDPR compliance checklist for a B2B SaaS company acting as a data processor, organized into "
    "sections: Contractual Requirements, Technical Security Measures, Organizational Measures, and "
    "Breach Notification Obligations. Each checklist item must cite its specific source."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a multi-source GDPR legal research task and produced a well-structured compliance checklist.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Must use EUR-Lex AND ICO (ico.org.uk) AND CNIL (cnil.fr) — all three required
- EUR-Lex content: Must reference GDPR Article 28 and Article 32 specifically
- ICO content: Must reference guidance on data processor contracts or security
- CNIL content: Must reference their processor checklist or guidance
- Output: A checklist organized into four named sections with source citations per item
- No fabrication: Checklist items must be traceable to real regulatory text

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate EUR-Lex and retrieve GDPR Article 28 and Article 32?
- Did the agent navigate the ICO website and find processor guidance?
- Did the agent navigate the CNIL website and find their processor checklist?
- Is the output organized into the four required sections?
- Does each checklist item include a source citation?

### Step 2: Dimension Scoring

#### A. Source Coverage
Did the agent access all three required sources (EUR-Lex, ICO, CNIL)?

5 — All three sources accessed; agent cites specific pages or articles from each.
4 — Two of three sources accessed with specific content; third attempted but limited.
3 — Two of three sources accessed; third source not used or not credibly cited.
2 — Only one source accessed; the other two are absent or fabricated.
1 — No credible source navigation; response is entirely from prior knowledge.

#### B. GDPR Article Accuracy
Are GDPR Article 28 and Article 32 correctly identified and accurately summarized?

5 — Both Article 28 (processor contract requirements) and Article 32 (security measures including encryption, pseudonymization, resilience) correctly summarized with specific clause-level detail.
4 — Both articles addressed but one is summarized with minor inaccuracies or missing key obligations.
3 — One article correctly addressed; the other is vague or confused with other articles.
2 — Articles referenced by number but content is generic or largely incorrect.
1 — Articles not identified or content is fabricated.

#### C. Checklist Structure & Completeness
Is the checklist organized into all four required sections with substantive items in each?

5 — All four sections present (Contractual Requirements, Technical Security, Organizational Measures, Breach Notification) with at least three actionable items each.
4 — All four sections present but one section has fewer than two items.
3 — Three of four sections present with adequate items; one section missing.
2 — Checklist exists but organized differently; fewer than two of the four required sections.
1 — No checklist structure; narrative response only.

#### D. Source Citations
Does each checklist item (or group) cite a specific source?

5 — Every checklist item (or section) cites a specific source with article number, guidance section, or page reference.
4 — Most items (>75%) cite a source; a few items lack citations.
3 — Some items (~50%) cite a source; others do not.
2 — A few items cite sources but most are uncited.
1 — No source citations anywhere in the checklist.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "source_coverage": <1-5>,
  "gdpr_article_accuracy": <1-5>,
  "checklist_structure": <1-5>,
  "source_citations": <1-5>,
  "dimension_reasoning": {{
    "source_coverage": "<one sentence citing specific evidence>",
    "gdpr_article_accuracy": "<one sentence citing specific evidence>",
    "checklist_structure": "<one sentence citing specific evidence>",
    "source_citations": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "source_coverage":       0.25,
    "gdpr_article_accuracy": 0.30,
    "checklist_structure":   0.25,
    "source_citations":      0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
    return ""

def _parse(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

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
    except Exception as e:
        return {"error": str(e)}

def _vote(votes):
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
