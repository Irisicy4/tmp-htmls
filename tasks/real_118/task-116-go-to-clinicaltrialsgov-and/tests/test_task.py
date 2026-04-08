"""
LLM-as-judge evaluator for EvolveBench task-116.

Category: Medical & Clinical & Bio
Task: Search ClinicalTrials.gov for Phase 2/3 Alzheimer's trials currently recruiting,
      cross-reference each drug's mechanism with PubMed evidence. Compile a pipeline table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to ClinicalTrials.gov and search for Phase 2 or Phase 3 clinical trials for Alzheimer's "
    "disease treatments that are currently recruiting, sponsored by a pharmaceutical company, and "
    "started after January 1, 2022. Identify four trials and for each record: NCT number, trial "
    "title, sponsor, intervention (drug name and mechanism), enrollment target, primary endpoint, "
    "and estimated completion date. Then go to PubMed and search for the most recent peer-reviewed "
    "publication related to each drug's mechanism of action; record the PubMed ID, publication year, "
    "and key finding. Compile a clinical pipeline landscape table with all nine specified columns."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a clinical research task combining ClinicalTrials.gov and PubMed data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: ClinicalTrials.gov AND PubMed — both required
- Filters: Phase 2 or 3, Alzheimer's disease, currently recruiting, started after 2022-01-01
- Trials: Four trials minimum; NCT numbers must be in format NCTxxxxxxxx
- Required trial fields: NCT number, title, sponsor, drug name + mechanism, enrollment target, primary endpoint, completion date
- PubMed: One publication per drug (most recent relevant paper); PubMed ID (PMID) format
- Output: Nine-column table for all four trials

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate ClinicalTrials.gov and apply correct filters (Phase 2/3, Alzheimer's, recruiting, post-2022)?
- Are four trials identified with NCT numbers in the correct format?
- Did the agent search PubMed for publications related to each drug?
- Are PMIDs provided for each drug's supporting publication?
- Is there a complete nine-column table?

### Step 2: Dimension Scoring

#### A. ClinicalTrials.gov Navigation & Filter Application
Did the agent navigate ClinicalTrials.gov with correct filters and find relevant trials?

5 — Navigated clinicaltrials.gov, applied Phase 2/3 filter, Alzheimer's condition, "currently recruiting" status, and post-2022 start date; found four trials with NCT numbers.
4 — Navigated ClinicalTrials.gov with most filters applied; one filter slightly off or the post-2022 filter not confirmed.
3 — Navigated ClinicalTrials.gov and found Alzheimer's trials but filters partially applied; some trials may be Phase 1 or not recruiting.
2 — ClinicalTrials.gov mentioned; trials cited but appear from prior knowledge rather than current filtered search.
1 — No ClinicalTrials.gov navigation; trials are fabricated or entirely from prior knowledge.

#### B. Trial Data Accuracy
Are NCT numbers, sponsors, drug names, and key trial fields specific and plausible?

5 — All four trials have valid NCT numbers (NCTxxxxxxxx format), named pharmaceutical sponsors, drug names with mechanisms, specific enrollment targets, primary endpoints, and completion dates.
4 — Three of four trials fully detailed; one is missing one field (e.g., enrollment target or endpoint).
3 — All four trials named but two or more fields are generic (e.g., "TBD" or "N/A" for endpoints).
2 — Fewer than three trials have complete data; NCT numbers or drug names appear fabricated.
1 — No specific trial data; all fields are generic or absent.

#### C. PubMed Evidence Retrieval
Did the agent search PubMed and find supporting publications for each drug's mechanism?

5 — PubMed searched for all four drugs; PMID, publication year, and key finding reported for each drug.
4 — PubMed evidence found for three of four drugs with PMIDs.
3 — PubMed evidence found for at least two drugs; PMIDs present for some, key findings described for all.
2 — PubMed referenced for all drugs but no specific PMIDs; findings are generic descriptions of mechanisms.
1 — PubMed not used; publication evidence absent or fabricated.

#### D. Output Table Completeness
Is the clinical pipeline landscape table complete with all nine columns for all four trials?

5 — Complete nine-column table (NCT Number, Drug Name, Mechanism, Sponsor, Phase, Enrollment, Primary Endpoint, Estimated Completion, Supporting PubMed Evidence) for all four trials.
4 — Table with eight of nine columns for all four trials.
3 — Table with seven or fewer columns or one trial missing from the table.
2 — Partial table with fewer than six columns.
1 — No structured table; narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "clinicaltrials_navigation": <1-5>,
  "trial_data_accuracy": <1-5>,
  "pubmed_evidence": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "clinicaltrials_navigation": "<one sentence citing specific evidence>",
    "trial_data_accuracy": "<one sentence citing specific evidence>",
    "pubmed_evidence": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "clinicaltrials_navigation": 0.30,
    "trial_data_accuracy":       0.30,
    "pubmed_evidence":           0.20,
    "output_table":              0.20,
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
