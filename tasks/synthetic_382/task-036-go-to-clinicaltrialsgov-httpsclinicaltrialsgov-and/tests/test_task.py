"""
LLM-as-judge evaluator for EvolveBench task.

Category: Medical & Clinical & Bio
Task: Evaluate whether the agent successfully identified and extracted details of clinical trials and related publications for CRISPR-based gene therapies for sickle cell disease.
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


TASK_INSTRUCTION = """Go to ClinicalTrials.gov (https://clinicaltrials.gov/) and search for clinical trials investigating CRISPR-based gene therapies for sickle cell disease. Apply filters for trials in Phase 2 or Phase 3 that are currently recruiting and sponsored by either biotechnology companies or universities. Extract details for the first five trials that meet these criteria, including: NCT number, title, sponsor, intervention name, enrollment target, and primary outcome. Then visit PubMed (https://pubmed.ncbi.nlm.nih.gov/) and find the most recent publication (if any) related to each trial's intervention, extracting the PubMed ID and publication year."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to identify clinical trials related to CRISPR-based gene therapies for sickle cell disease, apply specific filters, and extract structured details about the trials. The agent must also find related publications for each trial's intervention using PubMed. A successful completion requires accurate data extraction and adherence to the specified filters.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to ClinicalTrials.gov (https://clinicaltrials.gov/) and search for clinical trials investigating CRISPR-based gene therapies for sickle cell disease. Apply filters for trials in Phase 2 or Phase 3 that are currently recruiting and sponsored by either biotechnology companies or universities. Extract details for the first five trials that meet these criteria, including: NCT number, title, sponsor, intervention name, enrollment target, and primary outcome. Then visit PubMed (https://pubmed.ncbi.nlm.nih.gov/) and find the most recent publication (if any) related to each trial's intervention, extracting the PubMed ID and publication year.

## Task-Specific Constraints
- Must visit both ClinicalTrials.gov and PubMed platforms.
- Must apply filters for Phase 2 or Phase 3 trials that are recruiting.
- Must extract structured details for at least five trials.
- Must find related publications for each trial's intervention.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to ClinicalTrials.gov and PubMed as required?
- Did the agent apply the correct filters for Phase 2 or Phase 3 trials that are recruiting?
- Are details for at least five trials present in the response?
- Are related publications for each trial's intervention included?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the extracted trial details and publication information are correct and complete.

5 — All trial details and publication information are accurate and complete for five trials.
4 — Minor inaccuracies or missing details for one trial.
3 — Partial completion: accurate details for at least three trials.
2 — Significant inaccuracies or missing details for most trials.
1 — No accurate details or publications provided.

#### B. Coverage of Required Platforms and Filters (0.30)
Measures whether the agent visited both platforms and applied the correct filters.

5 — Both platforms visited, and filters applied correctly.
4 — Both platforms visited, but minor filter issues.
3 — One platform visited or filters partially applied.
2 — One platform visited with incorrect filters.
1 — No platforms visited or filters applied.

#### C. Depth and Specificity of Extracted Information (0.25)
Measures the level of detail in the extracted trial and publication information.

5 — Includes all required details (e.g., NCT number, title, sponsor, intervention, enrollment target, primary outcome, PubMed ID, publication year).
4 — Missing one or two minor details.
3 — Missing several details but includes key elements.
2 — Minimal details provided.
1 — No meaningful details provided.

#### D. Output Structure and Organization (0.10)
Measures the clarity and organization of the response.

5 — Output is well-organized as a structured list or table.
4 — Mostly organized but with minor formatting issues.
3 — Partially organized, difficult to interpret in places.
2 — Poorly organized, hard to follow.
1 — Completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms_and_filters": <1-5>,
  "depth_and_specificity_of_extracted_information": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms_and_filters": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_extracted_information": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms_and_filters": 0.30,
    "depth_and_specificity_of_extracted_information": 0.25,
    "output_structure_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())