"""
LLM-as-judge evaluator for EvolveBench task-117.

Category: Medical & Clinical & Bio
Task: Look up GLP-1 drugs (semaglutide, liraglutide, tirzepatide) on Drugs@FDA, Drugs.com,
      and FAERS dashboard. Compile a clinical risk summary table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to the FDA's Drugs@FDA database and look up the approval history, approved indications, "
    "and prescribing information for three GLP-1 receptor agonists: semaglutide (Ozempic/Wegovy), "
    "liraglutide (Victoza/Saxenda), and tirzepatide (Mounjaro/Zepbound). For each drug, record: "
    "brand names, approval dates for each indication, manufacturer, and the boxed warning (if any) "
    "from the prescribing information. Then go to Drugs.com and look up the most common adverse "
    "effects and significant drug interactions for each. Finally, check the FDA MedWatch FAERS "
    "public dashboard for the total number of adverse event reports for each drug. Compile a "
    "clinical risk summary table with all eight specified columns."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a multi-source pharmaceutical safety research task using FDA databases, Drugs.com, and FAERS.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: FDA Drugs@FDA (accessdata.fda.gov/scripts/cder/daf/), Drugs.com, AND FDA FAERS dashboard — all three required
- Drugs: semaglutide, liraglutide, AND tirzepatide — all three
- FDA fields: brand names, approval date(s) per indication, manufacturer, boxed warning text/description
- Drugs.com fields: top 3 adverse effects, significant drug interactions
- FAERS: total adverse event report count per drug
- Output: Eight-column table for all three drugs

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Drugs@FDA and look up approval data for all three drugs?
- Are brand names, approval dates, and boxed warnings retrieved for each drug?
- Did the agent access Drugs.com for adverse effects and interactions?
- Did the agent check the FAERS dashboard for adverse event report counts?
- Is there a complete eight-column table?

### Step 2: Dimension Scoring

#### A. Drugs@FDA Navigation & Approval Data
Did the agent navigate Drugs@FDA and retrieve specific approval and labeling data?

5 — Navigated accessdata.fda.gov/scripts/cder/daf/ for all three drugs; retrieved brand names, approval dates (YYYY-MM-DD), manufacturer name, and boxed warning description for each.
4 — Navigated Drugs@FDA for two of three drugs with full data; third drug's data is from a secondary source or has a missing field.
3 — Navigated Drugs@FDA and retrieved approval dates and brand names for all three drugs; boxed warning present for at least two drugs.
2 — Drugs@FDA mentioned; data retrieved for one drug only or data appears to be from prior knowledge.
1 — Drugs@FDA not navigated; all drug approval data is from prior knowledge.

#### B. FDA Data Accuracy
Are brand names, approval dates, manufacturers, and boxed warnings correct for all three drugs?

5 — All three drugs have correct brand names (semaglutide→Ozempic/Wegovy, liraglutide→Victoza/Saxenda, tirzepatide→Mounjaro/Zepbound), plausible approval dates (YYYY format), correct manufacturers (Novo Nordisk, Eli Lilly), and boxed warning descriptions.
4 — Two of three drugs have fully accurate data; one has a minor error (e.g., missing one brand name).
3 — Brand names and manufacturers correct for most drugs; approval dates are year-level only; boxed warning vague for one drug.
2 — Some correct brand names but manufacturer or approval dates are wrong or missing for multiple drugs.
1 — Data is generic or mostly fabricated; no specific approval dates or boxed warning text.

#### C. Drugs.com & FAERS Data
Did the agent retrieve adverse effects and interactions from Drugs.com AND FAERS report counts?

5 — Drugs.com visited; top 3 adverse effects and significant drug interactions retrieved for all three drugs. FAERS dashboard checked; specific adverse event report counts found for all three.
4 — Drugs.com data found for all three drugs; FAERS count found for two of three drugs.
3 — Drugs.com data for at least two drugs; FAERS referenced but counts approximate or from one drug only.
2 — Adverse effects cited for all three drugs from Drugs.com or similar; FAERS not checked or no specific counts.
1 — Neither Drugs.com nor FAERS navigated; all adverse effect and report count data is from prior knowledge.

#### D. Output Table Completeness
Is the clinical risk summary table complete with all eight columns for all three drugs?

5 — Complete eight-column table (Drug, Brand Names, Indication, Approval Date, Boxed Warning, Top 3 Adverse Effects, Key Drug Interactions, FAERS Report Count) for all three drugs.
4 — Table with seven of eight columns for all three drugs.
3 — Table with six or fewer columns or one drug missing from the table.
2 — Partial table with fewer than five columns.
1 — No structured table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "fda_navigation": <1-5>,
  "fda_data_accuracy": <1-5>,
  "drugscom_faers_data": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "fda_navigation": "<one sentence citing specific evidence>",
    "fda_data_accuracy": "<one sentence citing specific evidence>",
    "drugscom_faers_data": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "fda_navigation":     0.25,
    "fda_data_accuracy":  0.30,
    "drugscom_faers_data": 0.25,
    "output_table":       0.20,
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
