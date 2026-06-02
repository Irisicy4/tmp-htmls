"""
LLM-as-judge evaluator for EvolveBench task-101.

Category: Legal
Task: Search SEC EDGAR for climate-risk disclosures in S&P 500 energy company 10-K filings,
      cross-reference with CourtListener for related federal lawsuits, and compile a
      structured legal-exposure summary table.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to SEC EDGAR and search for the three most recent 10-K filings from S&P 500 energy "
    "companies that discuss 'climate risk' as a legal or regulatory liability. For each filing, "
    "extract: company name, ticker, filing date, and a verbatim quote of the key climate-risk "
    "disclosure paragraph. Then go to CourtListener and search for any federal lawsuits filed "
    "against those same companies involving climate liability within the past two years; note "
    "the case name, court, and current status for any matches. Compile findings into a structured "
    "legal-exposure summary table with columns: Company, Ticker, 10-K Filing Date, Climate-Risk "
    "Disclosure Quote, Related Lawsuit (if any), Court, Case Status."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a multi-source legal research task combining SEC EDGAR filings and CourtListener case search.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: Must use SEC EDGAR for 10-K filings AND CourtListener for case search (both required)
- Companies: Must be S&P 500 energy sector companies (e.g. ExxonMobil, Chevron, ConocoPhillips, EOG, Pioneer)
- EDGAR: Must retrieve actual filing details (company, ticker, date, verbatim disclosure quote)
- CourtListener: Must perform an actual search and report findings or explicitly state no matches found
- Output: A structured table with all seven specified columns

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate SEC EDGAR and find 10-K filings from energy companies? Cite evidence.
- Are the companies named actual S&P 500 energy sector companies?
- Did the agent extract verbatim climate-risk disclosure quotes from the filings?
- Did the agent search CourtListener for climate liability lawsuits?
- Is there a structured table with the required columns in the final response?

### Step 2: Dimension Scoring

#### A. Source Navigation & Data Retrieval
Did the agent navigate both SEC EDGAR and CourtListener and retrieve real data?

5 — Agent navigated EDGAR and retrieved actual 10-K filings with real data AND searched CourtListener with documented findings.
4 — Agent retrieved data from one source properly; the other source was attempted but incomplete.
3 — Agent retrieved partial data from both sources but with notable gaps (e.g., missing tickers or filing dates).
2 — Agent used only one source or relied heavily on internal knowledge without real navigation.
1 — No evidence of actual navigation; response is from prior knowledge or fabricated.

#### B. Data Accuracy & Specificity
Are the company names, tickers, filing dates, and disclosure quotes specific and plausible?

5 — All three companies are named S&P 500 energy firms with correct tickers, specific filing dates (YYYY-MM-DD), and verbatim or near-verbatim disclosure quotes.
4 — Two of three entries are fully specific; one is missing a detail (e.g., no verbatim quote).
3 — Companies are named but details are vague, approximate, or partially fabricated.
2 — Generic or clearly hallucinated company details with no specific filing evidence.
1 — No specific company data; completely generic or irrelevant response.

#### C. CourtListener Search Quality
Did the agent meaningfully search CourtListener and report findings?

5 — Agent searched CourtListener, found specific cases (or explicitly confirmed no matches with search evidence) and reported case names, courts, and status.
4 — Agent searched CourtListener and reported results but with incomplete case details.
3 — Agent mentioned CourtListener but search results are vague or not clearly tied to the EDGAR companies.
2 — Agent attempted CourtListener but could not retrieve results; no meaningful findings.
1 — CourtListener not used; lawsuit information is fabricated or omitted entirely.

#### D. Output Structure & Completeness
Is the final output a complete, well-structured table with all required columns?

5 — Complete table with all seven columns (Company, Ticker, 10-K Filing Date, Climate-Risk Disclosure Quote, Related Lawsuit, Court, Case Status) for all three companies.
4 — Table present with six of seven columns, or one company entry is incomplete.
3 — Table-like structure but missing two or more columns or one company entirely.
2 — Narrative response with no table; information present but unstructured.
1 — Missing output or completely unstructured response.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "source_navigation": <1-5>,
  "data_accuracy": <1-5>,
  "courtlistener_search": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "source_navigation": "<one sentence citing specific evidence>",
    "data_accuracy": "<one sentence citing specific evidence>",
    "courtlistener_search": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "source_navigation": 0.30,
    "data_accuracy":     0.30,
    "courtlistener_search": 0.20,
    "output_structure":  0.20,
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
