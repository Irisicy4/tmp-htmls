"""
LLM-as-judge evaluator for EvolveBench task-111.

Category: Logistics & Supply Chain
Task: Look up HTS codes and tariff rates for 3 products on USITC HTS, then find one CBP CROSS
      ruling per product. Compile a tariff exposure report.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to the USITC Harmonized Tariff Schedule online database and look up the HTS classification "
    "codes and current duty rates for three products: (1) lithium-ion batteries for electric vehicles, "
    "(2) solar photovoltaic panels, (3) cotton T-shirts. For each product, record the 10-digit HTS "
    "code, general duty rate, and any applicable Section 301 (China) tariff surcharge. Then go to "
    "the CBP CROSS Ruling database and search for binding customs rulings related to each product; "
    "find one relevant ruling per product and note the ruling number, date, and key classification "
    "decision. Compile a tariff exposure report table with columns: Product, HTS Code, Base Duty "
    "Rate, Section 301 Surcharge, Total Effective Rate, CBP Ruling Number, Ruling Date, Key Finding."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a tariff research task using the USITC HTS database and CBP CROSS ruling database.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Sources: USITC HTS (hts.usitc.gov) AND CBP CROSS (rulings.cbp.gov) — both required
- Products: lithium-ion EV batteries, solar PV panels, cotton T-shirts — all three
- HTS data: 10-digit code, general (MFN) duty rate, Section 301 surcharge (if applicable)
- CBP CROSS: one ruling per product (ruling number in format NYxxx or HQxxx or similar)
- Output: Eight-column table with all three products

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate the USITC HTS database and look up codes for all three products?
- Are 10-digit HTS codes provided and plausible for each product?
- Did the agent navigate the CBP CROSS database and find rulings?
- Are ruling numbers in a plausible CBP format (e.g., N123456, HQ123456)?
- Is there a complete eight-column table?

### Step 2: Dimension Scoring

#### A. USITC HTS Navigation & Code Accuracy
Did the agent navigate the USITC HTS database and retrieve plausible codes for all three products?

5 — Navigated hts.usitc.gov (or official HTS tool), retrieved 10-digit codes for all three products that are plausible (e.g., EV batteries ~8507.60.00, solar panels ~8541.40.00, T-shirts ~6109.10.00).
4 — Navigated USITC and retrieved codes for two of three products with 10-digit precision; one product has an 8-digit or less specific code.
3 — Navigated USITC but codes for one or two products appear wrong or are 6-digit only.
2 — HTS codes cited but not clearly from USITC navigation; codes appear to be from prior knowledge.
1 — No HTS database navigation; codes absent or clearly incorrect.

#### B. Tariff Rate Data
Are general duty rate and Section 301 surcharge correctly identified for all three products?

5 — General duty rate and Section 301 surcharge (or explicit confirmation that none applies) for all three products, with specific percentages.
4 — General duty rate and Section 301 info for two of three products; one is incomplete.
3 — General duty rates present for most products but Section 301 surcharges absent or vague.
2 — Duty rates cited for all products but appear to be rough estimates rather than database lookups.
1 — No specific tariff rates; only general descriptions.

#### C. CBP CROSS Ruling Retrieval
Did the agent navigate CBP CROSS and find plausible rulings for all three products?

5 — Navigated rulings.cbp.gov, found one ruling per product with a specific ruling number, date (YYYY-MM-DD format), and key classification decision for all three.
4 — Rulings found for two of three products with specific numbers and dates.
3 — Rulings found for at least two products; ruling numbers present but dates or key findings missing.
2 — CBP CROSS mentioned but rulings found for only one product or ruling numbers appear fabricated.
1 — CBP CROSS not used; ruling information absent or entirely fabricated.

#### D. Output Table Completeness
Is the tariff exposure report table complete with all eight columns for all three products?

5 — Complete eight-column table (Product, HTS Code, Base Duty, Section 301, Total Rate, Ruling Number, Ruling Date, Key Finding) for all three products.
4 — Table with seven of eight columns for all three products.
3 — Table with six or fewer columns, or one product missing from the table.
2 — Partial table (fewer than five columns) or one product missing.
1 — No table structure; narrative only.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "hts_navigation": <1-5>,
  "tariff_rate_data": <1-5>,
  "cross_ruling_retrieval": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "hts_navigation": "<one sentence citing specific evidence>",
    "tariff_rate_data": "<one sentence citing specific evidence>",
    "cross_ruling_retrieval": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "hts_navigation":        0.25,
    "tariff_rate_data":      0.30,
    "cross_ruling_retrieval": 0.25,
    "output_table":          0.20,
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
