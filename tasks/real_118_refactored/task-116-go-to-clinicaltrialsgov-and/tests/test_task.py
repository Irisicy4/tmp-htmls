import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("Go to ClinicalTrials.gov and search for Phase 2 or Phase 3 clinical trials for Alzheimer's "
    "disease treatments that are currently recruiting, sponsored by a pharmaceutical company, and "
    "started after January 1, 2022. Identify four trials and for each record: NCT number, trial "
    "title, sponsor, intervention (drug name and mechanism), enrollment target, primary endpoint, "
    "and estimated completion date. Then go to PubMed and search for the most recent peer-reviewed "
    "publication related to each drug's mechanism of action; record the PubMed ID, publication year, "
    "and key finding. Compile a clinical pipeline landscape table with all nine specified columns.")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
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
  "dimension_reasoning": {{
    "clinicaltrials_navigation": "<one sentence citing specific evidence>",
    "trial_data_accuracy": "<one sentence citing specific evidence>",
    "pubmed_evidence": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "clinicaltrials_navigation": <1-5>,
  "trial_data_accuracy": <1-5>,
  "pubmed_evidence": <1-5>,
  "output_table": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "clinicaltrials_navigation": 0.35,
    "trial_data_accuracy": 0.28,
    "pubmed_evidence": 0.19,
    "output_table": 0.18,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
