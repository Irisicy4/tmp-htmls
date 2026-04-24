"""LLM-as-judge evaluator for task-36-want-to-buy-an-m5.

Category: Shopping
Task: I want to buy an M5 MacBook Pro with at least 1TB SSD and 16GB RAM. Research apple.com, Amazon, and Best Buy for the best prices and detailed specs. Then open Google Docs, create a new document and ad
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'I want to buy an M5 MacBook Pro with at least 1TB SSD and 16GB RAM. Research apple.com, Amazon, and Best Buy for the best prices and detailed specs. Then open Google Docs, create a new document and add the specs, pricing and other details to it.'

SYSTEM_PROMPT_EXTRA = "You are an expert evaluator for AI agent benchmarks assessing a multi-site product research task followed by Google Docs document creation. Note: As of early 2026, the M5 MacBook Pro may not yet exist — the agent should handle this gracefully (search, note availability status, and use the closest available model if M5 isn't released)."

CONSTRAINTS = """- Specs required: ≥1TB SSD AND ≥16GB RAM (both are hard minimums)
- Sites: apple.com, Amazon, Best Buy — all three must be checked
- Note: M5 MacBook Pro may not be released; agent should note availability and use M4 Pro/M4 Max if necessary
- Output: Google Docs document with specs, pricing from all three sites, and comparison"""

EVIDENCE_QUESTIONS = """- Did the agent visit all 3 sites (apple.com, Amazon, Best Buy)?
- Were ≥1TB/≥16GB models found? Are specs verified?
- Did the agent handle M5 availability (note if not released, use M4)?
- Was a Google Docs document created with the findings?"""

DIMENSION_RUBRICS = """#### A. Multi-Site Research
Did the agent research all three required sites?

5 — All 3 sites (apple.com, Amazon, Best Buy) navigated; prices retrieved from each.
4 — 2 of 3 sites researched; one was inaccessible or skipped.
3 — 1 site researched directly; others referenced from prior knowledge or general search.
2 — None of the 3 sites navigated; response from prior knowledge.
1 — No research performed.

#### B. Spec & Price Accuracy
Are the specifications and prices accurate and correctly filtered?

5 — ≥1TB/≥16GB models identified with correct specs; prices from all available sites; M5 availability correctly handled.
4 — Specs correct but price from 1 site missing or M5 availability not explicitly addressed.
3 — Models found but specs not verified to meet minimums; prices approximate.
2 — Models listed without spec verification; prices not from the required sites.
1 — Specs/prices fabricated or completely wrong.

#### C. Google Docs Save
Was a Google Docs document created with the research?

5 — Google Docs document created; URL or trace confirmation; contains specs, pricing comparison, and summary.
4 — Document creation attempted; trace confirms Google Docs API call; content present.
3 — Agent compiled research in response but did not create Google Docs document.
2 — Agent mentioned creating a doc without doing so.
1 — No document created.

#### D. Information Quality
Is the overall information well-organised and useful for a purchase decision?

5 — Price comparison table across sites; specs clearly listed; best value recommendation given.
4 — Information present but comparison is narrative rather than structured table.
3 — Information present but disorganised; hard to compare options at a glance.
2 — Partial information; missing prices or specs from most sites.
1 — No useful purchase information."""

DIMENSION_WEIGHTS = {
    'multi_site_research': 0.3,
    'spec_price_accuracy': 0.25,
    'google_docs_save': 0.3,
    'information_quality': 0.15,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
