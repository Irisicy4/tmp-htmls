Go to Airbnb (https://www.airbnb.com/) and search for entire-home listings in Lisbon, Portugal with a capacity of at least 4 guests. Find three active listings with 10+ reviews and record: listing title, nightly rate, occupancy-implied availability (count available nights in the next 30 days from the calendar), number of reviews, and average review score. Then search for the same property type on VRBO (https://www.vrbo.com/) in the same city and find two comparable listings. Finally, visit AirDNA's free market overview (https://www.airdna.co/vacation-rental-data/app/pt/lisbon/lisbon/overview) to note the published average daily rate and occupancy rate for Lisbon. Using all gathered data, produce a vacation rental ROI analysis table comparing each listing on: Platform, Nightly Rate, Estimated Monthly Revenue (nightly rate × occupancy × 30), Review Score, and an ROI feasibility comment.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `airbnb_listings` (list[object]) — Exactly 3 Airbnb listings.
- `airbnb_listings[].title` (string) — 
- `airbnb_listings[].url` (string) — Canonical airbnb.com URL.
- `airbnb_listings[].nightly_rate_usd` (number) — USD/night.
- `airbnb_listings[].available_nights_30d` (integer) — Next-30-day available nights.
- `airbnb_listings[].review_count` (integer) — ≥10.
- `airbnb_listings[].review_score` (number) — Out of 5 (Airbnb) or 10 (VRBO).
- `vrbo_listings` (list[object]) — Exactly 2 VRBO listings.
- `vrbo_listings[].title` (string) — 
- `vrbo_listings[].url` (string) — Canonical vrbo.com URL.
- `vrbo_listings[].nightly_rate_usd` (number) — 
- `airdna.avg_daily_rate_usd` (number) — 
- `airdna.occupancy_rate` (number 0-1) — Decimal fraction.
- `airdna.source_url` (string) — 
- `revenue_table` (list[object]) — ROI rows for the 5 listings.
- `revenue_table[].platform` (string) — 'airbnb'|'vrbo'.
- `revenue_table[].monthly_revenue_usd` (number) — 
- `revenue_table[].roi_comment` (string) — 

**Optional but graded if present:**

- `airbnb_listings[].guests_capacity` (integer) — 
- `airbnb_listings[].entire_home` (boolean) — 
- `vrbo_listings[].review_count` (integer) — 
- `vrbo_listings[].review_score` (number) — 
- `methodology_notes` (string) — 

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "airbnb_listings": [],
  "vrbo_listings": [],
  "airdna": {
    "avg_daily_rate_usd": 162,
    "occupancy_rate": 0.72,
    "source_url": "https://www.airdna.co/vacation-rental-data/app/pt/lisbon/lisbon/overview"
  },
  "revenue_table": []
}
```
