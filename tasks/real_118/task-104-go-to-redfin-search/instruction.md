Go to Redfin (https://www.redfin.com/) and search for single-family homes for sale in Austin, TX with 3+ bedrooms, 2+ bathrooms, and a price between $400,000 and $600,000. Identify the five listings with the most days on market. For each property, record: address, list price, days on market, square footage, price per square foot, and the listing agent's name. Then look up each property's assessed value and tax history on the Travis County Appraisal District website (https://www.traviscad.org/). Finally, calculate the list-price-to-assessed-value ratio for each property and produce a buyer's market analysis table with columns: Address, List Price, Assessed Value, L/A Ratio, Days on Market, Price/sqft, and a brief "opportunity signal" note (e.g., "listed below assessed value" or "overpriced vs assessment").
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `filters_applied.city` (string) — Should be 'Austin'.
- `filters_applied.beds_min` (integer) — 
- `filters_applied.baths_min` (integer) — 
- `filters_applied.price_min` (integer) — 
- `filters_applied.price_max` (integer) — 
- `filters_applied.property_type` (string) — 'single-family'.
- `listings` (list[object]) — 5 listings, most-days-on-market first.
- `listings[].address` (string) — 
- `listings[].list_price` (integer) — USD.
- `listings[].days_on_market` (integer) — 
- `listings[].sqft` (integer) — 
- `listings[].price_per_sqft` (number) — 
- `listings[].agent_name` (string) — 
- `listings[].redfin_url` (string) — 
- `listings[].traviscad_url` (string) — 
- `listings[].assessed_value` (integer) — From TravisCAD.
- `listings[].la_ratio` (number) — list_price / assessed_value.
- `listings[].opportunity_signal` (string) — 

**Optional but graded if present:**

- `listings[].year_built` (integer) — 
- `listings[].tax_history` (list[object]) — Per-year tax payments.
- `listings[].photos_url` (string) — 
- `methodology_notes` (string) — 

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "filters_applied": {
    "city": "Austin",
    "beds_min": 3,
    "baths_min": 2,
    "price_min": 400000,
    "price_max": 600000,
    "property_type": "single-family"
  },
  "listings": [
    {
      "address": "1234 Oak St, Austin, TX",
      "list_price": 549000,
      "days_on_market": 142,
      "sqft": 2100,
      "price_per_sqft": 261.4,
      "agent_name": "Jane Doe",
      "redfin_url": "https://www.redfin.com/TX/Austin/1234-Oak-St-78704/home/...",
      "traviscad_url": "https://search.traviscad.org/Property/View/...",
      "assessed_value": 575000,
      "la_ratio": 0.955,
      "opportunity_signal": "listed below assessed value"
    }
  ]
}
```
