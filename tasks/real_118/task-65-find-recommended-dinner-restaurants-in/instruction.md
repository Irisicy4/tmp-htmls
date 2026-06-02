Find recommended dinner restaurants in Tuscany, Italy that are open on December 26th and 27th.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `location` (string) — Should contain 'Tuscany'.
- `meal_type` (string) — Should be 'dinner'.
- `date_range` (list[string]) — ['2025-12-26','2025-12-27'] or similar (must include both days).
- `restaurants` (list[object]) — Recommended dinner restaurants.
- `restaurants[].name` (string) — 
- `restaurants[].city` (string) — Town within Tuscany.
- `restaurants[].source_url` (string) — TripAdvisor/Google Maps/TheFork URL.
- `restaurants[].open_dec_26` (boolean) — Open dinner Dec 26?
- `restaurants[].open_dec_27` (boolean) — Open dinner Dec 27?
- `restaurants[].verification_method` (string) — How you verified the hours.

**Optional but graded if present:**

- `restaurants[].cuisine` (string) — 
- `restaurants[].price_band` (string) — $/$$/$$$/$$$$.
- `restaurants[].rating` (number 0-5) — 
- `restaurants[].reservation_url` (string) — 
- `sources_used` (list[string]) — All discovery platforms consulted.

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "location": "Tuscany, Italy",
  "meal_type": "dinner",
  "date_range": [
    "2025-12-26",
    "2025-12-27"
  ],
  "restaurants": [
    {
      "name": "Trattoria ...",
      "city": "Florence",
      "source_url": "https://www.tripadvisor.com/...",
      "open_dec_26": true,
      "open_dec_27": true,
      "verification_method": "Site hours page says 'open daily incl. holidays'"
    }
  ]
}
```
